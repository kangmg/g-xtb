"""
g-xTB ASE Calculator

ASE calculator interface for g-xTB v2.0.0+ (general-purpose semiempirical
quantum mechanical method approximating ωB97M-V/def2-TZVPPD properties).

Requires the g-xTB-enabled xtb binary (xtb --gxtb).
"""
import os
import shutil
import subprocess
import tempfile
import warnings
import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr, Debye
from ase.io import write


class gxTB(Calculator):
    """
    ASE calculator for g-xTB v2.0.0+

    g-xTB is a general-purpose semiempirical quantum mechanical method
    approximating ωB97M-V/def2-TZVPPD properties for elements Z=1-103.
    Implemented via tblite and driven by the xtb program with --gxtb flag.

    Parameters
    ----------
    keep_files : bool, default=False
        Keep working directory and output files after calculation.
        When True, the path is printed if verbose=True.
    command : str, default='xtb'
        Path or name of the g-xTB-enabled xtb executable.
    charge : int, optional
        Molecular charge. Overridden by atoms.info['charge'].
    uhf : int, optional
        Number of unpaired electrons. Overridden by atoms.info['uhf'] or
        atoms.info['spin'] (spin is checked when uhf is absent).
    spin : int, optional
        Alias for uhf. Ignored if uhf is also provided.
    verbose : bool, default=False
        Print xtb stdout and working directory path during calculation.
    capture_stdout : bool, default=False
        Store raw xtb stdout in self.stdout after each calculation.
    workdir : str or Path, optional
        Working directory for xtb input/output files. If None (default),
        a fresh temporary directory under the system's temp location is
        created for each calculation and removed afterwards. If specified,
        that directory is reused across calculations and only known output
        files are removed on cleanup (not the directory itself).
    gxtbhome : str or Path, optional
        Path to directory containing g-xTB parameter files (.gxtb, .eeq,
        .basisq). Sets the GXTBHOME environment variable for xtb. Defaults
        to the 'parameters/' directory bundled with this package.
    nprocs : int, default=1
        Number of OpenMP threads for xtb. Passed as ``--parallel N`` and
        also exported as ``OMP_NUM_THREADS=N`` in the subprocess environment.
        For large systems, consider also setting ``OMP_STACKSIZE`` in your
        shell (e.g. ``export OMP_STACKSIZE=4G``) to avoid stack overflows.
    """

    # 'dipole' is intentionally excluded: it is parsed from xtb stdout on a
    # best-effort basis and cannot be guaranteed.  Declaring it here would
    # cause ASE to raise PropertyNotImplementedError whenever the parser
    # finds nothing.  Use get_dipole_moment() to access it when available.
    implemented_properties = ['energy', 'forces', 'charges']

    def __init__(self, keep_files=False, command='xtb', charge=None, uhf=None,
                 spin=None, verbose=False, capture_stdout=False, workdir=None,
                 gxtbhome=None, nprocs=1, **kwargs):
        super().__init__(**kwargs)
        self.keep_files = keep_files
        self.command = command
        self.charge = charge
        self.uhf = uhf if uhf is not None else spin
        self.nprocs = nprocs
        self.verbose = verbose
        self.capture_stdout = capture_stdout
        self.workdir = Path(workdir) if workdir is not None else None
        self.stdout = None
        self._raw_stdout = ''  # always initialized; set by _run_command

        if gxtbhome is not None:
            self.gxtbhome = Path(gxtbhome)
        else:
            # parameters/ bundled inside the package directory
            self.gxtbhome = Path(__file__).parent / 'parameters'

    # ------------------------------------------------------------------
    # Core ASE interface
    # ------------------------------------------------------------------

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Run g-xTB and populate self.results."""
        if properties is None:
            properties = ['energy']
        if atoms is None:
            atoms = self.atoms
        super().calculate(atoms, properties, system_changes)

        charge = self._resolve_charge(atoms)
        uhf = self._resolve_uhf(atoms)
        grad = 'forces' in properties

        work_dir, is_temp = self._make_work_dir()
        try:
            write(str(work_dir / 'mol.xyz'), atoms, format='xyz')
            flags = ['--grad'] if grad else []
            cmd = self._build_command('mol.xyz', charge, uhf, flags)
            self._run_command(cmd, work_dir)
            self._parse_results(atoms, work_dir, parse_forces=grad)
        finally:
            self._cleanup(work_dir, is_temp)

    # ------------------------------------------------------------------
    # Extended properties
    # ------------------------------------------------------------------

    def get_hessian(self, atoms=None) -> np.ndarray:
        """
        Compute and return the full Cartesian Hessian.

        Runs ``xtb --gxtb --hess`` (numerical Hessian using analytic
        gradients) and also updates self.results with energy, forces,
        charges, and dipole moment from the same run.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            If None, uses self.atoms.

        Returns
        -------
        np.ndarray, shape (3N, 3N), units eV/Å²
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided")

        charge = self._resolve_charge(atoms)
        uhf = self._resolve_uhf(atoms)

        work_dir, is_temp = self._make_work_dir()
        try:
            write(str(work_dir / 'mol.xyz'), atoms, format='xyz')
            cmd = self._build_command('mol.xyz', charge, uhf, ['--hess'])
            self._run_command(cmd, work_dir)
            # --hess does not write a gradient file; parse energy/charges/dipole only
            self._parse_results(atoms, work_dir, parse_forces=False)
            hessian = self._parse_hessian(atoms, work_dir)
        finally:
            self._cleanup(work_dir, is_temp)

        return hessian

    def get_dipole_moment(self, atoms=None):
        """
        Return the molecular dipole moment vector parsed from xtb stdout.

        Dipole is populated automatically during any energy or force
        calculation. Call get_potential_energy() (or get_forces()) first,
        then call this method with no arguments to retrieve the cached value.

        If atoms is provided, a fresh single-point calculation is run first.

        Parameters
        ----------
        atoms : ase.Atoms, optional

        Returns
        -------
        np.ndarray, shape (3,), units e·Å

        Raises
        ------
        PropertyNotImplementedError
            If xtb did not print a dipole moment in its output.
        """
        from ase.calculators.calculator import PropertyNotImplementedError

        if atoms is not None:
            self.calculate(atoms, ['energy'])

        if 'dipole' not in self.results:
            raise PropertyNotImplementedError(
                "Dipole moment was not found in the xtb output. "
                "Ensure a calculation has been run first "
                "(e.g. atoms.get_potential_energy()), and that "
                "xtb --gxtb prints 'molecular dipole' for your system."
            )
        return self.results['dipole']

    # ------------------------------------------------------------------
    # Working directory management
    # ------------------------------------------------------------------

    def _make_work_dir(self):
        """
        Return (work_dir: Path, is_temp: bool).

        If self.workdir is None, create a fresh temp dir under the
        system temp location. Otherwise create/reuse self.workdir.
        """
        if self.workdir is None:
            return Path(tempfile.mkdtemp(prefix='gxtb_')), True
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir, False

    def _cleanup(self, work_dir, is_temp):
        if self.keep_files:
            if self.verbose:
                print(f"Work directory kept: {work_dir}")
            return
        if is_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            self._remove_known_files(work_dir)

    def _remove_known_files(self, work_dir):
        """Remove known xtb output files from an explicit directory."""
        known = [
            'mol.xyz',
            'energy', 'gradient', 'charges', 'hessian',
            'xtbrestart', 'wbo', 'bond_orders',
            'vibspectrum', 'g98.out', 'molden.input',
            'xtbopt.xyz', 'xtbopt.log', 'xtbhess.xyz', '.xtboptok',
            'xtbtopo.mol', 'gfnff_adjacency', 'gfnff_topo',
            '.CHRG', '.UHF', 'coord', 'gxtbrestart',
        ]
        for fname in known:
            f = work_dir / fname
            if f.exists():
                try:
                    f.unlink()
                    if self.verbose:
                        print(f"Removed: {f}")
                except OSError:
                    pass

    # kept for backward compatibility
    def clear_files(self):
        """Remove known xtb output files from self.workdir (if set)."""
        if self.workdir is not None:
            self._remove_known_files(self.workdir)

    # ------------------------------------------------------------------
    # Parameter resolution helpers
    # ------------------------------------------------------------------

    def _resolve_charge(self, atoms):
        charge = atoms.info.get('charge', self.charge)
        if charge is None:
            return None
        if int(charge) != charge:
            raise ValueError(
                f"Molecular charge must be an integer, got {charge!r}"
            )
        return int(charge)

    def _resolve_uhf(self, atoms):
        # Priority: atoms.info['uhf'] > atoms.info['spin'] > self.uhf
        uhf = atoms.info.get('uhf', atoms.info.get('spin', self.uhf))
        if uhf is None:
            return None
        if int(uhf) != uhf:
            raise ValueError(
                f"uhf/spin (unpaired electrons) must be an integer, got {uhf!r}"
            )
        return int(uhf)

    # ------------------------------------------------------------------
    # Command building and execution
    # ------------------------------------------------------------------

    def _build_command(self, coord_file, charge, uhf, extra_flags=None):
        cmd = [self.command, coord_file, '--gxtb']
        if charge is not None:
            cmd += ['--chrg', str(charge)]
        if uhf is not None:
            cmd += ['--uhf', str(uhf)]
        if self.nprocs > 1:
            cmd += ['--parallel', str(self.nprocs)]
        if extra_flags:
            cmd += extra_flags
        return cmd

    def _run_command(self, cmd, work_dir):
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nprocs)
        if self.gxtbhome.exists():
            env['GXTBHOME'] = str(self.gxtbhome)
        else:
            warnings.warn(
                f"g-xTB parameter directory not found: {self.gxtbhome}. "
                "xtb will use its default parameter location, which may "
                "not have the g-xTB parameters. Run gxtb_install() to "
                "download the parameter files.",
                RuntimeWarning,
                stacklevel=3,
            )

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=work_dir, env=env,
        )
        self._raw_stdout = result.stdout

        if self.verbose:
            print(result.stdout)
        if self.capture_stdout:
            self.stdout = result.stdout

        if result.returncode != 0:
            raise RuntimeError(
                f"xtb --gxtb failed (exit {result.returncode}):\n"
                f"{result.stderr}\n{result.stdout}"
            )

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(self, atoms, work_dir, parse_forces=False):
        self.results['energy'] = self._parse_energy(work_dir)

        if parse_forces:
            grad_file = work_dir / 'gradient'
            if not grad_file.exists():
                raise RuntimeError(
                    "xtb ran successfully but did not produce a 'gradient' "
                    "file. This is unexpected; check the xtb output above."
                )
            self.results['forces'] = self._parse_forces(atoms, work_dir)

        if (work_dir / 'charges').exists():
            self.results['charges'] = self._parse_charges(atoms, work_dir)

        dipole = self._parse_dipole_from_stdout(self._raw_stdout)
        if dipole is not None:
            self.results['dipole'] = dipole

    def _parse_energy(self, work_dir):
        """Parse total energy in eV; tries 'energy' file then stdout."""
        energy_file = work_dir / 'energy'
        if energy_file.exists():
            try:
                return self._parse_energy_file(energy_file)
            except Exception:
                pass
        return self._parse_energy_stdout()

    def _parse_energy_file(self, energy_file):
        with open(energy_file) as f:
            lines = f.readlines()
        in_section = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('$energy'):
                in_section = True
                continue
            if in_section:
                if stripped.startswith('$'):
                    break
                parts = stripped.split()
                if len(parts) >= 2:
                    return float(parts[1]) * Hartree
        raise RuntimeError("Could not parse energy from 'energy' file")

    def _parse_energy_stdout(self):
        if not self._raw_stdout:
            raise RuntimeError("No stdout captured; cannot parse energy")
        for line in self._raw_stdout.splitlines():
            if 'TOTAL ENERGY' in line:
                parts = line.split()
                try:
                    idx = parts.index('ENERGY')
                    return float(parts[idx + 1]) * Hartree
                except (ValueError, IndexError):
                    pass
        raise RuntimeError("Could not parse TOTAL ENERGY from xtb stdout")

    def _parse_forces(self, atoms, work_dir):
        """
        Parse analytic forces from the 'gradient' file.

        The gradient file is turbomole format: coordinates in Bohr then
        gradients in Hartree/Bohr. Forces = -gradient, converted to eV/Å.
        """
        with open(work_dir / 'gradient') as f:
            lines = f.readlines()

        in_grad = False
        coords_done = False
        coord_count = 0
        gradients = []
        n_atoms = len(atoms)

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('$grad'):
                in_grad = True
                continue
            if not in_grad:
                continue
            if stripped.startswith('$end') or (stripped.startswith('$') and 'grad' not in stripped):
                break
            if not stripped or stripped.startswith('cycle') or 'SCF energy' in stripped:
                continue

            parts = stripped.split()

            if not coords_done:
                # Coordinate lines: x y z element (4 tokens, last is non-numeric)
                if len(parts) == 4:
                    try:
                        float(parts[0]); float(parts[1]); float(parts[2])
                        coord_count += 1
                        if coord_count == n_atoms:
                            coords_done = True
                        continue
                    except ValueError:
                        pass

            # Gradient lines: dx dy dz (3 tokens, may use Fortran D notation)
            if len(parts) == 3:
                try:
                    gx = float(parts[0].replace('D', 'E').replace('d', 'e'))
                    gy = float(parts[1].replace('D', 'E').replace('d', 'e'))
                    gz = float(parts[2].replace('D', 'E').replace('d', 'e'))
                    gradients.append([gx, gy, gz])
                except ValueError:
                    continue

        if len(gradients) != n_atoms:
            raise RuntimeError(f"Expected {n_atoms} gradients, got {len(gradients)}")

        # F = -dE/dR; convert Hartree/Bohr → eV/Å
        return np.array(gradients) * (-Hartree / Bohr)

    def _parse_charges(self, atoms, work_dir):
        """Parse partial charges from the 'charges' file (one float per line)."""
        with open(work_dir / 'charges') as f:
            values = [float(ln) for ln in f if ln.strip()]
        if len(values) != len(atoms):
            raise RuntimeError(
                f"Expected {len(atoms)} charges, got {len(values)}"
            )
        return np.array(values)

    def _parse_dipole_from_stdout(self, stdout):
        """
        Parse molecular dipole moment from xtb stdout.

        Returns the [x, y, z] vector in ASE units (e·Å), or None if not found.
        """
        if not stdout:
            return None
        in_dipole = False
        for line in stdout.splitlines():
            lower = line.lower()
            if 'molecular dipole' in lower:
                in_dipole = True
                continue
            if not in_dipole:
                continue
            # 'full:' line has x y z [tot] in Debye
            if 'full:' in lower:
                parts = line.split()
                try:
                    idx = next(i for i, p in enumerate(parts) if 'full' in p.lower())
                    x = float(parts[idx + 1])
                    y = float(parts[idx + 2])
                    z = float(parts[idx + 3])
                    return np.array([x, y, z]) * Debye
                except (StopIteration, IndexError, ValueError):
                    pass
            # Older xtb versions use 'q+calc:' for the full dipole
            if 'q+calc:' in lower:
                parts = line.split()
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    return np.array([x, y, z]) * Debye
                except (IndexError, ValueError):
                    pass
        return None

    def _parse_hessian(self, atoms, work_dir):
        """
        Parse Cartesian Hessian from the 'hessian' file.

        The file uses turbomole $hessian format with values in Hartree/Bohr².
        Returns a (3N, 3N) array in eV/Å².
        """
        hessian_file = work_dir / 'hessian'
        if not hessian_file.exists():
            raise RuntimeError("'hessian' file not found after --hess run")

        with open(hessian_file) as f:
            content = f.read()

        values = []
        in_section = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('$hessian'):
                in_section = True
                continue
            if in_section:
                if stripped.startswith('$'):
                    break
                if stripped:
                    try:
                        values.extend(float(v) for v in stripped.split())
                    except ValueError as e:
                        raise RuntimeError(
                            f"Hessian: could not parse numeric value in line "
                            f"{stripped!r}: {e}"
                        ) from e

        n = len(atoms)
        expected = (3 * n) ** 2
        if len(values) != expected:
            raise RuntimeError(
                f"Hessian: expected {expected} elements for {n} atoms, got {len(values)}"
            )

        H = np.array(values).reshape(3 * n, 3 * n)
        return H * (Hartree / Bohr ** 2)  # Hartree/Bohr² → eV/Å²
