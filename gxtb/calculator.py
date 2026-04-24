"""
g-xTB ASE Calculator

ASE calculator interface for g-xTB v2.0.0+ (general-purpose semiempirical
quantum mechanical method approximating ωB97M-V/def2-TZVPPD properties).

Requires the g-xTB-enabled xtb binary (xtb --gxtb).
"""
import os
import subprocess
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
        Keep temporary xtb output files after calculation.
    command : str, default='xtb'
        Path or name of the g-xTB-enabled xtb executable.
    charge : int, optional
        Molecular charge. Overridden by atoms.info['charge'].
    uhf : int, optional
        Number of unpaired electrons. Overridden by atoms.info['uhf'].
    verbose : bool, default=False
        Print xtb stdout during calculation.
    capture_stdout : bool, default=False
        Store raw xtb stdout in self.stdout after each calculation.
    gxtbhome : str or Path, optional
        Path to directory containing g-xTB parameter files (.gxtb, .eeq,
        .basisq). Sets the GXTBHOME environment variable for xtb. Defaults
        to the 'parameters/' directory bundled with this package.
    """

    implemented_properties = ['energy', 'forces', 'charges', 'dipole']

    def __init__(self, keep_files=False, command='xtb', charge=None, uhf=None,
                 verbose=False, capture_stdout=False, gxtbhome=None, **kwargs):
        super().__init__(**kwargs)
        self.keep_files = keep_files
        self.command = command
        self.charge = charge
        self.uhf = uhf
        self.verbose = verbose
        self.capture_stdout = capture_stdout
        self.stdout = None

        if gxtbhome is not None:
            self.gxtbhome = Path(gxtbhome)
        else:
            # parameters/ directory bundled two levels up from this file
            self.gxtbhome = Path(__file__).parent.parent / 'parameters'

    # ------------------------------------------------------------------
    # Core ASE interface
    # ------------------------------------------------------------------

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Run g-xTB and populate self.results."""
        if atoms is None:
            atoms = self.atoms
        super().calculate(atoms, properties, system_changes)

        non_position_changes = {'numbers', 'cell', 'pbc', 'initial_charges', 'initial_magmoms'}
        if non_position_changes & set(system_changes):
            self.clear_files()

        coord_file = 'TMP_gxtb.xyz'
        write(coord_file, atoms, format='xyz')

        charge = atoms.info.get('charge', self.charge)
        uhf = atoms.info.get('uhf', self.uhf)
        grad = 'forces' in properties

        try:
            cmd = self._build_command(coord_file, charge, uhf, ['--grad'] if grad else [])
            self._run_command(cmd)
            self._parse_results(atoms, parse_forces=grad)
        finally:
            if not self.keep_files:
                self.clear_files()

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

        coord_file = 'TMP_gxtb.xyz'
        write(coord_file, atoms, format='xyz')

        charge = atoms.info.get('charge', self.charge)
        uhf = atoms.info.get('uhf', self.uhf)
        cmd = self._build_command(coord_file, charge, uhf, ['--hess'])

        try:
            self._run_command(cmd)
            self._parse_results(atoms, parse_forces=True)
            hessian = self._parse_hessian(atoms)
        finally:
            if not self.keep_files:
                self.clear_files()

        return hessian

    # ------------------------------------------------------------------
    # Command building and execution
    # ------------------------------------------------------------------

    def _build_command(self, coord_file, charge, uhf, extra_flags=None):
        cmd = [self.command, coord_file, '--gxtb']
        if charge is not None:
            cmd += ['--chrg', str(int(charge))]
        if uhf is not None:
            cmd += ['--uhf', str(int(uhf))]
        if extra_flags:
            cmd += extra_flags
        return cmd

    def _run_command(self, cmd):
        env = os.environ.copy()
        if self.gxtbhome and self.gxtbhome.exists():
            env['GXTBHOME'] = str(self.gxtbhome)

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        raw_stdout = result.stdout

        if self.verbose:
            print(raw_stdout)
        # Always keep stdout available for internal parsing; expose to user
        # only when capture_stdout=True.
        self._raw_stdout = raw_stdout
        if self.capture_stdout:
            self.stdout = raw_stdout

        if result.returncode != 0:
            raise RuntimeError(
                f"xtb --gxtb failed (exit {result.returncode}):\n"
                f"{result.stderr}\n{result.stdout}"
            )

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(self, atoms, parse_forces=False):
        self.results['energy'] = self._parse_energy()
        if parse_forces and os.path.exists('gradient'):
            self.results['forces'] = self._parse_forces(atoms)
        if os.path.exists('charges'):
            self.results['charges'] = self._parse_charges(atoms)
        dipole = self._parse_dipole_from_stdout(self._raw_stdout)
        if dipole is not None:
            self.results['dipole'] = dipole

    def _parse_energy(self):
        """Parse total energy in eV; tries 'energy' file then stdout."""
        if os.path.exists('energy'):
            try:
                return self._parse_energy_file()
            except Exception:
                pass
        return self._parse_energy_stdout()

    def _parse_energy_file(self):
        with open('energy') as f:
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

    def _parse_forces(self, atoms):
        """
        Parse analytic forces from the 'gradient' file.

        The gradient file is turbomole format: coordinates in Bohr then
        gradients in Hartree/Bohr. Forces = -gradient, converted to eV/Å.
        """
        with open('gradient') as f:
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

    def _parse_charges(self, atoms):
        """Parse partial charges from the 'charges' file (one float per line)."""
        with open('charges') as f:
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

    def _parse_hessian(self, atoms):
        """
        Parse Cartesian Hessian from the 'hessian' file.

        The file uses turbomole $hessian format with values in Hartree/Bohr².
        Returns a (3N, 3N) array in eV/Å².
        """
        if not os.path.exists('hessian'):
            raise RuntimeError("'hessian' file not found after --hess run")

        with open('hessian') as f:
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
                    values.extend(float(v) for v in stripped.split())

        n = len(atoms)
        expected = (3 * n) ** 2
        if len(values) != expected:
            raise RuntimeError(
                f"Hessian: expected {expected} elements for {n} atoms, got {len(values)}"
            )

        H = np.array(values).reshape(3 * n, 3 * n)
        return H * (Hartree / Bohr ** 2)  # Hartree/Bohr² → eV/Å²

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_files(self):
        """Remove temporary xtb output files. Can be called manually."""
        temp_files = [
            'TMP_gxtb.xyz',
            # xtb standard outputs
            'energy', 'gradient', 'charges', 'hessian',
            'xtbrestart', 'wbo', 'bond_orders',
            'vibspectrum', 'g98.out', 'molden.input',
            'xtbopt.xyz', 'xtbopt.log', 'xtbhess.xyz', '.xtboptok',
            'gfnff_adjacency', 'gfnff_topo',
            # legacy gxtb files (kept for v1 backward compat)
            '.CHRG', '.UHF', 'coord', 'gxtbrestart',
        ]
        for fname in temp_files:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                    if self.verbose:
                        print(f"Removed: {fname}")
                except OSError:
                    pass
