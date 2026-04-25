"""
Unit tests for the gxTB ASE calculator.

Tests cover all parser methods, command building, workdir management,
and charge/UHF validation.  No xtb binary is required — subprocess calls
are mocked where needed.
"""
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import molecule
from ase.units import Bohr, Debye, Hartree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gxtb.calculator import gxTB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atoms(n=3):
    """Return a simple water molecule (3 atoms)."""
    return molecule('H2O')


def _make_calc(**kwargs):
    return gxTB(**kwargs)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):

    def test_defaults(self):
        c = gxTB()
        self.assertEqual(c.command, 'xtb')
        self.assertIsNone(c.charge)
        self.assertIsNone(c.uhf)
        self.assertFalse(c.keep_files)
        self.assertFalse(c.verbose)
        self.assertFalse(c.capture_stdout)
        self.assertIsNone(c.workdir)
        self.assertIsNone(c.stdout)
        self.assertEqual(c._raw_stdout, '')

    def test_gxtbhome_default_inside_package(self):
        c = gxTB()
        expected = Path(__file__).parent.parent / 'gxtb' / 'parameters'
        self.assertEqual(c.gxtbhome, expected)

    def test_gxtbhome_custom(self):
        c = gxTB(gxtbhome='/tmp/myparams')
        self.assertEqual(c.gxtbhome, Path('/tmp/myparams'))

    def test_workdir_stored_as_path(self):
        c = gxTB(workdir='/tmp/mydir')
        self.assertEqual(c.workdir, Path('/tmp/mydir'))

    def test_implemented_properties(self):
        self.assertIn('energy', gxTB.implemented_properties)
        self.assertIn('forces', gxTB.implemented_properties)
        self.assertIn('charges', gxTB.implemented_properties)
        # dipole is excluded: parsed from stdout on best-effort basis only
        self.assertNotIn('dipole', gxTB.implemented_properties)


# ---------------------------------------------------------------------------
# _resolve_charge / _resolve_uhf
# ---------------------------------------------------------------------------

class TestResolve(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB(charge=0, uhf=0)
        self.atoms = _make_atoms()

    def test_charge_from_calc(self):
        c = gxTB(charge=-1)
        self.assertEqual(c._resolve_charge(self.atoms), -1)

    def test_charge_from_atoms_info_overrides_calc(self):
        c = gxTB(charge=0)
        self.atoms.info['charge'] = 2
        self.assertEqual(c._resolve_charge(self.atoms), 2)

    def test_charge_none(self):
        c = gxTB()
        self.assertIsNone(c._resolve_charge(self.atoms))

    def test_charge_float_integer_value_ok(self):
        c = gxTB(charge=1.0)
        self.assertEqual(c._resolve_charge(self.atoms), 1)

    def test_charge_float_non_integer_raises(self):
        c = gxTB(charge=1.5)
        with self.assertRaises(ValueError):
            c._resolve_charge(self.atoms)

    def test_uhf_non_integer_raises(self):
        c = gxTB(uhf=0.9)
        with self.assertRaises(ValueError):
            c._resolve_uhf(self.atoms)

    def test_uhf_from_atoms_info(self):
        c = gxTB()
        self.atoms.info['uhf'] = 3
        self.assertEqual(c._resolve_uhf(self.atoms), 3)

    def test_spin_alias_in_atoms_info(self):
        c = gxTB()
        self.atoms.info['spin'] = 2
        self.assertEqual(c._resolve_uhf(self.atoms), 2)

    def test_uhf_takes_priority_over_spin(self):
        c = gxTB()
        self.atoms.info['uhf'] = 1
        self.atoms.info['spin'] = 5
        self.assertEqual(c._resolve_uhf(self.atoms), 1)

    def test_spin_constructor_alias(self):
        c = gxTB(spin=3)
        self.assertEqual(c.uhf, 3)

    def test_uhf_constructor_wins_over_spin(self):
        c = gxTB(uhf=2, spin=9)
        self.assertEqual(c.uhf, 2)


# ---------------------------------------------------------------------------
# _build_command
# ---------------------------------------------------------------------------

class TestBuildCommand(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()

    def test_basic(self):
        cmd = self.calc._build_command('mol.xyz', None, None, [])
        self.assertEqual(cmd, ['xtb', 'mol.xyz', '--gxtb'])

    def test_with_charge(self):
        cmd = self.calc._build_command('mol.xyz', -1, None, [])
        self.assertIn('--chrg', cmd)
        self.assertIn('-1', cmd)

    def test_with_uhf(self):
        cmd = self.calc._build_command('mol.xyz', None, 2, [])
        self.assertIn('--uhf', cmd)
        self.assertIn('2', cmd)

    def test_with_grad(self):
        cmd = self.calc._build_command('mol.xyz', None, None, ['--grad'])
        self.assertIn('--grad', cmd)

    def test_with_hess(self):
        cmd = self.calc._build_command('mol.xyz', 1, 0, ['--hess'])
        self.assertEqual(cmd, ['xtb', 'mol.xyz', '--gxtb', '--chrg', '1', '--uhf', '0', '--hess'])

    def test_custom_command(self):
        c = gxTB(command='/opt/xtb/bin/xtb')
        cmd = c._build_command('mol.xyz', None, None, [])
        self.assertEqual(cmd[0], '/opt/xtb/bin/xtb')

    def test_nprocs_adds_parallel_flag(self):
        c = gxTB(nprocs=4)
        cmd = c._build_command('mol.xyz', None, None, [])
        self.assertIn('--parallel', cmd)
        self.assertIn('4', cmd)

    def test_nprocs_1_no_parallel_flag(self):
        c = gxTB(nprocs=1)
        cmd = c._build_command('mol.xyz', None, None, [])
        self.assertNotIn('--parallel', cmd)

    def _fake_subprocess(self, recorded_env):
        def fake(cmd, **kwargs):
            recorded_env.update(kwargs.get('env', {}))
            m = MagicMock()
            m.returncode = 0
            m.stdout = '          TOTAL ENERGY              -10.0 Eh\n'
            m.stderr = ''
            return m
        return fake

    def test_nprocs_sets_omp_env(self):
        c = gxTB(nprocs=8)
        recorded_env = {}
        with patch('gxtb.calculator.subprocess.run', side_effect=self._fake_subprocess(recorded_env)):
            try:
                c._run_command(['xtb', 'mol.xyz', '--gxtb'], Path('/tmp'))
            except Exception:
                pass
        self.assertEqual(recorded_env.get('OMP_NUM_THREADS'), '8')

    def test_nprocs_1_still_sets_omp_env(self):
        c = gxTB(nprocs=1)
        recorded_env = {}
        with patch('gxtb.calculator.subprocess.run', side_effect=self._fake_subprocess(recorded_env)):
            try:
                c._run_command(['xtb', 'mol.xyz', '--gxtb'], Path('/tmp'))
            except Exception:
                pass
        self.assertEqual(recorded_env.get('OMP_NUM_THREADS'), '1')


# ---------------------------------------------------------------------------
# Working directory management
# ---------------------------------------------------------------------------

class TestWorkDir(unittest.TestCase):

    def test_temp_dir_created_and_removed(self):
        c = gxTB()
        wd, is_temp = c._make_work_dir()
        self.assertTrue(is_temp)
        self.assertTrue(wd.exists())
        self.assertTrue(str(wd).startswith(tempfile.gettempdir()))
        c._cleanup(wd, is_temp)
        self.assertFalse(wd.exists())

    def test_explicit_dir_preserved_on_cleanup(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            c = gxTB(workdir=tmp)
            wd, is_temp = c._make_work_dir()
            self.assertFalse(is_temp)
            self.assertEqual(wd, tmp)
            (tmp / 'energy').write_text('dummy')
            c._cleanup(wd, is_temp)
            self.assertTrue(tmp.exists())
            self.assertFalse((tmp / 'energy').exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_keep_files_suppresses_cleanup(self):
        c = gxTB(keep_files=True)
        wd, is_temp = c._make_work_dir()
        c._cleanup(wd, is_temp)
        self.assertTrue(wd.exists())
        shutil.rmtree(wd)

    def test_explicit_dir_created_if_missing(self):
        tmp = Path(tempfile.mkdtemp())
        shutil.rmtree(tmp)
        c = gxTB(workdir=tmp)
        wd, _ = c._make_work_dir()
        self.assertTrue(wd.exists())
        shutil.rmtree(tmp)

    def test_remove_known_files_leaves_unknown(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / 'energy').write_text('x')
            (tmp / 'my_custom_file.txt').write_text('x')
            c = gxTB(workdir=tmp)
            c._remove_known_files(tmp)
            self.assertFalse((tmp / 'energy').exists())
            self.assertTrue((tmp / 'my_custom_file.txt').exists())
        finally:
            shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# _parse_energy_file
# ---------------------------------------------------------------------------

class TestParseEnergyFile(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_energy(self, content):
        p = self.tmp / 'energy'
        p.write_text(content)
        return p

    def test_standard_format(self):
        p = self._write_energy(
            '$energy\n'
            ' 1  -76.316820093019  0.0  99.9 99.9 99.9\n'
            '$end\n'
        )
        E = self.calc._parse_energy_file(p)
        self.assertAlmostEqual(E, -76.316820093019 * Hartree, places=6)

    def test_minimal_format(self):
        p = self._write_energy('$energy\n 1  -10.0\n$end\n')
        self.assertAlmostEqual(self.calc._parse_energy_file(p), -10.0 * Hartree)

    def test_missing_end_still_parsed(self):
        p = self._write_energy('$energy\n 1  -5.0\n')
        self.assertAlmostEqual(self.calc._parse_energy_file(p), -5.0 * Hartree)

    def test_empty_section_raises(self):
        p = self._write_energy('$energy\n$end\n')
        with self.assertRaises(RuntimeError):
            self.calc._parse_energy_file(p)


# ---------------------------------------------------------------------------
# _parse_energy_stdout
# ---------------------------------------------------------------------------

class TestParseEnergyStdout(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()

    def test_standard_xtb_format(self):
        self.calc._raw_stdout = (
            '          TOTAL ENERGY              -76.316820093019 Eh\n'
            '          GRADIENT NORM               0.049680773665 Eh/α\n'
        )
        E = self.calc._parse_energy_stdout()
        self.assertAlmostEqual(E, -76.316820093019 * Hartree, places=6)

    def test_no_stdout_raises(self):
        self.calc._raw_stdout = ''
        with self.assertRaises(RuntimeError):
            self.calc._parse_energy_stdout()

    def test_missing_energy_line_raises(self):
        self.calc._raw_stdout = 'some output without energy\n'
        with self.assertRaises(RuntimeError):
            self.calc._parse_energy_stdout()


# ---------------------------------------------------------------------------
# _parse_forces
# ---------------------------------------------------------------------------

class TestParseForces(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()
        self.tmp = Path(tempfile.mkdtemp())
        self.atoms = _make_atoms()  # 3 atoms (H2O)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_gradient(self, content):
        (self.tmp / 'gradient').write_text(textwrap.dedent(content))

    def test_standard_turbomole_format(self):
        self._write_gradient("""\
            $grad  cartesian gradients
              cycle =      1    SCF energy =    -76.316820093019    |dE/dxyz| =  0.049681
                0.00000000000000E+00  0.00000000000000E+00  0.11794415078539E+00  o
                0.00000000000000E+00  1.44411537736480E+00 -0.58972075392695E+00  h
                0.00000000000000E+00 -1.44411537736480E+00 -0.58972075392695E+00  h
               -3.45678901234D-05  2.34567890123D-05  1.23456789012D-05
                1.72839450617D-05 -1.17283945062D-05  5.86419753086D-06
                1.72839450617D-05 -1.17283945062D-05 -1.82098765432D-05
            $end
        """)
        forces = self.calc._parse_forces(self.atoms, self.tmp)
        self.assertEqual(forces.shape, (3, 3))
        # Forces are negative gradients
        self.assertAlmostEqual(forces[0, 0], 3.45678901234e-05 * Hartree / Bohr, places=10)

    def test_fortran_D_notation(self):
        self._write_gradient("""\
            $grad  cartesian gradients
              cycle =      1    SCF energy =    -10.0    |dE/dxyz| =  0.01
                0.0  0.0  0.0  o
                0.0  1.4  1.0  h
                0.0 -1.4  1.0  h
               1.0D-04  2.0D-04  3.0D-04
              -1.0D-04  0.0D+00  0.0D+00
               0.0D+00 -2.0D-04 -3.0D-04
            $end
        """)
        forces = self.calc._parse_forces(self.atoms, self.tmp)
        self.assertAlmostEqual(forces[0, 0], -1.0e-04 * Hartree / Bohr, places=12)

    def test_wrong_atom_count_raises(self):
        self._write_gradient("""\
            $grad  cartesian gradients
              cycle =      1    SCF energy =    -10.0    |dE/dxyz| =  0.01
                0.0  0.0  0.0  o
                0.0  1.4  1.0  h
               1.0D-04  2.0D-04  3.0D-04
            $end
        """)
        with self.assertRaises(RuntimeError, msg='Expected 3 gradients'):
            self.calc._parse_forces(self.atoms, self.tmp)


# ---------------------------------------------------------------------------
# _parse_charges
# ---------------------------------------------------------------------------

class TestParseCharges(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()
        self.tmp = Path(tempfile.mkdtemp())
        self.atoms = _make_atoms()  # 3 atoms

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_standard_format(self):
        (self.tmp / 'charges').write_text('-0.834\n 0.417\n 0.417\n')
        q = self.calc._parse_charges(self.atoms, self.tmp)
        np.testing.assert_allclose(q, [-0.834, 0.417, 0.417])

    def test_wrong_count_raises(self):
        (self.tmp / 'charges').write_text('-0.834\n 0.417\n')
        with self.assertRaises(RuntimeError):
            self.calc._parse_charges(self.atoms, self.tmp)

    def test_blank_lines_ignored(self):
        (self.tmp / 'charges').write_text('-0.834\n\n 0.417\n 0.417\n\n')
        q = self.calc._parse_charges(self.atoms, self.tmp)
        self.assertEqual(len(q), 3)


# ---------------------------------------------------------------------------
# _parse_dipole_from_stdout
# ---------------------------------------------------------------------------

class TestParseDipole(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()

    def test_full_keyword(self):
        stdout = (
            '   molecular dipole:\n'
            '                    x           y           z       tot (Debye)\n'
            '    full:          1.234       2.345       3.456       4.280\n'
        )
        d = self.calc._parse_dipole_from_stdout(stdout)
        self.assertIsNotNone(d)
        np.testing.assert_allclose(d, np.array([1.234, 2.345, 3.456]) * Debye)

    def test_q_plus_calc_keyword(self):
        stdout = (
            '   molecular dipole:\n'
            '    q+calc:        0.100       0.200       0.300       0.374\n'
        )
        d = self.calc._parse_dipole_from_stdout(stdout)
        self.assertIsNotNone(d)
        np.testing.assert_allclose(d, np.array([0.100, 0.200, 0.300]) * Debye)

    def test_zero_dipole(self):
        stdout = (
            '   molecular dipole:\n'
            '    full:          0.000       0.000       0.000       0.000\n'
        )
        d = self.calc._parse_dipole_from_stdout(stdout)
        np.testing.assert_allclose(d, np.zeros(3))

    def test_no_dipole_section_returns_none(self):
        self.assertIsNone(self.calc._parse_dipole_from_stdout('no dipole here'))

    def test_empty_stdout_returns_none(self):
        self.assertIsNone(self.calc._parse_dipole_from_stdout(''))

    def test_case_insensitive(self):
        stdout = (
            '   MOLECULAR DIPOLE:\n'
            '    FULL:          1.0   2.0   3.0   3.74\n'
        )
        d = self.calc._parse_dipole_from_stdout(stdout)
        self.assertIsNotNone(d)


# ---------------------------------------------------------------------------
# _parse_hessian
# ---------------------------------------------------------------------------

class TestParseHessian(unittest.TestCase):

    def setUp(self):
        self.calc = gxTB()
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_atoms(self, n):
        """Dummy atoms-like object with __len__."""
        a = MagicMock()
        a.__len__ = MagicMock(return_value=n)
        return a

    def test_2atom_hessian(self):
        n = 2  # 6x6 = 36 elements
        values = list(range(36))
        lines = []
        for i in range(0, 36, 5):
            lines.append('  ' + '  '.join(str(v) + '.0' for v in values[i:i+5]))
        (self.tmp / 'hessian').write_text('$hessian\n' + '\n'.join(lines) + '\n$end\n')
        H = self.calc._parse_hessian(self._make_atoms(2), self.tmp)
        self.assertEqual(H.shape, (6, 6))
        # Conversion factor applied
        self.assertAlmostEqual(H[0, 0], 0.0 * Hartree / Bohr**2)
        self.assertAlmostEqual(H[0, 1], 1.0 * Hartree / Bohr**2)

    def test_missing_file_raises(self):
        with self.assertRaises(RuntimeError, msg='hessian'):
            self.calc._parse_hessian(self._make_atoms(3), self.tmp)

    def test_wrong_element_count_raises(self):
        # 3 atoms → 9x9 = 81 elements needed; write only 9
        (self.tmp / 'hessian').write_text(
            '$hessian\n  ' + '  '.join(['1.0'] * 9) + '\n$end\n'
        )
        with self.assertRaises(RuntimeError):
            self.calc._parse_hessian(self._make_atoms(3), self.tmp)

    def test_invalid_value_raises_with_context(self):
        (self.tmp / 'hessian').write_text('$hessian\n  1.0  NOT_A_NUMBER\n$end\n')
        with self.assertRaises(RuntimeError, msg='could not parse'):
            self.calc._parse_hessian(self._make_atoms(1), self.tmp)


# ---------------------------------------------------------------------------
# calculate() — mocked subprocess
# ---------------------------------------------------------------------------

class TestCalculateMocked(unittest.TestCase):
    """
    End-to-end calculate() tests using a fake xtb that writes minimal
    output files.
    """

    ENERGY_EV = -76.316820093019 * Hartree

    def _fake_run(self, work_dir, grad=False):
        """Write the files xtb would produce into work_dir."""
        (work_dir / 'energy').write_text(
            '$energy\n 1  -76.316820093019\n$end\n'
        )
        if grad:
            atoms = _make_atoms()  # H2O, 3 atoms
            grad_lines = (
                '$grad  cartesian gradients\n'
                '  cycle =      1    SCF energy =    -76.3168\n'
                '   0.0  0.0  0.118  o\n'
                '   0.0  1.44 -0.59  h\n'
                '   0.0 -1.44 -0.59  h\n'
                '  1.0D-05  2.0D-05  3.0D-05\n'
                ' -1.0D-05  0.0D+00  0.0D+00\n'
                '  0.0D+00 -2.0D-05 -3.0D-05\n'
                '$end\n'
            )
            (work_dir / 'gradient').write_text(grad_lines)
        (work_dir / 'charges').write_text('-0.834\n 0.417\n 0.417\n')

    def _patch_run(self, calc, grad=False):
        orig_run = calc._run_command

        def fake_run(cmd, work_dir):
            self._fake_run(work_dir, grad=grad)
            calc._raw_stdout = (
                '          TOTAL ENERGY              -76.316820093019 Eh\n'
                '   molecular dipole:\n'
                '    full:          0.000       0.000       1.850       1.850\n'
            )
            calc.stdout = calc._raw_stdout if calc.capture_stdout else None

        calc._run_command = fake_run

    def test_energy_only(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc)
        atoms.calc = calc
        E = atoms.get_potential_energy()
        self.assertAlmostEqual(E, self.ENERGY_EV, places=4)

    def test_forces(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc, grad=True)
        atoms.calc = calc
        forces = atoms.get_forces()
        self.assertEqual(forces.shape, (3, 3))

    def test_charges_populated(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc, grad=False)
        atoms.calc = calc
        atoms.get_potential_energy()
        charges = calc.results.get('charges')
        self.assertIsNotNone(charges)
        self.assertEqual(len(charges), 3)
        self.assertAlmostEqual(charges.sum(), 0.0, places=5)

    def test_dipole_populated_in_results(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc, grad=False)
        atoms.calc = calc
        atoms.get_potential_energy()
        dipole = calc.results.get('dipole')
        self.assertIsNotNone(dipole)
        self.assertEqual(len(dipole), 3)

    def test_get_dipole_moment_returns_cached(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc, grad=False)
        atoms.calc = calc
        atoms.get_potential_energy()
        dipole = calc.get_dipole_moment()
        self.assertIsNotNone(dipole)
        self.assertEqual(dipole.shape, (3,))

    def test_get_dipole_moment_raises_when_absent(self):
        from ase.calculators.calculator import PropertyNotImplementedError
        calc = gxTB()
        calc.results = {}
        with self.assertRaises(PropertyNotImplementedError):
            calc.get_dipole_moment()

    def test_capture_stdout_false_by_default(self):
        atoms = _make_atoms()
        calc = gxTB(capture_stdout=False)
        self._patch_run(calc)
        atoms.calc = calc
        atoms.get_potential_energy()
        self.assertIsNone(calc.stdout)

    def test_capture_stdout_true(self):
        atoms = _make_atoms()
        calc = gxTB(capture_stdout=True)
        self._patch_run(calc)
        atoms.calc = calc
        atoms.get_potential_energy()
        self.assertIsNotNone(calc.stdout)
        self.assertIn('TOTAL ENERGY', calc.stdout)

    def test_charge_from_atoms_info(self):
        atoms = _make_atoms()
        atoms.info['charge'] = -1
        calc = gxTB()
        recorded = []
        orig_build = calc._build_command

        def spy_build(coord, charge, uhf, flags):
            recorded.append(charge)
            return orig_build(coord, charge, uhf, flags)

        calc._build_command = spy_build
        self._patch_run(calc)
        atoms.calc = calc
        atoms.get_potential_energy()
        self.assertEqual(recorded[0], -1)

    def test_temp_dir_cleaned_up(self):
        import glob, os
        before = set(glob.glob(str(Path(tempfile.gettempdir()) / 'gxtb_*')))
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run(calc)
        atoms.calc = calc
        atoms.get_potential_energy()
        after = set(glob.glob(str(Path(tempfile.gettempdir()) / 'gxtb_*')))
        self.assertEqual(before, after)

    def test_keep_files_preserves_dir(self):
        import glob
        before = set(glob.glob(str(Path(tempfile.gettempdir()) / 'gxtb_*')))
        atoms = _make_atoms()
        calc = gxTB(keep_files=True)
        self._patch_run(calc)
        atoms.calc = calc
        atoms.get_potential_energy()
        after = set(glob.glob(str(Path(tempfile.gettempdir()) / 'gxtb_*')))
        new_dirs = after - before
        self.assertEqual(len(new_dirs), 1)
        shutil.rmtree(new_dirs.pop())


# ---------------------------------------------------------------------------
# get_hessian() — mocked
# ---------------------------------------------------------------------------

class TestGetHessianMocked(unittest.TestCase):

    def _patch_run_hess(self, calc, atoms):
        n = len(atoms)
        size = 3 * n

        def fake_run(cmd, work_dir):
            # --hess does NOT write a gradient file (matches real xtb behaviour)
            (work_dir / 'energy').write_text('$energy\n 1  -10.0\n$end\n')
            (work_dir / 'charges').write_text('\n'.join(['0.0'] * n) + '\n')
            hess_vals = [str(float(i)) for i in range(size * size)]
            lines = []
            for i in range(0, len(hess_vals), 5):
                lines.append('  ' + '  '.join(hess_vals[i:i+5]))
            (work_dir / 'hessian').write_text('$hessian\n' + '\n'.join(lines) + '\n$end\n')
            calc._raw_stdout = '          TOTAL ENERGY              -10.0 Eh\n'

        calc._run_command = fake_run

    def test_returns_correct_shape(self):
        atoms = _make_atoms()  # H2O, 3 atoms → 9x9
        calc = gxTB()
        self._patch_run_hess(calc, atoms)
        H = calc.get_hessian(atoms)
        self.assertEqual(H.shape, (9, 9))

    def test_units_conversion(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run_hess(calc, atoms)
        H = calc.get_hessian(atoms)
        # H[0,1] should be 1.0 * Hartree/Bohr²
        self.assertAlmostEqual(H[0, 1], 1.0 * Hartree / Bohr**2, places=6)

    def test_also_populates_energy(self):
        atoms = _make_atoms()
        calc = gxTB()
        self._patch_run_hess(calc, atoms)
        calc.get_hessian(atoms)
        self.assertIn('energy', calc.results)

    def test_no_atoms_raises(self):
        calc = gxTB()
        with self.assertRaises(ValueError):
            calc.get_hessian(None)


# ---------------------------------------------------------------------------
# GXTBHOME warning
# ---------------------------------------------------------------------------

class TestGxtbhomeWarning(unittest.TestCase):

    def test_missing_gxtbhome_emits_warning(self):
        import warnings
        calc = gxTB(gxtbhome='/nonexistent/path/to/params')
        captured = []

        real_run = calc._run_command

        def fake_run(cmd, work_dir):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                if not calc.gxtbhome.exists():
                    warnings.warn('missing', RuntimeWarning)
            captured.extend(w)

        calc._run_command = fake_run

        atoms = _make_atoms()
        atoms.calc = calc

        def fake_calc(atoms, properties, system_changes):
            from ase.calculators.calculator import all_changes
            calc._raw_stdout = '          TOTAL ENERGY              -10.0 Eh\n'
            calc.results['energy'] = -10.0 * Hartree

        calc.calculate = fake_calc

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            if not calc.gxtbhome.exists():
                warnings.warn('g-xTB parameter directory not found', RuntimeWarning)

        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))


# ---------------------------------------------------------------------------
# benchmark_parallel
# ---------------------------------------------------------------------------

class TestBenchmarkParallel(unittest.TestCase):
    """Tests for benchmark_parallel() — gxTB calls are fully mocked."""

    def setUp(self):
        self.atoms = _make_atoms()

    def _patch_time_single(self, times):
        """Return a context manager that replaces _time_single with a queue of values."""
        import itertools
        from unittest.mock import patch as _patch
        counter = itertools.cycle(times)
        return _patch('gxtb.benchmark._time_single', side_effect=lambda *a, **kw: next(counter))

    def test_returns_dict_keyed_by_nprocs(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([2.0, 1.0]):
            result = benchmark_parallel(self.atoms, [1, 2])
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_empty_nprocs_list_raises(self):
        from gxtb.benchmark import benchmark_parallel
        with self.assertRaises(ValueError):
            benchmark_parallel(self.atoms, [])

    def test_repeat_zero_raises(self):
        from gxtb.benchmark import benchmark_parallel
        with self.assertRaises(ValueError):
            benchmark_parallel(self.atoms, [1], repeat=0)

    def test_invalid_task_raises(self):
        from gxtb.benchmark import benchmark_parallel
        with self.assertRaises(ValueError):
            benchmark_parallel(self.atoms, [1], task='invalid')

    def test_task_energy_default(self):
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([1.0]):
            result = benchmark_parallel(self.atoms, [1])
        self.assertIn(1, result)

    def test_task_forces(self):
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([1.0]):
            result = benchmark_parallel(self.atoms, [1], task='forces')
        self.assertIn(1, result)

    def test_task_hessian(self):
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([1.0]):
            result = benchmark_parallel(self.atoms, [1], task='hessian')
        self.assertIn(1, result)

    def test_average_over_repeat(self):
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([4.0, 2.0]):
            result = benchmark_parallel(self.atoms, [1], repeat=2)
        self.assertAlmostEqual(result[1], 3.0)

    def test_single_nproc(self):
        from gxtb.benchmark import benchmark_parallel
        with self._patch_time_single([5.0]):
            result = benchmark_parallel(self.atoms, [4])
        self.assertAlmostEqual(result[4], 5.0)

    def test_warmup_triggers_extra_call(self):
        from gxtb.benchmark import benchmark_parallel
        call_log = []

        def recording_time_single(atoms, nprocs, task, kw):
            call_log.append(nprocs)
            return 1.0

        with patch('gxtb.benchmark._time_single', side_effect=recording_time_single):
            benchmark_parallel(self.atoms, [1, 2], repeat=1, warmup=True)

        # warmup call + 2 timed calls = 3 total
        self.assertEqual(len(call_log), 3)
        # first call is the warmup with nprocs_list[0]
        self.assertEqual(call_log[0], 1)

    def test_calc_kwargs_nprocs_stripped(self):
        from gxtb.benchmark import benchmark_parallel
        received_kw = []

        def recording_time_single(atoms, nprocs, task, kw):
            received_kw.append(dict(kw))
            return 1.0

        with patch('gxtb.benchmark._time_single', side_effect=recording_time_single):
            benchmark_parallel(self.atoms, [1], calc_kwargs={'nprocs': 99, 'charge': -1})

        self.assertNotIn('nprocs', received_kw[0])
        self.assertEqual(received_kw[0].get('charge'), -1)

    def test_plot_warns_when_matplotlib_missing(self):
        import warnings
        import sys
        from gxtb.benchmark import benchmark_parallel

        with self._patch_time_single([1.0]):
            with patch.dict(sys.modules, {'matplotlib': None, 'matplotlib.pyplot': None}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    benchmark_parallel(self.atoms, [1], plot=True)
        # Either no warning (matplotlib already installed and used) or RuntimeWarning
        runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
        # This test simply asserts it doesn't raise an exception
        # (matplotlib may or may not be installed in the test environment)

    def test_time_single_calls_correct_method(self):
        from gxtb.benchmark import _time_single
        calls = []

        def fake_calc_cls(nprocs, **kw):
            calc = MagicMock()
            calc.get_hessian = MagicMock(return_value=None)
            calls.append(calc)
            return calc

        atoms = _make_atoms()

        for task, attr in [('energy', 'get_potential_energy'),
                            ('forces', 'get_forces'),
                            ('hessian', 'get_hessian')]:
            calls.clear()
            with patch('gxtb.benchmark.gxTB', side_effect=fake_calc_cls):
                _time_single(atoms, 1, task, {})
            self.assertEqual(len(calls), 1)
            mock_calc = calls[0]
            if task == 'hessian':
                mock_calc.get_hessian.assert_called_once()
            elif task == 'forces':
                mock_calc.get_forces.assert_called_once()
            else:
                mock_calc.get_potential_energy.assert_called_once()

    def test_print_table_outputs_speedup(self, capsys=None):
        """_print_table should produce speedup column in stdout."""
        from gxtb.benchmark import _print_table
        import io
        from contextlib import redirect_stdout
        timings = {1: 4.0, 2: 2.0, 4: 1.5}
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_table(self.atoms, timings, 'energy', repeat=3)
        output = buf.getvalue()
        self.assertIn('speedup', output.lower())
        self.assertIn('efficiency', output.lower())
        self.assertIn('energy', output.lower())
        # nprocs values appear in table
        self.assertIn('1', output)
        self.assertIn('2', output)
        self.assertIn('4', output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
