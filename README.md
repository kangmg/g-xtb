# g-xTB ASE Calculator (unofficial)

**Unofficial ASE calculator wrapper for g-xTB.**  
Official integration into the [tblite](https://github.com/tblite/tblite) library is planned; this package provides early access in the meantime.

g-xTB is a general-purpose semiempirical quantum mechanical method approximating ωB97M-V/def2-TZVPPD properties for elements Z = 1–103.  
Starting from v2.0.0, g-xTB runs as `xtb --gxtb` (via a modified xtb 6.7.1 binary).

---

## Installation

```bash
pip install git+https://github.com/kangmg/g-xtb.git
```

Then download the xtb binary and parameter files:

```python
from gxtb import gxtb_install
gxtb_install()          # installs to ~/bin by default
```

Or specify a custom location:

```python
gxtb_install(install_dir='/opt/xtb-gxtb/bin')
```

If the xtb binary is already on your PATH, no install step is needed.

---

## Quick start

```python
from ase.build import molecule
from gxtb import gxTB

atoms = molecule('H2O')
atoms.calc = gxTB()

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Å
print(f"Energy: {energy:.4f} eV")
print(f"Forces:\n{forces}")
```

---

## Properties

### Energy and forces

```python
atoms.calc = gxTB()

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Å,  shape (N, 3)
```

Analytic gradients are used — much faster than the numerical gradients in g-xTB v1.

### Partial charges

```python
atoms.calc = gxTB()
atoms.get_potential_energy()            # triggers calculation

charges = atoms.calc.get_charges()     # shape (N,),  units e
print(f"Charges: {charges}")
print(f"Sum: {charges.sum():.4f}")     # ≈ total molecular charge
```

### Dipole moment

```python
atoms.calc = gxTB()
atoms.get_potential_energy()

dipole = atoms.calc.get_dipole_moment()  # shape (3,),  units e·Å
print(f"Dipole: {dipole}")
```

### Hessian

`get_hessian()` runs `xtb --gxtb --hess` (numerical Hessian via analytic gradients) and returns the full Cartesian force-constant matrix.  
The same run also updates energy, forces, charges, and dipole.

```python
atoms.calc = gxTB()

H = atoms.calc.get_hessian(atoms)  # shape (3N, 3N),  units eV/Å²
print(f"Hessian shape: {H.shape}")
```

Computing vibrational frequencies with ASE:

```python
from ase.vibrations import Vibrations

atoms.calc = gxTB()
vib = Vibrations(atoms)
vib.run()
vib.summary()
```

---

## Charge and spin

Priority order (highest first):

| Source | Key |
|---|---|
| `atoms.info` | `'charge'` / `'uhf'` / `'spin'` |
| Constructor argument | `charge=` / `uhf=` / `spin=` |

`spin` is an alias for `uhf` (number of unpaired electrons).  
`atoms.info['uhf']` takes priority over `atoms.info['spin']` when both are set.

```python
# Via atoms.info
atoms.info['charge'] = -1
atoms.info['spin'] = 2          # or atoms.info['uhf'] = 2
atoms.calc = gxTB()

# Via constructor
atoms.calc = gxTB(charge=-1, spin=2)

# Mix: atoms.info overrides the constructor
atoms.info['charge'] = 0        # overrides charge=-1 set in constructor
atoms.calc = gxTB(charge=-1)
```

---

## Calculator parameters

| Parameter | Default | Description |
|---|---|---|
| `command` | `'xtb'` | Path or name of the g-xTB-enabled xtb binary |
| `charge` | `None` | Molecular charge (integer) |
| `uhf` / `spin` | `None` | Number of unpaired electrons (integer) |
| `workdir` | `None` | Working directory for xtb files (see below) |
| `keep_files` | `False` | Keep working directory after calculation |
| `verbose` | `False` | Print xtb stdout during calculation |
| `capture_stdout` | `False` | Store xtb stdout in `calc.stdout` |
| `gxtbhome` | *(bundled)* | Path to g-xTB parameter directory |

### workdir

By default each calculation runs in a fresh temporary directory that is deleted afterwards, keeping your working directory clean and preventing file conflicts between concurrent calculations.

```python
# Default: isolated temp dir per calculation (auto-cleaned)
atoms.calc = gxTB()

# Explicit directory: reused across calls, known output files cleaned up
atoms.calc = gxTB(workdir='./gxtb_run')

# Keep all files for inspection (prints path when verbose=True)
atoms.calc = gxTB(keep_files=True, verbose=True)
```

### Custom binary path

```python
atoms.calc = gxTB(command='/opt/xtb-gxtb/bin/xtb')
```

### Custom parameter directory

```python
atoms.calc = gxTB(gxtbhome='/path/to/gxtb/parameters')
```

---

## Integration with ASE workflows

### Geometry optimisation

```python
from ase.optimize import LBFGS

atoms.calc = gxTB()
opt = LBFGS(atoms)
opt.run(fmax=0.01)          # eV/Å
```

### Molecular dynamics

```python
from ase.md.langevin import Langevin
from ase.units import fs, kB

atoms.calc = gxTB()
md = Langevin(atoms, timestep=1.0*fs, temperature_K=300, friction=0.01)
md.run(steps=1000)
```

### Transition state search (NEB)

```python
from ase.neb import NEB
from ase.optimize import BFGS

images = [atoms.copy() for _ in range(5)]
for img in images:
    img.calc = gxTB()

neb = NEB(images)
opt = BFGS(neb)
opt.run(fmax=0.05)
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers all parser methods, command building, workdir management, charge/spin validation, and mocked end-to-end calculations.  
No xtb binary is required to run the tests.

---

## Acknowledgements

- [grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb) — original g-xTB method and binary
- [tblite](https://github.com/tblite/tblite) — future home of g-xTB
- [ASE](https://wiki.fysik.dtu.dk/ase/) — Atomic Simulation Environment
