# g-xTB ASE interface(unofficial)

**This repository is an unofficial implementation of g-xTB for temporary use until its official release.**

The g-xTB method is currently under active development and is scheduled to be officially implemented in the `tblite` library. This repository provides a way to use g-xTB in ase interface before it is officially released.

## Installation

You can install the package directly from this GitHub repository using pip:

```bash
pip install git+https://github.com/kangmg/g-xtb.git
```

## Usage

More examples can be found in  <a href="https://colab.research.google.com/github/kangmg/g-xtb/blob/main/example/gxTB_tutorial.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
#######################
# Basic usage example # 
#######################

from ase.build import molecule
from gxtb import gxTB

# Create an ASE Atoms object
atoms = molecule('CH3COOH')

# Set charge and unrestricted Hartree-Fock (UHF) parameters
atoms.info['charge'] = 0
atoms.info['uhf'] = 0

# Initialize the g-xTB calculator
# You can also pass parameters directly: gxTB(charge=0, uhf=0)
atoms.calc = gxTB()

# Calculate energy and forces
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy(eV): {energy}\n")
print(f"Forces(eV/A):\n{forces}")
```
