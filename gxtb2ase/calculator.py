"""
g-xTB ASE Calculator

ASE calculator interface for g-xTB (general-purpose semiempirical quantum mechanical method)
"""
import os
import subprocess
import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr
from ase.io import write, read
import requests

class gxTB(Calculator):
    """
    ASE calculator for g-xTB
    
    g-xTB is a general-purpose semiempirical quantum mechanical method 
    approximating ωB97M-V/def2-TZVPPD properties.
    
    Parameters:
    -----------
    keep_files : bool, default=False
        If True, temporary files (.CHRG, .UHF, TMP_gxtb.xyz) are not deleted
        after calculation
    command : str, default='gxtb'
        Command to run g-xTB executable
    charge : int, optional
        Charge of the system. Can be overridden by atoms.info['charge']
    uhf : int, optional
        Number of unpaired electrons. Can be overridden by atoms.info['uhf']
    verbose : bool, default=False
        If True, print verbose output
    capture_stdout : bool, default=False
        If True, capture and store stdout in atoms.info['gxtb_stdout']
        If False, redirect stdout to /dev/null (or NUL on Windows)
    """
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, keep_files=False, command='gxtb', charge=None, uhf=None, 
                 verbose=False, capture_stdout=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_files = keep_files
        self.command = command
        self.charge = charge
        self.uhf = uhf
        self.verbose = verbose
        self.capture_stdout = capture_stdout
        self.stdout = None
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Calculate energy and forces using g-xTB
        """
        if atoms is None:
            atoms = self.atoms
        
        super().calculate(atoms, properties, system_changes)
        
        # If there are system changes other than positions, clear old files
        # This ensures clean calculation when charge, cell, etc. change
        non_position_changes = ['numbers', 'cell', 'pbc', 'initial_charges', 'initial_magmoms']

        if non_position_changes:
            self.clear_files()
        
        # Write coordinate file
        coord_file = 'TMP_gxtb.xyz'
        write(coord_file, atoms, format='xyz')
        
        # Handle charge and spin
        self._write_control_files(atoms)
        
        # Run g-xTB calculation
        try:
            if 'forces' in properties:
                self._calculate_forces(atoms)
            else:
                self._calculate_energy(atoms)
        except Exception as e:
            if not self.keep_files:
                self.clear_files()
            raise e
        
        # Clean up files if requested
        if not self.keep_files:
            self.clear_files()
    
    def _write_control_files(self, atoms):
        """Write .CHRG and .UHF control files if needed"""
        
        # Handle charge - priority: atoms.info['charge'] > self.charge
        charge = atoms.info.get('charge', self.charge)
        if charge is not None:
            with open('.CHRG', 'w') as f:
                f.write(f"{int(charge)}\n")
        
        # Handle spin (UHF) - priority: atoms.info['uhf'] > self.uhf
        uhf = atoms.info.get('uhf', self.uhf)
        if uhf is not None:
            with open('.UHF', 'w') as f:
                f.write(f"{int(uhf)}\n")
    
    def _run_gxtb_command(self, cmd, atoms):
        """Run g-xTB command with appropriate stdout handling"""
        if self.capture_stdout:
            # Capture stdout and store in atoms.info
            result = subprocess.run(
                cmd, capture_output=True, 
                text=True, check=True
                )
            self.stdout = result.stdout
            return result
        else:
            # Redirect stdout to null device
            null_device = os.devnull
            with open(null_device, 'w') as devnull:
                result = subprocess.run(
                    cmd, stdout=devnull, 
                    stderr=subprocess.PIPE, 
                    text=True, check=True
                    )
            return result
    
    def _calculate_energy(self, atoms):
        """Calculate energy only"""
        cmd = [self.command, '-c', 'TMP_gxtb.xyz']
        
        try:
            self._run_gxtb_command(cmd, atoms)
            
            # Parse energy from output
            energy = self._parse_energy()
            self.results['energy'] = energy
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"g-xTB calculation failed: {e.stderr}")
    
    def _calculate_forces(self, atoms):
        """Calculate energy and forces"""
        cmd = [self.command, '-c', 'TMP_gxtb.xyz', '-grad']
        
        try:
            self._run_gxtb_command(cmd, atoms)
            
            # Parse energy from output
            energy = self._parse_energy()
            self.results['energy'] = energy
            
            # Parse forces from gradient file
            forces = self._parse_forces(atoms)
            self.results['forces'] = forces
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"g-xTB gradient calculation failed: {e.stderr}")
    
    def _parse_energy(self):
        """Parse energy from energy file"""
        energy_file = 'energy'
        
        if not os.path.exists(energy_file):
            raise RuntimeError("Energy file not found")
        
        try:
            with open(energy_file, 'r') as f:
                lines = f.readlines()
            
            # Look for $energy section
            in_energy_section = False
            for line in lines:
                line = line.strip()
                
                if line.startswith('$energy'):
                    in_energy_section = True
                    continue
                
                if in_energy_section:
                    if line.startswith('$end'):
                        break
                    
                    # Parse energy line format: "1  -76.43708019638818  26.05970947993875  99.9 99.9 99.9"
                    if line and not line.startswith('$'):
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                energy = float(parts[1])  # Second column is the energy
                                return energy * Hartree  # Convert to eV
                        except (ValueError, IndexError):
                            continue
            
            raise RuntimeError("Could not parse energy from energy file")
        
        except Exception as e:
            raise RuntimeError(f"Could not read energy file: {e}")
    
    def _parse_forces(self, atoms):
        """Parse forces from gradient file"""
        grad_file = 'gradient'
        
        if not os.path.exists(grad_file):
            raise RuntimeError("Gradient file not found")
        
        try:
            with open(grad_file, 'r') as f:
                lines = f.readlines()
            
            # Find the gradient section
            in_grad_section = False
            gradients = []
            n_atoms = len(atoms)
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('$grad'):
                    in_grad_section = True
                    continue
                
                if in_grad_section:
                    if line.startswith('$end'):
                        break
                    
                    # Skip header lines and coordinate lines
                    if (line.startswith('cycle') or 
                        'SCF energy' in line or
                        line.startswith('$') or
                        not line):
                        continue
                    
                    # Skip coordinate lines (they typically have 4 columns with element symbol)
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            # If first three are numbers and last is string, it's a coordinate line
                            float(parts[0])
                            float(parts[1]) 
                            float(parts[2])
                            # Fourth part should be element symbol
                            continue
                        except ValueError:
                            pass
                    
                    # Parse gradient lines (should have 3 columns)
                    if len(parts) == 3:
                        try:
                            # Handle scientific notation format like "0.1131140940250D-01"
                            grad_x = float(parts[0].replace('D', 'E'))
                            grad_y = float(parts[1].replace('D', 'E'))
                            grad_z = float(parts[2].replace('D', 'E'))
                            gradients.append([grad_x, grad_y, grad_z])
                        except (ValueError, IndexError):
                            continue
            
            if len(gradients) != n_atoms:
                raise RuntimeError(f"Expected {n_atoms} gradients, got {len(gradients)}")
            
            # Convert gradients to forces (negative gradient)
            forces = np.array(gradients) * (-1.0)
            
            # Convert from Hartree/Bohr to eV/Å
            forces = forces * (Hartree / Bohr)
            
            return forces
            
        except Exception as e:
            raise RuntimeError(f"Could not parse forces from gradient file: {e}")
    
    def clear_files(self):
        """
        Clear all temporary files (.CHRG, .UHF, TMP_gxtb.xyz, etc.)
        This method can be called manually to clean up files
        """
        temp_files = [
            '.CHRG', '.UHF', 'TMP_gxtb.xyz', 'gradient', 
            'energy', 'coord', 'gxtbrestart'
            ]
        
        for filename in temp_files:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    if self.verbose:
                        print(f"Removed: {filename}")
                except OSError:
                    if self.verbose:
                        print(f"File not found: {filename}")