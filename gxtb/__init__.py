import requests
from pathlib import Path
import os
import shutil
from .calculator import gxTB

__version__ = '0.0.1'


def gxtb_install(verbose=True):
    """
    Install g-xTB binary and parameter files
    Downloads necessary files from the official g-xTB repository
    
    Parameters:
    -----------
    verbose : bool, default=True
        If True, print detailed installation progress
    """
    home_dir = Path.home()
    bin_dir = home_dir / "bin"

    # URLs for parameter files and binary
    files_to_download = [
        {
            'url': 'https://raw.githubusercontent.com/kangmg/g-xtb/refs/heads/main/parameters/.gxtb',
            'path': home_dir / '.gxtb'
        },
        {
            'url': 'https://raw.githubusercontent.com/kangmg/g-xtb/refs/heads/main/parameters/.basisq',
            'path': home_dir / '.basisq'
        },
        {
            'url': 'https://raw.githubusercontent.com/kangmg/g-xtb/refs/heads/main/parameters/.eeq',
            'path': home_dir / '.eeq'
        },
        {
            'url': 'https://github.com/kangmg/g-xtb/raw/refs/heads/main/binary/gxtb',
            'path': bin_dir / 'gxtb'
        }
    ]
    
    # Create bin directory if it doesn't exist
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    print("Installing g-xTB files...")
    
    for file_info in files_to_download:
        url = file_info['url']
        file_path = file_info['path']
        
        try:
            if verbose:
                print(f"Downloading {url} -> {file_path}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Make gxtb binary executable
            if file_path.name == 'gxtb':
                os.chmod(file_path, 0o755)
                if verbose:
                    print(f"Made {file_path} executable")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
        except OSError as e:
            raise RuntimeError(f"Failed to write {file_path}: {e}")
    
    # Add ~/bin to PATH if not already there
    home_bin = f"{os.environ['HOME']}/bin"
    if home_bin not in os.environ['PATH']:
        os.environ['PATH'] += f":{home_bin}"
        if verbose:
            print(f"Added {home_bin} to PATH")
    
    print("g-xTB installation completed!")
    print(f"Binary installed at: {bin_dir / 'gxtb'}")
    print(f"Parameter files installed in: {home_dir}")
    print(f"~/bin has been added to PATH for this session.")


# Check if gxtb executable is available in PATH
if shutil.which('gxtb') is None:
    print("gxtb command not found. Installing...")
    gxtb_install(verbose=True)
