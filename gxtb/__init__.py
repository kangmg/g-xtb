import platform
import shutil
import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path
from .calculator import gxTB

__version__ = '0.1.0'

# Platform → (tarball name, sha256 filename)
_BINARY_VERSION = 'xtb-6.7.1-gxtb-210426'
_BINARY_BASE_URL = 'https://github.com/grimme-lab/g-xtb/raw/main/binaries/'
_BINARY_MAP = {
    ('Linux',  'x86_64'): (f'{_BINARY_VERSION}-linux-x86_64.tar.xz',  'xtb'),
    ('Darwin', 'arm64'):  (f'{_BINARY_VERSION}-macos-arm64.tar.gz',   'xtb'),
    ('Windows','AMD64'):  (f'{_BINARY_VERSION}-windows-x86_64.zip',   'xtb.exe'),
}

_PARAM_FILES = ['.gxtb', '.eeq', '.basisq']
_PARAM_URL_BASE = 'https://raw.githubusercontent.com/grimme-lab/g-xtb/main/parameters/'


def _detect_platform():
    sys = platform.system()
    mach = platform.machine()
    return sys, mach


def gxtb_install(install_dir=None, verbose=True, overwrite=False):
    """
    Download and install the g-xTB-enabled xtb binary and parameter files.

    The binary is downloaded from the official grimme-lab/g-xtb repository
    and placed in ``install_dir`` (default: ``~/bin``). Parameter files are
    updated in the ``parameters/`` directory bundled with this package so
    that GXTBHOME is set automatically by the calculator.

    Parameters
    ----------
    install_dir : str or Path, optional
        Directory to install the xtb binary. Defaults to ~/bin.
    verbose : bool, default=True
        Print progress messages.
    overwrite : bool, default=False
        Overwrite existing binary/parameter files.
    """
    sys_name, machine = _detect_platform()
    key = (sys_name, machine)

    if key not in _BINARY_MAP:
        raise RuntimeError(
            f"No pre-built binary available for {sys_name}/{machine}. "
            f"Supported: {list(_BINARY_MAP.keys())}"
        )

    archive_name, exe_name = _BINARY_MAP[key]
    install_dir = Path(install_dir) if install_dir else Path.home() / 'bin'
    install_dir.mkdir(parents=True, exist_ok=True)
    exe_path = install_dir / exe_name

    # --- Download and extract binary ---
    if exe_path.exists() and not overwrite:
        if verbose:
            print(f"Binary already exists: {exe_path}  (use overwrite=True to replace)")
    else:
        archive_url = _BINARY_URL = _BINARY_BASE_URL + archive_name
        tmp_archive = install_dir / archive_name
        if verbose:
            print(f"Downloading {archive_url} ...")
        urllib.request.urlretrieve(archive_url, tmp_archive)

        if verbose:
            print(f"Extracting {archive_name} ...")
        if archive_name.endswith('.tar.xz') or archive_name.endswith('.tar.gz'):
            with tarfile.open(tmp_archive) as tf:
                # Extract the xtb executable (may be at bin/xtb inside the archive)
                for member in tf.getmembers():
                    if member.name.endswith('/' + exe_name) or member.name == exe_name:
                        member.name = exe_name
                        tf.extract(member, path=install_dir)
                        break
                else:
                    raise RuntimeError(f"Could not find '{exe_name}' inside {archive_name}")
        elif archive_name.endswith('.zip'):
            with zipfile.ZipFile(tmp_archive) as zf:
                for name in zf.namelist():
                    if name.endswith('/' + exe_name) or name == exe_name:
                        data = zf.read(name)
                        with open(exe_path, 'wb') as f:
                            f.write(data)
                        break
                else:
                    raise RuntimeError(f"Could not find '{exe_name}' inside {archive_name}")

        tmp_archive.unlink()
        exe_path.chmod(0o755)
        if verbose:
            print(f"Installed: {exe_path}")

    # --- Update bundled parameter files ---
    param_dir = Path(__file__).parent.parent / 'parameters'
    param_dir.mkdir(exist_ok=True)
    for fname in _PARAM_FILES:
        dest = param_dir / fname
        if dest.exists() and not overwrite:
            if verbose:
                print(f"Parameter file exists: {dest}  (use overwrite=True to replace)")
            continue
        url = _PARAM_URL_BASE + fname
        if verbose:
            print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        if verbose:
            print(f"Updated: {dest}")

    # --- Add install_dir to PATH for this session ---
    home_bin = str(install_dir)
    if home_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = home_bin + os.pathsep + os.environ.get('PATH', '')
        if verbose:
            print(f"Added {home_bin} to PATH for this session")

    if verbose:
        print(f"\ng-xTB installation complete.")
        print(f"  Binary : {exe_path}")
        print(f"  Params : {param_dir}")
        print(f"  Usage  : gxTB(command='{exe_path}')")


def find_xtb_binary():
    """
    Return the path to the g-xTB-enabled xtb binary, or None if not found.

    Searches PATH for 'xtb'. Does not verify that the binary actually
    supports --gxtb; install via gxtb_install() to get the correct build.
    """
    return shutil.which('xtb')
