import hashlib
import os
import platform
import shutil
import tarfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from .calculator import gxTB
from .benchmark import benchmark_parallel

__version__ = '0.1.0'

# Bundled parameter files live inside the package so they are always
# available after a normal `pip install` (wheel or editable).
_PARAM_DIR = Path(__file__).parent / 'parameters'

_BINARY_VERSION = 'xtb-6.7.1-gxtb-210426'
_BINARY_BASE_URL = 'https://github.com/kangmg/g-xtb/raw/main/binaries/'

# (platform.system(), platform.machine()) → (archive filename, exe name)
# Intel-Mac is intentionally absent: the upstream does not ship an x86_64
# macOS binary; users on Intel Macs should build from source.
_BINARY_MAP = {
    ('Linux',   'x86_64'): (f'{_BINARY_VERSION}-linux-x86_64.tar.xz', 'xtb'),
    ('Darwin',  'arm64'):  (f'{_BINARY_VERSION}-macos-arm64.tar.gz',  'xtb'),
    ('Windows', 'AMD64'):  (f'{_BINARY_VERSION}-windows-x86_64.zip',  'xtb.exe'),
}

_PARAM_FILES = ['.gxtb', '.eeq', '.basisq']
_PARAM_URL_BASE = 'https://raw.githubusercontent.com/kangmg/g-xtb/main/parameters/'

_LFS_POINTER_MAGIC = b'version https://git-lfs.github.com/spec/v1'


def _platform_key():
    return platform.system(), platform.machine()


def _sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _check_lfs_pointer(path):
    """Raise if the downloaded file is a Git LFS pointer instead of the real binary."""
    if path.stat().st_size > 1024:
        return
    try:
        with open(path, 'rb') as f:
            header = f.read(len(_LFS_POINTER_MAGIC))
        if header == _LFS_POINTER_MAGIC:
            raise RuntimeError(
                f"Downloaded file is a Git LFS pointer, not the actual binary.\n"
                f"Try downloading directly from:\n"
                f"  https://media.githubusercontent.com/media/kangmg/g-xtb/main/binaries/{path.name}"
            )
    except OSError:
        pass


def gxtb_install(install_dir=None, verbose=True, overwrite=False):
    """
    Download and install the g-xTB-enabled xtb binary and parameter files.

    The binary is downloaded from the kangmg/g-xtb repository and placed in
    ``install_dir`` (default: ``~/bin``). Parameter files are updated inside
    the ``gxtb/parameters/`` package directory so that GXTBHOME is resolved
    automatically by the calculator.

    Parameters
    ----------
    install_dir : str or Path, optional
        Directory to install the xtb binary. Defaults to ~/bin.
    verbose : bool, default=True
        Print progress messages.
    overwrite : bool, default=False
        Overwrite existing binary/parameter files.
    """
    sys_name, machine = _platform_key()
    key = (sys_name, machine)

    if key not in _BINARY_MAP:
        raise RuntimeError(
            f"No pre-built binary available for {sys_name}/{machine}. "
            f"Supported platforms: {list(_BINARY_MAP.keys())}"
        )

    archive_name, exe_name = _BINARY_MAP[key]
    install_dir = (
        Path(install_dir).expanduser() if install_dir else Path.home() / 'bin'
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    exe_path = install_dir / exe_name

    # --- Download, verify, and extract binary ---
    if exe_path.exists() and not overwrite:
        if verbose:
            print(f"Binary already exists: {exe_path}  (use overwrite=True to replace)")
    else:
        archive_url = _BINARY_BASE_URL + archive_name
        sha_url = archive_url + '.sha256'
        tmp_archive = install_dir / archive_name

        if verbose:
            print(f"Downloading {archive_url} ...")
        try:
            urllib.request.urlretrieve(archive_url, tmp_archive)

            _check_lfs_pointer(tmp_archive)

            # Verify SHA256 checksum when available
            try:
                with urllib.request.urlopen(sha_url) as resp:
                    expected_sha = resp.read().decode().strip().split()[0]
                actual_sha = _sha256_of_file(tmp_archive)
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"SHA256 mismatch for {archive_name}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
                if verbose:
                    print("SHA256 verified OK")
            except urllib.error.URLError:
                if verbose:
                    print("Warning: could not fetch SHA256 checksum; skipping verification")

            if verbose:
                print(f"Extracting {archive_name} ...")

            if archive_name.endswith(('.tar.xz', '.tar.gz')):
                with tarfile.open(tmp_archive) as tf:
                    for member in tf.getmembers():
                        if member.name == exe_name or member.name.endswith('/' + exe_name):
                            src = tf.extractfile(member)
                            if src is None:
                                continue
                            with open(exe_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                            break
                    else:
                        raise RuntimeError(
                            f"Could not find '{exe_name}' inside {archive_name}"
                        )
            elif archive_name.endswith('.zip'):
                with zipfile.ZipFile(tmp_archive) as zf:
                    for name in zf.namelist():
                        if name == exe_name or name.endswith('/' + exe_name):
                            with zf.open(name) as src, open(exe_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                            break
                    else:
                        raise RuntimeError(
                            f"Could not find '{exe_name}' inside {archive_name}"
                        )
        finally:
            tmp_archive.unlink(missing_ok=True)

        exe_path.chmod(0o755)
        if verbose:
            print(f"Installed: {exe_path}")

    # --- Update bundled parameter files ---
    _PARAM_DIR.mkdir(exist_ok=True)
    for fname in _PARAM_FILES:
        dest = _PARAM_DIR / fname
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
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    if str(install_dir) not in path_dirs:
        os.environ['PATH'] = str(install_dir) + os.pathsep + os.environ.get('PATH', '')
        if verbose:
            print(f"Added {install_dir} to PATH for this session")

    if verbose:
        print(f"\ng-xTB installation complete.")
        print(f"  Binary : {exe_path}")
        print(f"  Params : {_PARAM_DIR}")
        print(f"  Usage  : gxTB(command='{exe_path}')")


def find_xtb_binary():
    """
    Return the path to the g-xTB-enabled xtb binary, or None if not found.

    Searches PATH for 'xtb'. Does not verify that the binary actually
    supports --gxtb; install via gxtb_install() to get the correct build.
    """
    return shutil.which('xtb')
