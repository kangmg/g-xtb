"""
Command-line entry points for the gxtb package.
"""
import argparse
import sys


def _install():
    from gxtb import gxtb_install

    parser = argparse.ArgumentParser(
        prog='gxtb-install',
        description='Download and install the g-xTB-enabled xtb binary and parameter files.',
    )
    parser.add_argument(
        '--dir',
        metavar='PATH',
        default=None,
        help='Directory to install the xtb binary (default: ~/bin).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing binary and parameter files.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages.',
    )

    args = parser.parse_args()

    try:
        gxtb_install(
            install_dir=args.dir,
            verbose=not args.quiet,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
