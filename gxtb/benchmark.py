"""
Parallel scaling benchmark for the gxTB calculator.
"""
import time
import warnings
from typing import Dict, List, Optional

from ase import Atoms

from .calculator import gxTB


def benchmark_parallel(
    atoms: Atoms,
    nprocs_list: List[int],
    repeat: int = 1,
    warmup: bool = False,
    plot: bool = False,
    calc_kwargs: Optional[dict] = None,
) -> Dict[int, float]:
    """
    Measure wall-clock time for a single-point g-xTB calculation across
    different OpenMP thread counts and report speedup and efficiency.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to benchmark. A copy is used for each run.
    nprocs_list : list of int
        Thread counts to test. The first entry is used as the baseline
        for speedup calculation.
    repeat : int, default=1
        Number of timed repetitions per thread count. The average is
        reported.
    warmup : bool, default=False
        If True, run one untimed calculation before benchmarking starts
        (helps amortise binary loading and filesystem caching overhead).
    plot : bool, default=False
        If True, display a speedup and efficiency plot via matplotlib.
    calc_kwargs : dict, optional
        Extra keyword arguments forwarded to gxTB() (e.g. charge, uhf,
        gxtbhome, command). The 'nprocs' key is ignored here — use
        nprocs_list to control thread counts.

    Returns
    -------
    dict
        Mapping of nprocs → average wall-clock time in seconds.

    Examples
    --------
    >>> results = benchmark_parallel(atoms, [1, 2, 4, 8], repeat=3, plot=True)
    """
    if not nprocs_list:
        raise ValueError("nprocs_list must not be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    kw = dict(calc_kwargs or {})
    kw.pop('nprocs', None)  # nprocs is controlled by nprocs_list

    if warmup:
        print("Running warmup calculation ...", flush=True)
        _time_single(atoms, nprocs_list[0], kw)

    timings: Dict[int, float] = {}
    for nprocs in nprocs_list:
        runs = []
        for i in range(repeat):
            print(
                f"  nprocs={nprocs}  run {i + 1}/{repeat} ...",
                end='\r', flush=True,
            )
            runs.append(_time_single(atoms, nprocs, kw))
        timings[nprocs] = sum(runs) / len(runs)

    print()  # clear the \r line
    _print_table(atoms, timings, repeat)

    if plot:
        _plot(timings)

    return timings


def _time_single(atoms: Atoms, nprocs: int, calc_kwargs: dict) -> float:
    a = atoms.copy()
    a.calc = gxTB(nprocs=nprocs, **calc_kwargs)
    t0 = time.perf_counter()
    a.get_potential_energy()
    return time.perf_counter() - t0


def _print_table(atoms: Atoms, timings: Dict[int, float], repeat: int) -> None:
    nprocs_list = sorted(timings)
    t_base = timings[nprocs_list[0]]
    formula = atoms.get_chemical_formula()
    n_atoms = len(atoms)

    col_w = 45
    print(f"\nParallel Scaling Benchmark")
    print(f"Molecule: {formula}  |  {n_atoms} atoms  |  repeat={repeat}")
    print("─" * col_w)
    print(f" {'nprocs':>6} │ {'time (s)':>10} │ {'speedup':>8} │ {'efficiency':>10}")
    print("─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 12)
    for nprocs in nprocs_list:
        t = timings[nprocs]
        speedup = t_base / t
        efficiency = speedup / nprocs * 100
        print(
            f" {nprocs:>6} │ {t:>10.3f} │ {speedup:>7.2f}× │ {efficiency:>9.1f}%"
        )
    print("─" * col_w)


def _plot(timings: Dict[int, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; install it to use plot=True.",
            RuntimeWarning,
        )
        return

    nprocs_list = sorted(timings)
    t_base = timings[nprocs_list[0]]
    speedups = [t_base / timings[n] for n in nprocs_list]
    efficiencies = [t_base / timings[n] / n * 100 for n in nprocs_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(nprocs_list, speedups, 'o-', label='actual')
    ax1.plot(nprocs_list, nprocs_list, '--', color='gray', alpha=0.6, label='ideal')
    ax1.set_xlabel('nprocs')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(nprocs_list, efficiencies, 's-', color='tab:orange')
    ax2.axhline(100, linestyle='--', color='gray', alpha=0.6, label='ideal')
    ax2.set_xlabel('nprocs')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency')
    ax2.set_ylim(0, 115)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('g-xTB Parallel Scaling', y=1.02)
    plt.tight_layout()
    plt.show()
