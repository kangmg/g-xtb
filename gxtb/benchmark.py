"""
Parallel scaling benchmark for the gxTB calculator.
"""
import time
import warnings
from typing import Dict, List, Optional

from ase import Atoms

from .calculator import gxTB

_VALID_TASKS = ('energy', 'forces', 'hessian')


def benchmark_parallel(
    atoms: Atoms,
    nprocs_list: List[int],
    task: str = 'energy',
    repeat: int = 1,
    warmup: bool = False,
    plot: bool = False,
    calc_kwargs: Optional[dict] = None,
) -> Dict[int, float]:
    """
    Measure wall-clock time for a g-xTB calculation across different OpenMP
    thread counts and report speedup.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to benchmark. A copy is used for each run.
    nprocs_list : list of int
        Thread counts to test. The first entry is used as the baseline
        for speedup calculation.
    task : {'energy', 'forces', 'hessian'}, default='energy'
        Which calculation to benchmark:
        - 'energy'  : single-point energy only (xtb --gxtb)
        - 'forces'  : energy + analytic gradient (xtb --gxtb --grad)
        - 'hessian' : numerical Hessian via analytic gradients (xtb --gxtb --hess)
    repeat : int, default=1
        Number of timed repetitions per thread count. The average is
        reported.
    warmup : bool, default=False
        If True, run one untimed calculation before benchmarking starts
        (helps amortise binary loading and filesystem caching overhead).
    plot : bool, default=False
        If True, display a speedup plot via matplotlib.
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
    >>> results = benchmark_parallel(atoms, [1, 2, 4, 8], task='energy', repeat=3)
    >>> results = benchmark_parallel(atoms, [1, 2, 4, 8], task='forces', repeat=3)
    >>> results = benchmark_parallel(atoms, [1, 2, 4],    task='hessian', repeat=2, plot=True)
    """
    if not nprocs_list:
        raise ValueError("nprocs_list must not be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got {task!r}")

    kw = dict(calc_kwargs or {})
    kw.pop('nprocs', None)  # nprocs is controlled by nprocs_list

    if warmup:
        print(f"Running warmup ({task}) ...", flush=True)
        _time_single(atoms, nprocs_list[0], task, kw)

    timings: Dict[int, float] = {}
    for nprocs in nprocs_list:
        runs = []
        for i in range(repeat):
            print(
                f"  nprocs={nprocs}  run {i + 1}/{repeat} ...",
                end='\r', flush=True,
            )
            runs.append(_time_single(atoms, nprocs, task, kw))
        timings[nprocs] = sum(runs) / len(runs)

    print()  # clear the \r line
    _print_table(atoms, timings, task, repeat)

    if plot:
        _plot(timings, task)

    return timings


def _time_single(atoms: Atoms, nprocs: int, task: str, calc_kwargs: dict) -> float:
    a = atoms.copy()
    calc = gxTB(nprocs=nprocs, **calc_kwargs)
    a.calc = calc
    t0 = time.perf_counter()
    if task == 'energy':
        a.get_potential_energy()
    elif task == 'forces':
        a.get_forces()
    elif task == 'hessian':
        calc.get_hessian(a)
    return time.perf_counter() - t0


def _print_table(atoms: Atoms, timings: Dict[int, float], task: str, repeat: int) -> None:
    nprocs_list = sorted(timings)
    t_base = timings[nprocs_list[0]]
    formula = atoms.get_chemical_formula()
    n_atoms = len(atoms)

    col_w = 34
    print(f"\nParallel Scaling Benchmark  [{task}]")
    print(f"Molecule: {formula}  |  {n_atoms} atoms  |  repeat={repeat}")
    print("─" * col_w)
    print(f" {'nprocs':>6} │ {'time (s)':>10} │ {'speedup':>8}")
    print("─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 10)
    for nprocs in nprocs_list:
        t = timings[nprocs]
        speedup = t_base / t
        print(f" {nprocs:>6} │ {t:>10.3f} │ {speedup:>7.2f}×")
    print("─" * col_w)


def _linear_fit(x: List[int], y: List[float]) -> List[float]:
    """Return least-squares linear-fit y values evaluated at the input x values.

    For a single input point, the same y value is returned because a slope
    cannot be estimated from one sample.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not x:
        return []
    if len(x) < 2:
        return [y[0]]

    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    if ss_xx == 0:
        return [y_mean for _ in x]
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return [slope * xi + intercept for xi in x]


def _plot(timings: Dict[int, float], task: str) -> None:
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
    fitted_speedups = _linear_fit(nprocs_list, speedups)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(nprocs_list, speedups, 'o-', label='actual')
    ax.plot(nprocs_list, fitted_speedups, '--', color='tab:red', alpha=0.9, label='linear fit')
    ax.set_xlabel('nprocs')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'g-xTB Parallel Scaling  [{task}]', y=1.02)
    plt.tight_layout()
    plt.show()
