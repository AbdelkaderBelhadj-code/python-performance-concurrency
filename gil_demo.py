"""The GIL (Global Interpreter Lock): in CPython only ONE thread runs Python bytecode at a time.

  * CPU-bound work (pure Python math): threads give NO speed-up. Processes do: each has its own
    interpreter and its own GIL (but starting processes and sending data to them costs time).
  * I/O-bound work (waiting for network, disk, database...): the GIL is released while waiting,
    so threads DO speed things up - and they are much cheaper than processes.
"""
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

WORKERS = 4


def cpu_task(n: int) -> int:
    """Pure Python number crunching: holds the GIL the whole time."""
    total = 0
    for i in range(n):
        total += i * i % 7
    return total


def io_task(seconds: float) -> float:
    """Simulated I/O (a network call, a disk read...): waiting releases the GIL."""
    time.sleep(seconds)
    return seconds


def timed(func, jobs: list, executor_class=None) -> tuple[float, list]:
    """Run func on every job: one after the other, or spread over a pool of WORKERS."""
    start = time.perf_counter()
    if executor_class is None:
        results = [func(job) for job in jobs]
    else:
        with executor_class(max_workers=WORKERS) as pool:  # the "with" block waits for all jobs
            results = list(pool.map(func, jobs))
    return time.perf_counter() - start, results


def run() -> None:
    workloads = [
        ("CPU-bound", cpu_task, [6_000_000] * WORKERS),
        ("I/O-bound", io_task, [0.25] * WORKERS),
    ]
    print(f"{WORKERS} jobs per line. CPU job = 6,000,000 loop iterations. I/O job = wait 0.25 s.")
    print(f"{'':<11}{'sequential':>12}{f'{WORKERS} threads':>18}{f'{WORKERS} processes':>18}")
    for name, func, jobs in workloads:
        sequential, expected = timed(func, jobs)
        threads, thread_results = timed(func, jobs, ThreadPoolExecutor)
        processes, process_results = timed(func, jobs, ProcessPoolExecutor)
        assert thread_results == expected and process_results == expected  # parallel must not change results
        print(f"{name:<11}{sequential:>11.2f}s"
              f"{threads:>10.2f}s ({sequential / threads:3.1f}x)"
              f"{processes:>10.2f}s ({sequential / processes:3.1f}x)")
    print("-> CPU-bound: threads ~1x (the GIL). Processes are faster (one GIL each), but each worker")
    print("   must start a new interpreter and re-import the modules first (costly on Windows).")
    print("-> I/O-bound: threads ~4x (the GIL is released while waiting); processes pay the start-up cost again.")
