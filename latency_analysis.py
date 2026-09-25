"""Statistics on latency data: the samples written by the C++ project (and by the Python pipeline).

Questions we answer:
  1. What does the distribution look like?  percentiles, mean vs median, log-scale histogram
  2. How sure are we about the p99?          bootstrap confidence interval
  3. Is there a warm-up effect?              first ticks vs the rest
  4. Final comparison                        C++ lock-free vs C++ mutex vs Python
"""
import csv
import math
from pathlib import Path

import numpy as np

# Samples measured by the C++ pipeline (repo: cpp-low-latency-pipeline). Pass --csv to analyze a fresh run.
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "latencies_cpp.csv"
PERCENTILES = {"p50": 50.0, "p90": 90.0, "p99": 99.0, "p99.9": 99.9}


def format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    return f"{ns / 1_000_000:.2f} ms"


def load_latencies(path: Path) -> dict[str, np.ndarray]:
    """CSV rows "queue,seq,latency_ns" -> {queue name: latencies in arrival order}."""
    columns: dict[str, list[int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            columns.setdefault(row["queue"], []).append(int(row["latency_ns"]))
    return {name: np.array(values, dtype=np.int64) for name, values in columns.items()}


def nearest_rank(values: np.ndarray, p: float) -> float:
    """Same definition as the C++ code: the smallest sample with at least p% of samples <= it.
    np.partition finds the k-th smallest value in O(n) without sorting everything (quickselect)."""
    k = min(max(math.ceil(p * len(values) / 100), 1), len(values)) - 1  # 0-based position once sorted
    return float(np.partition(values, k)[k])


def summarize(latencies) -> dict[str, float]:
    values = np.asarray(latencies)
    summary = {"mean": float(values.mean()), "max": float(values.max())}
    for name, p in PERCENTILES.items():
        summary[name] = nearest_rank(values, p)
    return summary


def bootstrap_ci(values: np.ndarray, p: float, resamples: int = 300, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of a percentile. We have ONE sample, so we simulate new ones by
    resampling it WITH replacement; the spread of the statistic across resamples ~ its uncertainty."""
    rng = np.random.default_rng(seed)
    stats = [nearest_rank(rng.choice(values, size=len(values)), p) for _ in range(resamples)]
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def log_histogram(values: np.ndarray, width: int = 36) -> str:
    """Buckets that double in size (64-128 ns, 128-256 ns...): latency spans several orders of magnitude,
    so equal-width buckets would put almost everything in the first bar."""
    exponents = np.floor(np.log2(np.maximum(values, 1))).astype(int)
    lowest = exponents.min()
    counts = np.bincount(exponents - lowest)
    lines = []
    for i, count in enumerate(counts):
        low = 2.0 ** (lowest + i)
        bar = "#" * math.ceil(width * count / counts.max())  # ceil: even 1 sample stays visible
        lines.append(f"    {format_ns(low):>8} - {format_ns(2 * low):<8}{count:>9,}  {bar}")
    return "\n".join(lines)


def describe(name: str, values: np.ndarray, histogram: bool = True) -> dict[str, float]:
    s = summarize(values)
    print(f"\n--- {name}: {len(values):,} samples ---")
    print("  " + " | ".join(f"{key} {format_ns(s[key])}" for key in ("mean", "p50", "p90", "p99", "p99.9", "max")))
    print(f"  mean / median = {s['mean'] / s['p50']:.1f}  (> 1: a long right tail pulls the mean up)")
    low, high = bootstrap_ci(values, 99.0)
    print(f"  p99 = {format_ns(s['p99'])}, 95% bootstrap CI [{format_ns(low)}, {format_ns(high)}]")
    head = max(len(values) // 100, 1)  # the first 1% of ticks
    if len(values) > 2 * head:
        print(f"  warm-up: p50 of the first {head:,} ticks = {format_ns(nearest_rank(values[:head], 50))}, "
              f"of the rest = {format_ns(nearest_rank(values[head:], 50))}")
    if histogram:
        print(log_histogram(values))
    return s


def run(csv_path: Path = DEFAULT_CSV, extra: dict[str, list[int]] | None = None) -> None:
    datasets: dict[str, np.ndarray] = {}
    if csv_path.exists():
        print(f"Loading C++ samples from {csv_path}")
        datasets.update({f"C++ {name}": values for name, values in load_latencies(csv_path).items()})
    else:
        print(f"No C++ data at {csv_path}: run the C++ pipeline and pass --csv (analyzing Python samples only).")
    datasets.update({name: np.array(values) for name, values in (extra or {}).items()})
    if not datasets:
        print("Nothing to analyze.")
        return

    summaries = {name: describe(name, values, histogram=name.startswith("C++")) for name, values in datasets.items()}

    fastest = min(summaries, key=lambda name: summaries[name]["p50"])
    print(f"\nFinal comparison (x = p50 relative to {fastest})")
    for name, s in sorted(summaries.items(), key=lambda item: item[1]["p50"]):
        ratio = s["p50"] / summaries[fastest]["p50"]
        print(f"  {name:<22} p50 {format_ns(s['p50']):>9}   p99 {format_ns(s['p99']):>9}   {ratio:>8,.0f}x")
