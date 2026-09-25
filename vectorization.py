"""Making Python fast for data: two different kinds of optimization.

1. Vectorization (same algorithm, faster execution): Python loop vs comprehension vs NumPy.
   Why NumPy wins:
     * a list stores POINTERS to float objects (32 bytes each, scattered in memory); a NumPy array
       stores raw 8-byte doubles side by side (contiguous memory: cache friendly, SIMD friendly);
     * the loop runs in compiled C, not in the interpreter: no type check, no reference counting
       and no bytecode dispatch for every element.
2. A better algorithm (less work): a moving average in O(n * w) vs O(n).
   No amount of vectorization saves a bad algorithm when the data grows.
"""
import math
import random
import sys
import time

import numpy as np


def make_prices(n: int, seed: int = 42) -> list[float]:
    """Geometric random walk: each price is the previous one times exp(small random return)."""
    rng = random.Random(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * math.exp(rng.gauss(0.0, 0.001)))
    return prices


# ---- 1. Volatility (standard deviation of log-returns), three ways -------------------------------

def volatility_loop(prices: list[float]) -> float:
    """Plain Python: one interpreted iteration per element."""
    returns = []
    for i in range(1, len(prices)):
        returns.append(math.log(prices[i] / prices[i - 1]))
    mean = sum(returns) / len(returns)
    squared = 0.0
    for r in returns:
        squared += (r - mean) ** 2
    return math.sqrt(squared / (len(returns) - 1))


def volatility_comprehension(prices: list[float]) -> float:
    """Idiomatic Python: shorter and clearer, but about the same speed - still one Python object per element."""
    returns = [math.log(b / a) for a, b in zip(prices, prices[1:])]
    mean = sum(returns) / len(returns)
    return math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))


def volatility_numpy(prices: np.ndarray) -> float:
    """Vectorized: whole-array operations, the loops run in C."""
    returns = np.diff(np.log(prices))  # log(p[i]) - log(p[i-1]) == log(p[i] / p[i-1])
    return float(returns.std(ddof=1))  # ddof=1 -> divide by (n - 1), like the loops above


# ---- 2. Moving average over a window of w values ----------------------------------------------

def moving_average_naive(xs: list[float], w: int) -> list[float]:
    """O(n * w): re-adds the whole window for every position."""
    return [sum(xs[i - w + 1:i + 1]) / w for i in range(w - 1, len(xs))]


def moving_average_running(xs: list[float], w: int) -> list[float]:
    """O(n): slide the window - add the value that enters, subtract the value that leaves."""
    window_sum = sum(xs[:w])
    averages = [window_sum / w]
    for i in range(w, len(xs)):
        window_sum += xs[i] - xs[i - w]
        averages.append(window_sum / w)
    return averages


def moving_average_numpy(xs: np.ndarray, w: int) -> np.ndarray:
    """O(n) and vectorized: sum of a window = difference of two cumulative sums."""
    cumulative = np.concatenate(([0.0], np.cumsum(xs)))
    return (cumulative[w:] - cumulative[:-w]) / w


def best_time(func, *args, repeat: int = 3) -> tuple[float, object]:
    """Best of `repeat` runs: the minimum is the least disturbed by other programs."""
    best, result = math.inf, None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args)
        best = min(best, time.perf_counter() - start)
    return best, result


def print_table(rows: list[tuple[str, float]], baseline: float) -> None:
    for name, seconds in rows:
        print(f"  {name:<34}{seconds * 1000:>9.1f} ms{baseline / seconds:>9.1f}x")


def run(n: int = 1_000_000, window: int = 100) -> None:
    prices = make_prices(n)
    array = np.array(prices)  # the conversion is paid once, outside the timings

    print(f"1) Volatility of {n:,} prices")
    t_loop, v_loop = best_time(volatility_loop, prices)
    t_comp, v_comp = best_time(volatility_comprehension, prices)
    t_np, v_np = best_time(volatility_numpy, array)
    print_table([("Python for-loop", t_loop), ("comprehension + sum()", t_comp), ("NumPy (vectorized)", t_np)], t_loop)
    print(f"  same result from all three: {v_loop:.8f} / {v_comp:.8f} / {v_np:.8f}")
    list_mb = (sys.getsizeof(prices) + sum(sys.getsizeof(p) for p in prices)) / 1e6
    print(f"  memory: Python list = {list_mb:.0f} MB, NumPy array = {array.nbytes / 1e6:.0f} MB")

    m = 200_000
    print(f"\n2) Moving average, window = {window}, on {m:,} prices")
    t_naive, _ = best_time(moving_average_naive, prices[:m], window, repeat=1)
    t_running, _ = best_time(moving_average_running, prices[:m], window)
    t_np_ma, _ = best_time(moving_average_numpy, array[:m], window)
    print_table([("naive O(n*w) Python", t_naive), ("running sum O(n) Python", t_running),
                 ("cumulative sum O(n) NumPy", t_np_ma)], t_naive)
    print("-> better algorithm: fewer operations. Vectorization: each operation is cheaper. Combine both.")
