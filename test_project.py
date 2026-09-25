"""Tests. Run:  python -m unittest -v

A fast version is only useful if it gives the SAME answer as the simple version:
most tests compare an optimized implementation against a straightforward one.
"""
import csv
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

import gil_demo
import latency_analysis
import py_pipeline
import vectorization


class PipelineTest(unittest.TestCase):
    def test_every_tick_arrives(self):
        latencies = py_pipeline.run_pipeline(num_ticks=300, gap_us=50, pacing="sleep")
        self.assertEqual(len(latencies), 300)
        self.assertTrue(all(latency >= 0 for latency in latencies))


class ConcurrencyTest(unittest.TestCase):
    def test_threads_and_processes_give_the_same_results(self):
        jobs = [10_000, 20_000, 30_000]
        expected = [gil_demo.cpu_task(n) for n in jobs]
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(gil_demo.cpu_task, jobs)), expected)
        with ProcessPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(gil_demo.cpu_task, jobs)), expected)


class VectorizationTest(unittest.TestCase):
    def setUp(self):
        self.prices = vectorization.make_prices(5_000)

    def test_volatility_versions_agree(self):
        expected = vectorization.volatility_loop(self.prices)
        self.assertAlmostEqual(vectorization.volatility_comprehension(self.prices), expected, places=12)
        self.assertAlmostEqual(vectorization.volatility_numpy(np.array(self.prices)), expected, places=12)

    def test_moving_average_versions_agree(self):
        expected = vectorization.moving_average_naive(self.prices, 20)
        self.assertTrue(np.allclose(vectorization.moving_average_running(self.prices, 20), expected))
        self.assertTrue(np.allclose(vectorization.moving_average_numpy(np.array(self.prices), 20), expected))


class LatencyAnalysisTest(unittest.TestCase):
    def test_nearest_rank_matches_the_cpp_definition(self):
        values = np.arange(1, 1001)  # 1..1000, same check as the C++ unit test
        self.assertEqual(latency_analysis.nearest_rank(values, 50), 500)
        self.assertEqual(latency_analysis.nearest_rank(values, 99), 990)
        self.assertEqual(latency_analysis.nearest_rank(values, 99.9), 999)
        self.assertEqual(latency_analysis.nearest_rank(values, 100), 1000)

    def test_bootstrap_ci_contains_the_true_median(self):
        data = np.random.default_rng(1).exponential(scale=100.0, size=5_000)
        low, high = latency_analysis.bootstrap_ci(data, 50, resamples=200)
        true_median = 100.0 * np.log(2)  # median of an exponential = scale * ln(2)
        self.assertLess(low, true_median)
        self.assertGreater(high, true_median)

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latencies.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows([["queue", "seq", "latency_ns"], ["A", 0, 100], ["A", 1, 300], ["B", 0, 50]])
            data = latency_analysis.load_latencies(path)
        self.assertEqual(sorted(data), ["A", "B"])
        self.assertEqual(data["A"].tolist(), [100, 300])


if __name__ == "__main__":
    unittest.main()
