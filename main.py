"""Python performance & concurrency: run every demo, or just one.

    python main.py                  # all demos
    python main.py gil              # one demo: pipeline | gil | vectorize | analyze
    python main.py analyze --csv path/to/latencies.csv   # a fresh run of the C++ pipeline
"""
import argparse
import time
from pathlib import Path

import gil_demo
import latency_analysis
import py_pipeline
import vectorization

DEMOS = ["pipeline", "gil", "vectorize", "analyze"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo", nargs="?", choices=DEMOS, help="run only this demo")
    parser.add_argument("--csv", type=Path, default=latency_analysis.DEFAULT_CSV,
                        help="latency CSV written by the C++ pipeline (default: data/latencies_cpp.csv)")
    args = parser.parse_args()
    selected = [args.demo] if args.demo else DEMOS

    python_samples: dict[str, list[int]] = {}
    for demo in selected:
        start = time.perf_counter()
        if demo == "pipeline":
            print("\n=== 1. The C++ pipeline in Python: 2 threads + queue.Queue ===")
            python_samples = py_pipeline.run()
        elif demo == "gil":
            print("\n=== 2. The GIL: threads vs processes ===")
            gil_demo.run()
        elif demo == "vectorize":
            print("\n=== 3. Optimization: loops vs NumPy, and a better algorithm ===")
            vectorization.run()
        elif demo == "analyze":
            print("\n=== 4. Latency statistics: C++ samples (+ Python samples if demo 1 ran) ===")
            latency_analysis.run(args.csv, python_samples)
        print(f"({time.perf_counter() - start:.2f} s)")


# REQUIRED here: on Windows (and macOS) worker processes start by re-importing this file.
# Without the guard, every worker would run main() again and start its own workers...
if __name__ == "__main__":
    main()
