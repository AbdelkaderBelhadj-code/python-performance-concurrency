"""The C++ tick pipeline (repo cpp-low-latency-pipeline) rewritten in Python: producer -> queue.Queue -> consumer.

Same idea, same measurement (latency = receive time - send time), so the two languages can be compared.

The producer must wait between two ticks. We try three ways:
  * "sleep": time.sleep() RELEASES the GIL, so the consumer can run as soon as a tick arrives.
  * "spin" : a busy-wait loop, exactly like the C++ producer. But in Python the spinning thread HOLDS
             the GIL: the consumer has to wait until the interpreter forces a switch, after 5 ms by
             default (sys.getswitchinterval()). On Windows that wait is rounded up to the OS timer tick
             (~15.6 ms), so it is even longer. Latency explodes: the GIL in action.
  * "spin" + a 0.1 ms switch interval: if the latency collapses, we have proved the GIL was the cause.
"""
import queue
import sys
import threading
import time
from dataclasses import dataclass

from latency_analysis import format_ns, summarize

END_OF_STREAM = None  # "poison pill": tells the consumer to stop


@dataclass(slots=True)  # slots: no per-object __dict__ -> less memory, faster attribute access
class Tick:
    seq: int
    price: float
    send_ns: int


def run_pipeline(num_ticks: int, gap_us: float, pacing: str) -> list[int]:
    """Send num_ticks ticks, one every gap_us microseconds. Return each tick's latency in ns."""
    ticks: queue.Queue = queue.Queue(maxsize=1024)  # thread-safe (lock + condition variables inside)
    latencies_ns: list[int] = []

    def producer() -> None:
        gap_ns = int(gap_us * 1_000)
        next_send = time.perf_counter_ns()
        price = 100.0
        for seq in range(num_ticks):
            if pacing == "sleep":
                delay_ns = next_send - time.perf_counter_ns()
                if delay_ns > 0:
                    time.sleep(delay_ns / 1e9)  # the GIL is released while sleeping
            else:
                while time.perf_counter_ns() < next_send:  # busy-wait: the GIL is NOT released
                    pass
            next_send += gap_ns
            price *= 1.0001
            ticks.put(Tick(seq, price, time.perf_counter_ns()))
        ticks.put(END_OF_STREAM)

    def consumer() -> None:
        while True:
            tick = ticks.get()  # blocks (and releases the GIL) while the queue is empty
            received = time.perf_counter_ns()
            if tick is END_OF_STREAM:
                break
            latencies_ns.append(received - tick.send_ns)

    threads = [threading.Thread(target=consumer), threading.Thread(target=producer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return latencies_ns


def run(num_ticks: int = 2_000, gap_us: float = 100.0) -> dict[str, list[int]]:
    default_interval = sys.getswitchinterval()
    configs = [
        ("Python sleep", "sleep", default_interval),
        ("Python spin", "spin", default_interval),
        ("Python spin 0.1ms", "spin", 0.0001),
    ]
    print(f"{num_ticks:,} ticks, one every {gap_us:g} us, 2 threads + queue.Queue "
          f"(default GIL switch interval = {default_interval * 1000:g} ms)")
    print(f"{'run':<20}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}")
    results = {}
    for name, pacing, interval in configs:
        sys.setswitchinterval(interval)
        try:
            latencies = run_pipeline(num_ticks, gap_us, pacing)
        finally:
            sys.setswitchinterval(default_interval)  # always restore the global setting
        s = summarize(latencies)
        print(f"{name:<20}" + "".join(f"{format_ns(s[k]):>10}" for k in ("mean", "p50", "p90", "p99", "max")))
        results[name] = latencies
    print("-> spin: the producer holds the GIL, the consumer only runs when the interpreter forces a switch.")
    print("   With a 50x shorter switch interval the latency collapses: the GIL was the bottleneck, not the queue.")
    return results
