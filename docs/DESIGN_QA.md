# Design Q&A — Python Performance, Concurrency & Latency Analysis

Questions an engineer or interviewer might ask about this project, and the Python, concurrency, NumPy and
latency-statistics concepts behind it. Performance answers always say what was *measured*, not guessed.

Contents:
1. [Questions about the project](#1-questions-about-the-project)
2. [Python concurrency: GIL, threads, processes, asyncio](#2-python-concurrency-gil-threads-processes-asyncio)
3. [Python language fundamentals](#3-python-language-fundamentals)
4. [NumPy, pandas and data processing](#4-numpy-pandas-and-data-processing)
5. [Performance and profiling in Python](#5-performance-and-profiling-in-python)
6. [Latency statistics and data analysis](#6-latency-statistics-and-data-analysis)
7. [Live-coding exercises](#7-live-coding-exercises-with-short-solutions)

---

## 1. Questions about the project

**1. Why rewrite the C++ pipeline in Python?**
To run the *same* experiment in both languages, so the difference measures only the language and runtime. It answers a real
design question: which parts of a system belong in C++ (the hot path), and which in Python (orchestration, research, analysis).

**2. Why does a Python handoff cost ~28 µs instead of 75 ns?**
- `queue.Queue` uses a lock and condition variables, so a waiting consumer *sleeps* and has to be woken by the OS.
- The woken thread must then take the GIL back.
- Every step is interpreted bytecode, with an object allocated per tick and reference counting.
The C++ SPSC consumer spins on a cache line, with no system calls and no allocations.

**3. Explain what happened in "spin" mode.**
The producer busy-waits in a Python loop, so it **never releases the GIL**. When a tick arrives, the consumer wakes up at the
OS level but can't run Python code until it gets the GIL. It waits for the switch interval, then asks the producer to drop it.
Ticks pile up in the meantime, so latency becomes milliseconds. The same busy-wait is harmless in C++, where threads really run
in parallel.

**4. How did you *prove* the GIL was the cause?**
With a controlled experiment: I changed only one variable, the switch interval (`sys.setswitchinterval(0.0001)`). The latency
collapsed from about 9 ms to about 16 µs. If the queue itself were slow, the switch interval wouldn't matter.

**5. What is the switch interval?**
`sys.getswitchinterval()` defaults to 5 ms. If a thread has waited that long for the GIL, it asks the holder to release it, and
the holder drops it at its next check. It trades latency (shorter) against throughput (fewer switches, less overhead).

**6. Why did 1, 2, 5 and 10 ms switch intervals give the same ~8 ms median on Windows?**
The interpreter waits for the GIL with a timeout in whole milliseconds, and Windows wakes waiting threads on its timer tick
(15.6 ms by default). So any timeout from 1 to 15 ms really lasts until the next tick, about 8 ms on average. Below 1 ms the
timeout rounds down to 0, so the request is immediate, and that's why 0.1 ms gave 16 µs. I checked this by measuring
7 intervals, and above the tick the latency grows with the interval (20 ms → 14.5 ms, 40 ms → 32 ms).

**7. Why did 4 processes give only 2.2× and not 4× on the CPU-bound work?**
- **Start-up cost:** on Windows, processes are *spawned*. Each worker starts a new interpreter and re-imports `main.py` and its
  imports (including NumPy), which cost about 0.4 s in total here.
- Only 4 physical cores, shared with the OS and other programs.
- Turbo frequency drops when all cores are busy.
- Arguments and results are pickled to cross between processes.
With bigger jobs, the fixed start-up cost matters less and the speed-up gets closer to 4×.

**8. Why did threads give 4× on the I/O-bound work?**
`time.sleep` (like waiting for a socket, a file or a database) releases the GIL, so the 4 waits overlap. Processes would work
too but pay the start-up cost again (0.54 s vs 0.25 s).

**9. So when do you use threads, processes, or asyncio?**
| Workload | Best tool | Why |
|---|---|---|
| Many network/disk waits (tens to thousands) | `asyncio` (or threads) | cheap concurrency while waiting; asyncio handles 10,000+ connections on one thread |
| Some blocking I/O and legacy libraries | threads (`ThreadPoolExecutor`) | the GIL is released while waiting, and it's simple |
| CPU-heavy pure Python | processes (`ProcessPoolExecutor`) | one GIL per process, true parallelism |
| CPU-heavy numeric work | NumPy / numba / C++ extension (+ threads) | compiled code releases the GIL and is 10–100× faster anyway |
| Lowest latency | not Python, or Python only as a control layer | [cpp-low-latency-pipeline](https://github.com/AbdelkaderBelhadj-code/cpp-low-latency-pipeline) |

**10. Why is NumPy 21× faster for the volatility?**
A Python list holds pointers to separate float objects (8-byte pointer + 24-byte object = 32 bytes per number, scattered in
memory). A NumPy array holds raw 8-byte doubles next to each other. NumPy loops run in C: there's no type check, no
reference-count update and no bytecode dispatch per element, and they can use SIMD. The project shows 32 MB vs 8 MB for
1 million prices.

**11. Why wasn't the list comprehension faster than the loop?**
The comprehension only speeds up building the list (no `append` lookup each time). The rest (`math.log` calls, the generator
of squared deviations) costs the same. Recent CPython versions (3.11+) also sped up plain loops. Lesson: idiomatic doesn't
automatically mean fast, so measure.

**12. Why is the O(n) moving average faster? Is there any risk?**
The naive version re-adds 100 values for every position (n·w operations). The sliding version adds one value and subtracts
one (2n operations). The risk is **floating-point drift**: rounding errors accumulate in the running sum over millions of
updates. Fix it by recomputing the sum exactly from time to time, or with `math.fsum`. The cumulative-sum trick has a similar
issue: subtracting two huge, close numbers loses precision on long series.

**13. How do you know the optimised versions are correct?**
Tests compare every fast version with the simple one (`assertAlmostEqual` to 12 decimals, `np.allclose`). A fast wrong answer
is worthless. The program also prints the three volatility results side by side.

**14. How do you time code reliably?**
Use `time.perf_counter()` (monotonic, high resolution), the same input for every version, and conversions (list → array) kept
outside the timing. Repeat, and keep the **minimum** (best-of-N: the run least disturbed by other programs), or use `timeit`.
Warm up first, avoid a busy machine, and report the spread when it matters.

**15. Why nearest-rank percentiles? Why do your numbers match the C++ output exactly?**
Nearest rank always returns a real sample (no interpolation), which is common for latency reports. The C++ and Python code use
the *same formula*, `rank = ceil(p × n / 100)`, so the results are identical to the nanosecond. That's a nice cross-language
check of both programs.

**16. Tell me about the numpy percentile surprise.**
`np.percentile(x, 99.9, method="inverted_cdf")` uses the same definition but gave a different p99.9. For n = 100,000 it
computes `99.9/100 = 0.9990000000000001` first, so `ceil(n·q) = 99,901` instead of 99,900. At exact boundaries, the *order*
of floating-point operations changes the result. Lesson: when two implementations must agree exactly, use the same formula,
and test the boundaries.

**17. What does the bootstrap CI for p99 tell you? Why is p99.9 less certain?**
The p99 of the SPSC queue is 116 ns, with a 95% CI of [115, 117] ns. With 100,000 samples, about 1,000 of them lie above p99,
so it's precise. p99.9 rests on only about 100 samples beyond it, and max on a single one, so those estimates are much noisier.
Rule of thumb: to estimate p99.9 well, you need well over 1,000 samples in the tail, i.e. at least a few hundred thousand runs.

**18. Why a log-scale histogram?**
The latencies go from about 20 ns to 4 ms, over 5 orders of magnitude. With equal-width buckets everything lands in the first
bar, and the tail is invisible. Buckets that double in size (like an HDR histogram) show the fast bulk *and* the tail. They
revealed the two peaks of the mutex queue.

**19. What did the warm-up check show?**
For the SPSC queue, the first 1,000 ticks had a p50 of 37 ns vs 76 ns afterwards. My hypothesis is that the two threads first
ran on sibling hyper-threads (shared L1/L2), and the OS later moved one to another core. The pinned benchmark in [cpp-low-latency-pipeline](https://github.com/AbdelkaderBelhadj-code/cpp-low-latency-pipeline)
supports it: 5.6 ns vs 35 ns per item. Always look at latency *over time*, not only its global distribution.

**20. Why the `csv` module and not pandas?**
To keep NumPy as the only dependency, and because the file is simple. In real work I'd use pandas or polars:
`pd.read_csv(path).groupby("queue")["latency_ns"].describe(percentiles=[.5, .9, .99])`. Note that pandas interpolates
percentiles by default, so its values can differ slightly from nearest rank.

**21. How would you make the Python pipeline faster?**
Don't busy-wait in Python. Batch several ticks per `put`. Use `multiprocessing` with shared memory for CPU work. Move the hot
loop into C++ with pybind11, or use numba. Try the free-threaded build (3.13t). Or keep Python out of the hot path and let it
control the system and analyse its results.

**22. Why is `if __name__ == "__main__":` *required* in `main.py`?**
With the *spawn* start method (Windows and macOS), each worker process imports the main module. Without the guard, every
worker would run `main()` again and try to create its own workers (Python stops this with a `RuntimeError`).

**23. What would change if the consumer were a process instead of a thread?**
You'd use `multiprocessing.Queue`. Every tick would be **pickled**, sent through a pipe and unpickled: more latency, but no GIL
contention. For big arrays you'd use `multiprocessing.shared_memory` to avoid copying.

**24. What is the main takeaway of the three projects together?**
Measure before optimising, and choose the tool per layer: C++ for the nanosecond hot path, Python and NumPy for research,
analysis and orchestration, and statistics to know whether a difference is real.

---

## 2. Python concurrency: GIL, threads, processes, asyncio

**What is the GIL? Why does it exist?**
The Global Interpreter Lock is a mutex in CPython that lets only one thread execute Python bytecode at a time. It protects the
interpreter's internal state, above all the *reference counts* of every object. It makes single-threaded code fast and C
extensions simple to write. The cost is that pure-Python threads can't use several cores.

**When is the GIL released?**
- During blocking I/O (files, sockets), `time.sleep`, and waiting on locks and queues.
- Inside many C extensions while they compute (NumPy on large arrays, `hashlib`, `zlib`, compression, many database drivers).
- Every switch interval (5 ms), when another thread asks for it.

**Does the GIL make my code thread-safe?**
**No.** It protects the interpreter, not *your* invariants. A read-modify-write (read `counter`, add 1, write it back) is
several steps, and if the thread switches between the read and the write, updates are lost. Use `threading.Lock`.
A subtlety I measured on Python 3.13: the classic `counter += 1` loop gave the *correct* total 5 times out of 5, because
CPython 3.10+ only switches threads at specific points (loop back-edges, function calls). With one function call between the
read and the write, 4 threads × 20,000 increments gave only about 20,200 instead of 80,000. The bug was always there, it was
just hidden, and a small code change or the free-threaded build exposes it.

**Are operations like `list.append` or `dict[key] = value` thread-safe?**
In CPython, single operations on built-in types are atomic because of the GIL. Compound operations (`if key not in d: d[key] = x`)
are not. Don't rely on these details, especially with free-threaded Python. Use locks or `queue.Queue`.

**What is free-threaded Python?**
PEP 703: a CPython build without the GIL (`python3.13t`). It's experimental in 3.13 and officially supported (still optional)
from 3.14. It uses per-object locking and biased reference counting. Pure-Python threads can then use all cores, but
single-threaded code is a bit slower and C extensions must be made compatible.

**What are sub-interpreters?**
Several interpreters in one process, each with its own GIL (PEP 684). Python 3.14 exposes them in `concurrent.interpreters` and
`InterpreterPoolExecutor`: parallelism without separate processes, but objects aren't shared freely between interpreters.

**Threads vs processes: the key differences?**
| | Threads | Processes |
|---|---|---|
| Memory | shared, so fast communication but needs locks | separate, so it's safe but data must be pickled or shared explicitly |
| CPU parallelism in CPython | no (the GIL) | yes |
| Start-up cost | ~50–100 µs | ~10–100+ ms (spawn) |
| A crash | kills the whole process | kills only that worker |

**What is `asyncio`? When is it the right tool?**
Cooperative multitasking on **one thread**. Coroutines (`async def`) give control back to the event loop at every `await`, and
the loop resumes whichever coroutine's I/O is ready. It's ideal for thousands of concurrent network operations (web clients and
servers, websockets, market-data connections), because a coroutine is much cheaper than a thread. It gives no CPU parallelism.

**What is the biggest `asyncio` pitfall?**
A blocking call (`time.sleep`, `requests.get`, heavy CPU work) inside a coroutine blocks the **whole** event loop. Use
`await asyncio.sleep`, async libraries (`aiohttp`), or `await asyncio.to_thread(blocking_function)`.

**`asyncio.gather` vs `asyncio.create_task`?**
`create_task` schedules a coroutine to run concurrently and returns a Task. `gather(*coroutines)` runs several and waits for all
their results. Use `asyncio.Semaphore` to limit how many run at once (for example 10 concurrent HTTP requests).

**What does `concurrent.futures` give you?**
A single interface: `ThreadPoolExecutor` and `ProcessPoolExecutor`, with `submit(fn, *args)` → a `Future` (`.result()`),
`map(fn, items)`, and `as_completed(futures)`. Switching from threads to processes is one word, which the project uses.

**What are the `multiprocessing` start methods?**
- `fork`: copies the parent process. It's fast, but dangerous if threads exist, because locks held by other threads are copied
  in the locked state, which can deadlock.
- `spawn`: a fresh interpreter that re-imports the main module. It's safe but slow. It's the default on Windows and macOS.
- `forkserver`: forks from a clean server process. It's the default on Linux since Python 3.14 (it was `fork` before).

**How do processes share data?**
`multiprocessing.Queue`/`Pipe` (pickled messages), `multiprocessing.shared_memory` (raw buffers, for example backing a NumPy
array with no copy), `Value`/`Array` (shared ctypes), a `Manager` (proxies: flexible but slow), and files or databases.

**Why must functions sent to a `ProcessPoolExecutor` be defined at module level?**
They're pickled *by reference* (module name + function name), and the worker imports them. Lambdas and nested functions can't
be pickled that way.

**Name the synchronisation primitives in `threading`.**
- `Lock` (mutex) and `RLock` (re-entrant, the same thread can acquire it again).
- `Semaphore` (at most N holders).
- `Event` (a flag that threads wait on).
- `Condition` (wait until notified, which is what `queue.Queue` uses internally).
- `Barrier` (wait until N threads arrive).
Always use them with `with lock:`.

**`queue.Queue` vs `collections.deque` vs `multiprocessing.Queue` vs `asyncio.Queue`?**
- `queue.Queue`: thread-safe, blocking, with maxsize, for threads.
- `deque`: fast O(1) at both ends. `append`/`popleft` are atomic in CPython but can't block or wait, so it's not a full
  producer/consumer queue.
- `multiprocessing.Queue`: between processes, pickles every item.
- `asyncio.Queue`: for coroutines, not thread-safe.

**What is a daemon thread?**
A thread that doesn't keep the program alive: when only daemon threads remain, Python exits and kills them abruptly (no
cleanup). Use it for background helpers, never for work that must finish, like writing files.

**How can Python threads deadlock? How do you avoid it?**
Thread 1 holds lock A and waits for B, while thread 2 holds B and waits for A. Avoid it with a fixed lock order, one lock
instead of two, timeouts (`lock.acquire(timeout=1)`), and minimal work while holding a lock.

**Can threads speed up NumPy code?**
Often, yes. Many NumPy operations release the GIL on large arrays, so several threads can compute in parallel. BLAS operations
like matrix multiplication are already multi-threaded internally.

**What is `threading.local()`?**
An object whose attributes differ per thread. It's useful for per-thread connections or buffers without locking.

---

## 3. Python language fundamentals

**Mutable vs immutable?**
Immutable objects can't change after creation: `int`, `float`, `str`, `tuple`, `frozenset`, `bytes`. Mutable ones can: `list`,
`dict`, `set`, and most objects. Only immutable (hashable) objects can be dict keys or set elements. Names are *references*:
`b = a` doesn't copy a list.

**`is` vs `==`?**
`==` compares values (`__eq__`). `is` compares identity (the same object in memory). Use `is` only for `None` (and sentinels,
as in the project's `END_OF_STREAM`). `x is 1000` depends on CPython's caching of small ints (−5 to 256) and string interning.

**Shallow copy vs deep copy?**
`copy.copy(x)`, `list(x)` or `x[:]` create a new outer container that still points to the same inner objects.
`copy.deepcopy(x)` recursively copies everything. Nested lists are the classic bug.

**What is the mutable default argument pitfall?**
`def f(x, items=[])`: the default list is created **once**, when the function is defined, and shared across calls. Use
`items=None` and then `items = [] if items is None else items`.

**`*args` and `**kwargs`? Keyword-only and positional-only parameters?**
`*args` collects extra positional arguments into a tuple and `**kwargs` collects extra keyword arguments into a dict. Parameters
after `*` must be passed by keyword, and parameters before `/` must be positional.

**Explain LEGB scope, `global`, `nonlocal`, closures.**
Names are looked up in the Local, then Enclosing function, then Global (module), then Built-in scopes. `global` and `nonlocal`
allow assignment to outer variables. A closure is an inner function that remembers the enclosing variables. Classic bug:
`[lambda: i for i in range(3)]` all return 2, because `i` is looked up when the lambda is *called*. Fix: `lambda i=i: i`.

**What is a decorator?**
A function that takes a function and returns a new one, usually wrapping it: `@timer def f(): ...` means `f = timer(f)`. Use
`functools.wraps` to keep the name and docstring. Examples: timing, caching (`@functools.lru_cache`), retry, logging,
`@dataclass`, `@property`.

**Generators and `yield`: why use them?**
A generator produces values lazily, one at a time, and keeps its state between them. It uses O(1) memory for huge or infinite
streams (reading a big file line by line). `(x*x for x in data)` is a generator expression, while `[x*x for x in data]` builds
the whole list. The project uses one in `sum(rng.random() < rate for _ in range(n))`.

**What are the iterator and iterable protocols?**
An iterable has `__iter__()`, which returns an iterator. An iterator has `__next__()` and raises `StopIteration` at the end.
`for` loops call these for you. Generators are iterators.

**What are context managers?**
Objects used with `with`: `__enter__` runs at the start and `__exit__` always runs at the end, even after an exception. Examples:
files, locks (`with lock:`), the executors in `gil_demo.py` (which wait for all jobs). You can write one with
`@contextlib.contextmanager` and a `yield`.

**List vs tuple vs set vs dict: when to use each? What are their complexities?**
| Type | Use it for | Key operations |
|---|---|---|
| list | an ordered, changeable sequence | index O(1), append O(1) amortised, insert/pop(0) O(n), `in` O(n) |
| tuple | a fixed record, dict keys | like list, but immutable and hashable |
| set | uniqueness, fast membership | add, remove and `in` O(1) on average |
| dict | key → value | get, set and `in` O(1) on average, insertion-ordered (3.7+) |

**How is a dict implemented?**
As a hash table with open addressing: `hash(key)` picks a slot, and collisions probe other slots. It resizes when about 2/3
full, and a compact design keeps insertion order. Keys need `__hash__` and `__eq__`, and objects that are equal must have
equal hashes.

**What do `__slots__` and `@dataclass(slots=True)` do?**
They replace the per-object `__dict__` with fixed slots: less memory per instance (useful for millions of ticks) and faster
attribute access. The downside is that you can't add new attributes at run time.

**`__repr__` vs `__str__`?**
`__repr__` is unambiguous and meant for developers (ideally valid Python). `__str__` is readable, for users. `print` uses
`__str__` and falls back to `__repr__`.

**`@staticmethod` vs `@classmethod` vs `@property`?**
A staticmethod gets no `self` or `cls` (a plain function in the class's namespace). A classmethod receives the class (used for
alternative constructors like `from_csv`). A property is a method accessed like an attribute (computed values, validation).

**What is the MRO? What does `super()` do?**
The Method Resolution Order is the order Python searches classes for an attribute (C3 linearisation, see `Class.__mro__`).
`super()` calls the next class in the MRO, not necessarily the parent, which matters with multiple inheritance.

**Duck typing, ABCs and Protocols?**
Duck typing: "if it has `.read()`, treat it as a file", with no inheritance needed. Abstract base classes (`abc.ABC`) enforce
methods at instantiation. `typing.Protocol` describes the expected methods for static type checkers without inheritance.

**Exceptions: `else` and `finally`? EAFP vs LBYL?**
`else` runs if no exception happened, and `finally` always runs (cleanup). EAFP ("easier to ask forgiveness": `try` it, catch
`KeyError`) is idiomatic Python. LBYL ("look before you leap": `if key in d`) can have race conditions. Never use a bare
`except:`.

**How does Python manage memory?**
Mainly **reference counting**: an object is freed as soon as its count drops to 0. A cyclic garbage collector handles
reference cycles. `del x` only removes a name. Small objects come from an internal allocator (pymalloc). `weakref` refers to an
object without keeping it alive.

**Python `int` vs NumPy `int64`?**
A Python `int` has arbitrary precision and never overflows (but is slow). A NumPy `int64` is a fixed 64-bit integer: fast, but it
**wraps around silently** on overflow in arrays.

**Why is `s += piece` in a loop slow? What is the fix?**
Strings are immutable, so each `+=` may copy the whole string: O(n²) overall. Collect the pieces in a list and use
`"".join(pieces)` once, which is O(n).

**`sorted()` vs `list.sort()`?**
`sorted` returns a new list from any iterable, and `.sort()` sorts in place and returns `None`. Both use Timsort: stable,
O(n log n), fast on partially sorted data. Use `key=` (for example `key=lambda t: t.price`).

**What are virtual environments for?**
An isolated folder with its own interpreter and packages, one per project, so versions don't conflict. Create one with
`python -m venv .venv`, record dependencies in `requirements.txt` or `pyproject.toml`, and make builds reproducible with pinned
versions or a lock file.

**Are type hints checked at run time?**
No. They're documentation plus static analysis (`mypy`, `pyright`, IDEs). This project uses modern syntax: `list[int]`,
`dict[str, np.ndarray]`, `X | None` (3.10+).

---

## 4. NumPy, pandas and data processing

**What is an ndarray, really?**
A block of contiguous memory plus metadata: `dtype` (the element type), `shape`, and `strides` (how many bytes to jump to move
along each axis). Many operations (transpose, slicing) only change the metadata, with no copy.

**Views vs copies?**
Basic slicing (`a[10:20]`, `a[:, 0]`) returns a **view**, so modifying it modifies the original. Fancy indexing (`a[[1, 5, 7]]`)
and boolean masks (`a[a > 0]`) return **copies**. Check with `np.shares_memory(a, b)`, and use `.copy()` when you need
independence.

**What is broadcasting?**
It combines arrays of different shapes without copying. Comparing shapes from the right, dimensions must be equal or 1, and a
size-1 dimension is stretched. Example: `(1000, 3) - (3,)` subtracts a row of means from every row.

**What does the `axis` argument mean?**
The axis that gets *collapsed*. `a.mean(axis=0)` averages down the rows (one result per column), and `axis=1` averages across
the columns (one result per row).

**What is vectorization? What is a ufunc?**
Expressing an operation on whole arrays instead of looping in Python. A ufunc (universal function: `np.log`, `np.add`, ...)
applies element-wise in compiled C, with broadcasting and often SIMD.

**How do you handle missing values?**
In NumPy, NaN propagates (`np.mean` gives nan), so use `np.nanmean` and friends. `NaN != NaN`, so test with `np.isnan`.
In pandas: `isna`, `fillna` (with a value, forward fill `ffill` for time series), `dropna`. Decide *why* data is missing before
filling it.

**Random numbers in NumPy: `default_rng` vs `np.random.seed`?**
`rng = np.random.default_rng(seed)` creates an independent generator object (PCG64), which is recommended: explicit and
reproducible. `np.random.seed` sets the legacy global state. Same idea as `random.Random` vs `random.seed` in [python-probability-stats-lab](https://github.com/AbdelkaderBelhadj-code/python-probability-stats-lab).

**`np.partition` vs `np.sort`?**
`np.sort` is O(n log n). `np.partition(a, k)` puts the k-th smallest value in position k in O(n) (quickselect, like C++'s
`nth_element`). The project uses it for percentiles inside the bootstrap loop.

**Why is `np.append` in a loop a bad idea?**
Each call allocates a new array and copies everything, so the loop is O(n²). Append to a Python list and convert once, or
pre-allocate with `np.empty(n)` and fill by index (as `bootstrap_ci` does with a list).

**C order vs Fortran order?**
C order is row-major (rows are contiguous) and Fortran order is column-major. Iterate along the contiguous axis to use the
cache well. A transposed array is non-contiguous, and `np.ascontiguousarray` fixes that when needed.

**Simple returns vs log returns?**
The simple return is p₁/p₀ − 1. The log return is ln(p₁/p₀). Log returns **add up over time** (the log return over a week is the
sum of the daily ones), are symmetric, and suit models with normal noise. For small moves they're almost equal. The project
computes volatility from log returns.

**How do you annualise volatility?**
Multiply by √(periods per year): σ_annual = σ_daily × √252 (trading days). Variance grows linearly with time (independent
returns), so the standard deviation grows with √time.

**pandas: `loc` vs `iloc`?**
`loc` selects by *labels* (the index and column names, and slices include the end). `iloc` selects by integer *positions*
(the end is excluded).

**Explain `groupby`.**
Split-apply-combine: split the rows by key, apply an aggregation (`mean`, `quantile`, `agg`, `transform`) to each group, and
combine the results. Example: `df.groupby("queue")["latency_ns"].quantile(0.99)`.

**What types of merge/join are there?**
`inner` (keys in both), `left` (all left rows), `right`, `outer` (all keys), and `cross`. Watch out for duplicated keys, which
multiply rows. Check sizes before and after, and use `validate="one_to_one"`. `pd.merge_asof` is the time-series join
("last quote before each trade").

**Why is `df.apply(func, axis=1)` slow?**
It calls a Python function once per row, which is a Python loop in disguise. Use vectorised column operations, `np.where`,
`np.select`, or `groupby` aggregations instead.

**How do you reduce a DataFrame's memory?**
Use the right dtypes: `category` for repeated strings (like the queue name), smaller numeric types (`float32`, `int32`) when the
precision allows, and load only the needed columns (`usecols`) with explicit `dtype=`.

**How do you process a file larger than memory?**
Stream it: `pd.read_csv(path, chunksize=1_000_000)`, or a generator over lines, keeping running aggregates (counts, sums,
Welford, histograms). Or use columnar formats (Parquet: compressed, read only some columns), `np.memmap`, polars' lazy mode,
DuckDB, Dask or Spark, or a database.

**CSV vs Parquet?**
CSV is text, row-oriented and human-readable, but slow to parse, large, and has no types. Parquet is binary, columnar,
compressed and typed: much faster and smaller, and you read only the columns you need. Use it for serious data work.

**Time-series operations in pandas?**
`resample("1min").mean()` changes the frequency, `rolling(20).std()` gives moving statistics, `shift(1)` gives lags (compute
returns with `pct_change()`), `ewm(span=20)` gives exponential weighting, and time zones need `tz_localize`/`tz_convert`.
Never use future rows to compute today's feature (look-ahead bias).

**pandas vs polars?**
Polars is written in Rust on Apache Arrow: multi-threaded, with a lazy query optimiser, and often 5–10× faster with lower
memory. pandas has the biggest ecosystem. It's useful to know both exist and why.

---

## 5. Performance and profiling in Python

**Which clock should you use to time code?**
`time.perf_counter()` (or `perf_counter_ns()`) is monotonic and has the highest resolution: use it for durations.
`time.time()` is wall-clock time, which can jump, so use it for timestamps. `time.process_time()` counts CPU time of this
process only (it excludes sleep).

**How do you find where a Python program spends its time?**
- `cProfile` (`python -m cProfile -s cumtime main.py`): time per function.
- `line_profiler`: time per line.
- `py-spy`: a sampling profiler that attaches to a running process with no code change and makes flame graphs.
- `tracemalloc` / `memray`: memory.
Profile first, then optimise the proven hot spot.

**Why is pure Python slow?**
Every value is a heap object with a type pointer and a reference count, every operation is dynamically dispatched by the
interpreter loop, and nothing is compiled for the specific types. A simple `a + b` costs tens of ns in Python vs under 1 ns in C.

**What is the ladder of Python speed-ups, from cheapest to most effort?**
1. A better algorithm or data structure (a set instead of a list: O(1) vs O(n)).
2. Built-ins and the standard library (`sum`, `sorted`, `collections`, `itertools`), which are written in C.
3. Caching (`functools.lru_cache` / `functools.cache`).
4. Vectorization with NumPy or pandas.
5. A JIT: numba (`@njit` on numeric loops), or PyPy.
6. Cython, or a C/C++ extension with pybind11 or nanobind.
7. Parallelism: processes, or threads around GIL-releasing code.

**What are the complexity traps with Python containers?**
`list.insert(0, x)` and `list.pop(0)` are O(n), so use `collections.deque`. `x in list` is O(n), so use a set. String `+=` in a
loop is O(n²), so use `join`. `np.append` in a loop is O(n²). Slicing copies (`xs[i-w+1:i+1]` in the naive moving average
copies 100 values each time).

**Are there micro-optimizations that matter in hot loops?**
Local variables are faster than globals (bind `log = math.log` before the loop), avoid repeated attribute lookups, avoid
creating objects inside the loop, and use comprehensions to build lists. But first, try to remove the Python loop altogether.

**What did CPython 3.11–3.13 change for speed?**
3.11 added the specialising adaptive interpreter (PEP 659): about 1.25× faster on average, and cheaper function calls. 3.12
and 3.13 brought more specialisation, and 3.13 has an experimental JIT (PEP 744) and the optional free-threaded build.

**When should you *not* optimise?**
When the code isn't the bottleneck (Amdahl's law: speeding up 5% of the runtime saves at most 5%), when it's run once, or when
it hurts readability for no measured gain. "Premature optimisation is the root of all evil", but measured optimisation of the
hot path is engineering.

---

## 6. Latency statistics and data analysis

**How many definitions of "percentile" are there?**
Hyndman and Fan list nine. The main split is between *nearest rank* (returns a real sample, the C++ and Python code here) and
*interpolation* (numpy and pandas default to linear). They agree closely for large n but differ for small n and at the tail.
State which one you use, and use the same formula everywhere you compare (see the numpy gotcha above).

**Can you average p99s from several servers or time windows?**
**No.** Percentiles don't aggregate: the mean of p99s isn't the p99 of the combined data. Keep the raw samples, or use
*mergeable* summaries: HDR histograms (add the bucket counts) or t-digests.

**What is an HDR histogram?**
A histogram with logarithmic buckets split into linear sub-buckets, which keeps a fixed *relative* precision (for example 0.1%)
from nanoseconds to hours in a few KB of fixed memory. Recording is O(1), and histograms can be added together. It's the
standard way to record latency in production.

**How do you compare two latency distributions properly?**
Compare **several percentiles**, not just the means, with bootstrap confidence intervals for the differences. Use rank-based
tests (Mann–Whitney U), or Kolmogorov–Smirnov for the whole shape. A t-test on means is fragile with skewed, heavy-tailed data.
And look at the histograms: two peaks tell a story that no summary number does.

**Should you remove outliers from latency data?**
Usually no: in latency, the outliers *are* the user experience (and the SLA). Investigate them (GC pauses, page faults,
scheduling). Only remove what's truly invalid, such as samples taken during warm-up, and report that separately.

**How do you detect and handle warm-up?**
Plot or summarise latency over time (in chunks of arrival order). The first chunks are often different (cold caches, lazy
initialisation, JIT, frequency ramp-up). Discard a warm-up period *explicitly*, or report it separately, but never silently.

**What does a latency-vs-load curve look like?**
Flat at low load, then a "hockey stick": as utilisation approaches 100%, queueing delay explodes (M/M/1: W = 1/(μ − λ)).
Capacity planning keeps systems well below the knee.

**What is an SLO? And an SLA?**
An SLO (Service Level Objective) is an internal target, like "p99 < 10 ms over 30 days". An SLA (Service Level Agreement) is a
contractual promise, with penalties. Both are usually written in percentiles, not averages.

**How do you estimate the uncertainty of a percentile?**
With the bootstrap (as in the project), or with order-statistic confidence intervals (the binomial distribution gives which
ranks bracket the true percentile). Uncertainty grows fast toward the tail: few samples lie beyond p99.9.

**What is the mean/median ratio good for?**
A quick skewness check. Close to 1 means roughly symmetric. The mutex queue's ratio of 3.3 means a heavy right tail, so the
mean alone would mislead.

**Explain Little's law with an example.**
Items in the system L = arrival rate λ × average time in the system W. If 200,000 ticks/s arrive and each waits 5 µs on
average, then on average 1 tick is in the queue. In burst mode a full queue (1,024 items) at 8.5 M items/s means about 120 µs of waiting.

---

## 7. Live-coding exercises (with short solutions)

**1. Show a race condition, then fix it**
```python
import threading, time
counter = 0
lock = threading.Lock()

def add_many(n):
    global counter
    for _ in range(n):
        with lock:                 # remove this lock -> about 20,000 instead of 80,000 (lost updates)
            value = counter        # read
            time.sleep(0)          # any function call here lets another thread run
            counter = value + 1    # write back: overwrites the other threads' updates

threads = [threading.Thread(target=add_many, args=(20_000,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)                     # 80000 with the lock
```
(A bare `counter += 1` often *looks* correct on CPython 3.10+, because threads only switch at certain points. That makes the
race hidden, not safe.)

**2. Producer/consumer with `queue.Queue` and a sentinel**
```python
import queue, threading
q, SENTINEL = queue.Queue(maxsize=100), object()

def producer():
    for i in range(10):
        q.put(i)
    q.put(SENTINEL)

def consumer():
    while (item := q.get()) is not SENTINEL:
        print("got", item)

threading.Thread(target=producer).start()
consumer()
```

**3. Parallel CPU work with processes**
```python
from concurrent.futures import ProcessPoolExecutor

def work(n):                       # module level -> picklable
    return sum(i * i for i in range(n))

if __name__ == "__main__":         # required with spawn (Windows/macOS)
    with ProcessPoolExecutor() as pool:
        print(list(pool.map(work, [10**6] * 8)))
```

**4. `asyncio`: 100 "requests" concurrently, at most 10 at a time**
```python
import asyncio

async def fetch(i, limit):
    async with limit:               # semaphore: at most 10 inside
        await asyncio.sleep(0.1)    # stands for a network call
        return i

async def main():
    limit = asyncio.Semaphore(10)
    results = await asyncio.gather(*(fetch(i, limit) for i in range(100)))
    print(len(results))             # ~1 s total instead of 10 s

asyncio.run(main())
```

**5. A timing decorator**
```python
import functools, time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__}: {time.perf_counter() - start:.4f} s")
    return wrapper
```

**6. An LRU cache from scratch**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)          # mark as most recently used
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)   # evict the least recently used
```

**7. A rolling mean over a stream, O(1) per value**
```python
from collections import deque

class RollingMean:
    def __init__(self, window):
        self.values, self.window, self.total = deque(), window, 0.0

    def add(self, x):
        self.values.append(x)
        self.total += x
        if len(self.values) > self.window:
            self.total -= self.values.popleft()
        return self.total / len(self.values)
```

**8. NumPy: returns, rolling volatility, and z-scores**
```python
import numpy as np
prices = np.array([100, 101, 99.5, 102, 103.5, 101])
log_returns = np.diff(np.log(prices))
window = 3
windows = np.lib.stride_tricks.sliding_window_view(log_returns, window)   # a view, no copy
rolling_vol = windows.std(axis=1, ddof=1)
data = np.random.default_rng(0).normal(size=(1000, 3))
zscores = (data - data.mean(axis=0)) / data.std(axis=0)                   # broadcasting per column
```

**9. Group means over a huge CSV, streaming (constant memory)**
```python
import csv
from collections import defaultdict

def group_means(path, key, value):
    sums, counts = defaultdict(float), defaultdict(int)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):       # one row in memory at a time
            sums[row[key]] += float(row[value])
            counts[row[key]] += 1
    return {k: sums[k] / counts[k] for k in sums}

# group_means("data/latencies_cpp.csv", "queue", "latency_ns")
```
