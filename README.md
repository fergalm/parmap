# Parallel Map
Parmap is a python equivalent to Matlab's parfor function. Parmap runs
trivially parallisable problems in multiple parallel processes.

A problem is trivially parallisable is each iteration of the loop
can be computed independently of every other iteration.

## Examples

```python
from parmap import parmap

x = np.arange(5)


# Parallelise a call to a function with one argument
def sqr(x):
    return x * x


parmap(sqr, x)
>> > [0, 1, 4, 9, 16, 25]


# Parallelise a function with two arguments
def hypot(x, y):
    return np.sqrt(x ** 2 + y ** 2)


# hypot is called on every combination of x[i] and x[j].
# result has one hundred elements
parmap(hypot, x, x)
>> > [0, 1, 2, ... 7.071]


# Parallelise a function with  a configuration option
def power(x, n):
    return x ** n


# parmap works accepts both positional and keyword arguments as keyword arguments
function_args = dict(n=3)
result = parmap(hypot, x, fargs=function_args)


def hypotn(x, y, n):
    return x ** n + y ** n


result = parmap(hypot, x, x, fargs=function_args)
```

## Choosing your method of concurrency
Python is technically a single threaded application that does not allow multiple calculations to be performed at one time. There are three main tricks for getting around this limit.

1. **Multiprocessing:** Multi-processing creates multiple, separate, Python processes on your computer that compete for resources. These processes are completely separate, and can communicate with each other only with some difficulty (the ability to communicate between processes is not exposed by parmap, which assumes the procesess are separate). Multi-processing is best for problems which involve lots of CPU calculations, but not a lot of reading/writing data from disk or the network.

2. **Threading:** In threading mode, multiple tasks take turns using the CPU to complete their work, but spend most of their time asleep. Threads are must cheaper to create than processes, both in terms of memory needed, or time to create. Only one thread can run at a time in Python, so threading is of no advantage for CPU heavy tasks. However, tasks that involve downloading multiple files from the internet spend most of their time waiting anyway, and are ideal for threads.

3. **Asyncio:** Asyncio is similar to threading, but each thread has very fine grain control over when it cedes control of the CPU. Where normal threads are told by the OS when to run and when to stop, asyncio "threadlets" announce when they've reached a good stopping point. Asyncio is more complicated to implement, and not recommended if your code is not already designed with async in mind.

Your choice of concurrency in parmap can be set using the "engine" keyword.

```python
is_prime = parmap(check_if_prime, x, engine="multi")
pdf_list = parmap(download_pdfs, url_list, engine="threads")
```

The "serial" engine disables concurrency and runs the tasks in series with a normal for loop. If one of the tasks throws an uncaught exception, the code halts, allowing you to debug. The other engines skip over failed tasks and try to complete as many as possible.

## Installation
`pip install parmap`

The file `implementation.py` is stand alone. If you prefer, you can simply copy it into your source code.
