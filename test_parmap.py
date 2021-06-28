# -*- coding: utf-8 -*-

"""
Created on Mon Aug  3 13:58:38 2020

TODO
o I need more control over maxtask, chunksize, etc.
o Better reporting of backtraces
@author: fergal
"""


import multiprocessing
import numpy as np
import functools
import itertools


def default_error_response(func, task):
    """Helper function"""
    return None


def parmap(func, *args, fargs=None,
               single_process=False,
               timeout_sec=1e6,
               on_error=default_error_response):
    """Apply map in parallel.

    This is a python equivalent to Matlab's parfor function. Runs
    trivially parallisable problems in multipe parallel processes

    Examples
    ----------

    .. code-block::

        x = np.arange(5)

        #Parallise a call to a function with one argument
        def sqr(x):
            return x*x
        parmap(sqr, x)
        >>> [0, 1, 4, 9, 16, 25]

        #Parallelise a function with two arguments
        def hypot(x, y):
            return np.sqrt(x**2 + y**2)

        #hypot is called on every combination of x[i] and x[j].
        #result has one hundred elements
        parmap(hypot, x, x)
        >>> [0, 1, 2, ... 7.071]

        #Parallelise a function with  a configuration option
        def power(x, n):
            return x**n

        #parmap works accepts both positional and keyword arguments as keyword arguments
        function_args = dict(n=3)
        result = parmap(hypot, x, fargs=function_args)

        def hypotn(x, y, n):
            return x**n + y**n
        result = parmap(hypot, x, x, fargs=function_args)


    Arguments
    -----------
    func
        The function to apply. Decorated functions may raise errors
        due to the way Python passes the function to the child processes

    *args
        One or more iterables. `func` is applied to every combination
        of elements. For example, if args = [(1,2), (3,4)], the func
        is called 4 times, with arguments of (1,3), (1,4), (2,3), (2,4)
    fargs
        (dict) A dictionary of non-iterable arguments. See examples above
    single_process
        (boolean) Apply the function serially in this current process.
        Useful for debugging (see error handling below)
    timeout_sec
        (int) Kill a task if it does not return after this many seconds.
        The default value is 11 days. If your tasks take that long you
        need to refactor your code.
    on_error
        (function) Error handling function. See below.


    Returns
    ---------
    A list of return values from the function.


    Error Handling
    ---------------
    The single_process flag controls the response to an exception raised
    inside of the called function. If single_process is **True**, parmap
    passes the exception up to the caller. This allows you to run
    the debugger and investigate the source.

    Debugging errors in parallel processes is more difficult. If single_process
    is **False**, parmap replaces the result for that task with the
    result of calling the function specified by `on_error`. The default
    error handler returns None, but you can write your own error handler
    to return what you like.

    Bear in mind that the error handler is not called in a separate processor,
    and is expected to return quickly. See `default_error_response()` for
    the signature of the error handler


    Note on Small Tasks
    ------------------
    For every short tasks (such as those shown in the example) the cost
    of forking a new process can exceed the runtime of the process. This
    function only results in speed gains for functions that take more than
    about half a second to run.
    """
    if fargs is None:
        fargs = {}

    tasks = list(itertools.product(*args))
    pfunc = functools.partial(func, **fargs)
    pfunc.__name__ = func.__name__

    if single_process:
        results = []
        for task in tasks:
            results.append(pfunc(*task))
    else:
        results = parallel_apply(pfunc, tasks, timeout_sec, on_error)
    return results


def parallel_apply(pfunc, tasks, timeout_sec, on_error):
    ncpu = multiprocessing.cpu_count() - 1

    results = []
    with multiprocessing.Pool(ncpu, maxtasksperchild=1) as pool:
        process_list  = map(lambda x: pool.apply_async(pfunc, x), tasks)
        process_list = list(process_list)  #Start the tasks running

        for i in range(len(process_list)):
            p = process_list[i]
            try:
                r = p.get(timeout=timeout_sec)
            except Exception as e:
                warn_on_error(pfunc, tasks[i], e)
                r = on_error(pfunc, tasks[i])

            results.append(r)
    return results


def warn_on_error(func, task, error):
    msg = "WARN: Function %s on task %s failed with error: '%s'"
    msg = msg % (func.__name__, str(task), repr(error))




#testing code
import pytest

def sqr(x):
    print("The squre of %i is %i" %(x, x**2))
    return x*x


def power(x, n):
    return x**n


def hypot(x, y):
    return np.sqrt(x**2 + y**2)

def hypotn(x, y, n):
    return x**n + y**n


def test_sqr():
    x = np.arange(10)

    for sp in [True, False]:
        res = parmap(sqr, x, single_process=sp)
        assert np.all(res == x **2)


def test_pow():
    x = np.arange(10)

    for sp in [True, False]:
        res = parmap(power, x, fargs={'n':3}, single_process=sp)
        assert np.all(res == x **3)

def test_hypot():
    x = np.arange(10)
    y = 10 + np.arange(10)

    for sp in [True, False]:
        res = parmap(hypot, x, y, single_process=sp)
        assert len(res) == 100
    return res


def test_hypotn():
    x = np.arange(10)
    y = np.arange(10)

    for sp in [True, False]:
        res = parmap(hypotn, x, y, fargs={'n':3}, single_process=sp)
        assert len(res) == 100
        assert res[-1] == 2*9**3


def failing_task(x):
    if x == 2:
        1/0
    return x

def test_task_that_fails():
    x = np.arange(5)

    with pytest.raises(ZeroDivisionError):
        res = parmap(failing_task, x, single_process=True)

    res = parmap(failing_task, x, single_process=False)
    assert res[2] is None
    assert res[1] == 1
