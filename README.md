# parmap
Parallel Map (A python replacement for Matlab's parfor)


This is a python equivalent to Matlab's parfor function. Runs
trivially parallisable problems in multiple parallel processes.

A problem is trivially parallisable is each iteration of the loop
can be computed independently of every other iteration

## Examples

```python
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
```


## Installation
The file parmap.py is stand alone. Simply copy it into your source code. A proper pip install is a work in progress.
