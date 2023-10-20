from parmap.implentation import parmap  # noqa F401

# Doesn't work in Python 3.7
try:
    from importlib.metadata import version

    __version__ = version("python-parmap")
except ImportError:
    __version__ = "unknown"
