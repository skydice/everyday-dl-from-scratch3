import contextlib

import numpy as np

from Config import Config


def as_array(x):
    if np.isscalar(x):
        return np.array(x)

    return x


def no_grad():
    return using_config('enable_backprop', False)


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
