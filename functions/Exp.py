import numpy as np

from functions.Function import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
