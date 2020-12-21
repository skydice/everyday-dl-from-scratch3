import numpy as np

from functions.Exp import Exp
from functions.Function import Function
from functions.Square import Square
from Variable import Variable


def test_variable():
    data = np.array(1.0)
    x = Variable(data)
    assert x.data == 1.0


def test_function():
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    assert y.data == 100


def test_function_chain():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.data == 1.648721270700128
