import numpy as np

from functions.Exp import Exp
from functions.Function import Function
from functions.Square import Square
from Variable import Variable
from functions.diff import numerical_diff


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


def test_numerical_diff():
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)

    assert dy == 4.000000000004


def test_composite_function_diff():
    def f(x):
        A = Square()
        B = Exp()
        C = Square()

        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)

    assert dy == 3.2974426293330694


def test_backward_calculte():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    assert x.grad == 3.297442541400256
