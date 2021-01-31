import numpy as np

from functions.Add import add
from functions.Exp import Exp
from functions.Square import Square, square
from Variable import Variable
from functions.diff import numerical_diff


def test_variable():
    x = Variable(np.array(1.0))
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


def test_backward_calculate():
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


def test_auto_backward_propagation():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))

    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.inputs[0] == b
    assert y.creator.inputs[0].creator == B
    assert y.creator.inputs[0].creator.inputs[0] == a
    assert y.creator.inputs[0].creator.inputs[0].creator == A
    assert y.creator.inputs[0].creator.inputs[0].creator.inputs[0] == x

    y.grad = np.array(1.0)
    y.backward()

    assert x.grad == 3.297442541400256


def test_add_class():
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    assert y.data == 5


def test_square_backward():
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()

    print(z.data)
    print(x.grad)
    print(y.grad)


def test_complex_graph_backward():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    assert y.data == 32.0
    assert x.grad == 64.0
