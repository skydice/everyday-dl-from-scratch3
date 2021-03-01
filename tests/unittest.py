import unittest

import numpy as np

from dezero.functions.Square import square
from dezero.functions.diff import numerical_diff
from dezero.variable import Variable


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class ComplexFunctionTest(unittest.TestCase):
    @staticmethod
    def sphere(x, y):
        z = x ** 2 + y ** 2
        return z

    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = self.sphere(x, y)
        z.backward()

        self.assertEqual(x.grad, 2.0)
        self.assertEqual(y.grad, 2.0)

    @staticmethod
    def matyas(x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = self.matyas(x, y)
        z.backward()

        self.assertAlmostEqual(x.grad, 0.04)
        self.assertAlmostEqual(y.grad, 0.04)

    @staticmethod
    def goldstein(x, y):
        z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
            (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
        return z

    def test_goldstein(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = self.goldstein(x, y)
        z.backward()

        self.assertEqual(x.grad, -5376.0)
        self.assertEqual(y.grad, 8064.0)
