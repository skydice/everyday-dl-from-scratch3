from dezero.functions.Add import add
from dezero.functions.Div import div, rdiv
from dezero.functions.Mul import mul
from dezero.functions.Neg import neg
from dezero.functions.Pow import pow_
from dezero.functions.Sub import sub
from dezero.variable import Variable


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow_
