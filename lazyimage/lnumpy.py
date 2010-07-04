"""
A numpy-like module with support for lazy evaluation
"""
import numpy
from .structures import Symbol, Impl
from .engine import compute

class NdarraySymbol(Symbol):
    def __array__(self):
        return numpy.asarray(compute(self))

    def __add__(self, other): return add(self, other)
    def __sub__(self, other): return subtract(self, other)
    def __mul__(self, other): return multiply(self, other)
    def __div__(self, other): return divide(self, other)

    def __radd__(other, self): return add(self, other)
    def __rsub__(other, self): return subtract(self, other)
    def __rmul__(other, self): return multiply(self, other)
    def __rdiv__(other, self): return divide(self, other)

def numpy_impl(name, n_outputs=1):
    fn = getattr(numpy, name)
    return 

class Elemwise(Impl):
    """
    Base for element-wise Implementations
    """

class NumpyElemwise (Elemwise):
    """
    Base for element-wise Implementations with native numpy implementations
    """
    def __init__(self, name):
        super(NumpyElemwise, self).__init__(
                fn=getattr(numpy,name),
                n_outputs=1,
                name=name)

# Elementwise, Unary functions
# that upcast to float
class Elemwise_unary_float_upcast(NumpyElemwise):
    def __call__(self, x):
        return NumpyElemwise.__call__(self, x)
for name in ['exp', 'log', 'log2', 'log10', 'log1p',
        'tanh', 'cosh', 'sinh', 'tan', 'cos', 'sin']:
    globals()[name] = Elemwise_unary_float_upcast(name)

# Elementwise, binary functions
# that upcast to float
class Elemwise_binary_float_upcast(NumpyElemwise):
    def __call__(self, x, y):
        return NumpyElemwise.__call__(self, x, y)
for name in ['subtract', 'power', 'divide']:
    globals()[name] = Elemwise_binary_float_upcast(name)

# Elementwise, N-ary functions
# that upcast to float
class Elemwise_Nary_float_upcast(NumpyElemwise):
    pass
for name in ['add', 'multiply']:
    globals()[name] = Elemwise_Nary_float_upcast(name)

# Elementwise, range comparisons
class Elemwise_range_cmp(NumpyElemwise):
    def __call__(self, x, y):
        return NumpyElemwise.__call__(self, x, y)
for name in ['greater']:
    globals()[name] = Elemwise_range_cmp(name)

