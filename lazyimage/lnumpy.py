"""
A numpy-like module with support for lazy evaluation
"""
import numpy
from .structures import Symbol, Impl, Type

class NdarrayType(Type):
    # None means unknown
    # For things which are not generally true (symmetric, P.S.D)
    # it is supposed to be convenient that unknown and False are both negative in terms of an
    # if statement.  For boolean properties, make sure that False corresponds to the default
    # setting.
    constant = False
    value = None
    dtype = None
    shape = None
    strides=None
    databuffer=None
    contiguous=None
    symmetric=None
    positive_semidefinite=None
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def is_conformant(self, obj):
        if self.constant: return (self.value is obj)
        if type(obj) != numpy.ndarray: return False
        if (self.dtype != None) and obj.dtype != self.dtype: return False
        if (self.shape != None) and obj.shape != self.shape: return False
        if (self.strides != None) and obj.strides != self.strides: return False
        if (self.databuffer != None) and obj.data != self.databuffer: return False
        return True
    def make_conformant(self, obj):
        if self.is_conformant(obj): return obj
        if self.constant: raise TypeError('asdf', (obj, self.value))
        if self.databuffer is not None: raise TypeError()
        if self.dtype:
            nda = numpy.asarray(obj, self.dtype)
        else:
            nda = numpy.asarray(obj)
        if self.shape is not None:
            nda.shape = self.shape
        if self.strides is not None and nda.strides != self.strides:
            raise TypeError()
        return nda
    def eq(self, v0, v1, approx=False):
        if approx:
            return numpy.allclose(v0, v1)
        else:
            return numpy.all(v0 == v1)
    @classmethod
    def as_constant(cls, value):
        #TODO: check for symmetry, psd, contiguity
        rval = cls(
                constant=True, 
                value=value,
                dtype=value.dtype,
                shape=value.shape,
                strides=value.strides,
                databuffer=value.data)
        assert rval.is_conformant(value)
        return rval

class NdarraySymbol(Symbol):
    def __array__(self):
        return numpy.asarray(self.compute())

    def __add__(self, other): return add(self, other)
    def __sub__(self, other): return subtract(self, other)
    def __mul__(self, other): return multiply(self, other)
    def __div__(self, other): return divide(self, other)

    def __radd__(other, self): return add(self, other)
    def __rsub__(other, self): return subtract(self, other)
    def __rmul__(other, self): return multiply(self, other)
    def __rdiv__(other, self): return divide(self, other)

class NdarrayImpl(Impl):
    def outputs_from_inputs(self, inputs):
        closure = inputs[0].closure
        outputs = [NdarraySymbol.new(closure, type=NdarrayType.new()) for o in range(self.n_outputs)]
        return outputs

class Elemwise(NdarrayImpl):
    """
    Base for element-wise Implementations
    """
    def __str__(self):
        return 'NumPy_%s'%self.name

    def as_input(self, closure, obj):
        """Convenience method - it's the default constructor for lazy Impl __call__ methods to
        use to easily turn all inputs into symbols.
        """
        def ndarray_const_caster(o):
            return NdarrayType.as_constant(numpy.asarray(o))
        return super(Elemwise, self).as_input(
                closure, 
                obj,
                constant_type=ndarray_const_caster)

    def infer_type(self, expr, changed):
        """Set output shapes according to numpy broadcasting rules
        """
        super(Elemwise, self).infer_type(expr, changed)
        print 'INFER_TYPE', expr, expr.inputs
        # if all inputs have an ndarray type
        if all(isinstance(i.type, NdarrayType) for i in expr.inputs):
            # get the shapes of the inputs
            shapes = [i.type.shape for i in expr.inputs if i.type.shape is not None]
            print 'SHAPES', expr, shapes, expr.inputs
            # if all the inputs have a known number of dimensions
            if len(shapes) == len(expr.inputs):
                # the outputs has the rank of the highest-rank input
                out_shp = [None]*max(len(s) for s in shapes)
                # left-pad shapes that are too short
                shapes = [[1]*(len(out_shp)-len(s)) + list(s) for s in shapes]
                for dim in range(len(shapes)):
                    dim_known = not any(s[dim] is None for s in shapes)
                    if dim_known:
                        # TODO:
                        # could detect size errors here
                        out_shp[dim]= max(s[dim] for s in shapes)
                out_shp = tuple(out_shp)
                for o in expr.outputs:
                    if o.type.shape != out_shp:
                        o.type.shape = out_shp
                        changed.add(o)


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

