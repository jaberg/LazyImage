import numpy
from lazyimage import (lnumpy, function, Symbol, symbol, set_value, get_value,
    MissingValue, get_default_closure)
import lazyimage

closure = get_default_closure()

NDS = lnumpy.NdarraySymbol
NDS.new(closure)
assert len(closure.elements) == 1, closure.elements

print 'nonlazy'
assert numpy.allclose(lnumpy.tanh(5), numpy.tanh(5)), lnumpy.tanh(5)

print 'lazy'
assert isinstance(lnumpy.tanh(NDS.new(closure)), Symbol)

s = NDS.new(closure)
assert len(closure.elements) == 4, closure.elements
print 'compute with missing variable'
r = lnumpy.tanh(s)
print r.closure.elements
try:
    (lnumpy.tanh(s)).compute()
    assert False
except AttributeError,e: #tanh not found in None
    assert 'tanh' in str(e)

print 'compute tanh'
closure.set_value(s,3)
print s._value
assert numpy.allclose((lnumpy.tanh(s)).compute(), numpy.tanh(3))

print 'compute adding'
assert numpy.allclose((lnumpy.tanh(6 - s)).compute(), numpy.tanh(3))

s = symbol(value=5, ctor=NDS.blank)
lazy_output = lnumpy.tanh(6 - s)
assert numpy.allclose((lazy_output).compute(), numpy.tanh(1))
set_value(s, 8)
assert numpy.allclose((lazy_output).compute(), numpy.tanh(-2))

f = function([s], lazy_output )
assert numpy.allclose(f(1), numpy.tanh(5))
assert numpy.allclose(f(3), numpy.tanh(3))
assert numpy.allclose(f(7), numpy.tanh(-1))


a1 = numpy.random.randn(4).astype('int32')

# importing registers opencl optimizations
from lazyimage import lnumpy_opencl
import lazyimage
assert lnumpy_opencl.replace_numpy_with_opencl in lazyimage.transform_db

# test that the opencl tanh function works
cl_tanh = lnumpy_opencl.impl_from_fnname('tanh')
assert numpy.allclose(cl_tanh(a1), numpy.tanh(a1))

# test that the opencl tanh optimization works
f = function([s], lazy_output )
f.printprog()
assert isinstance(list(f.expr_graph.expr_iter())[1].impl, lnumpy_opencl.UnaryElemwiseCpu)
assert numpy.allclose(f(1), numpy.tanh(5))
assert numpy.allclose(f(7), numpy.tanh(-1))

# test that the opencl optimization can be disabled
lnumpy_opencl.replace_numpy_with_opencl.enabled=False
f = function([s], lazy_output )
f.printprog()
assert not isinstance(list(f.expr_graph.expr_iter())[1].impl, lnumpy_opencl.UnaryElemwiseCpu)
assert numpy.allclose(f(1), numpy.tanh(5))



