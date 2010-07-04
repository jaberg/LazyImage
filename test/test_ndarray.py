import numpy
from lazyimage import (lnumpy, compute, function, Symbol, symbol, set_value, get_value,
    MissingValue)
import lazyimage

NDS = lnumpy.NdarraySymbol

print 'nonlazy'
assert numpy.allclose(lnumpy.tanh(5), numpy.tanh(5)), lnumpy.tanh(5)

print 'lazy'
assert isinstance(lnumpy.tanh(NDS.blank()), Symbol)

s = NDS.blank()
print 'compute with missing variable'
try:
    compute(lnumpy.tanh(s))
    assert False
except MissingValue:
    pass

print 'test you cant set a nonexistant variable'
try:
    lazyimage.default_closure.set_value(s,5)
    assert False
except KeyError:
    pass

print 'compute tanh'
lazyimage.default_closure.init_value(s,5)
assert numpy.allclose(compute(lnumpy.tanh(s)), numpy.tanh(5))

print 'compute adding'
assert numpy.allclose(compute(lnumpy.tanh(6 - s)), numpy.tanh(1))

s = symbol(value=5, ctor=NDS.blank)
lazy_output = lnumpy.tanh(6 - s)
assert numpy.allclose(compute(lazy_output), numpy.tanh(1))
set_value(s, 8)
assert numpy.allclose(compute(lazy_output ), numpy.tanh(-2))

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

# test that the opencl optimization can be disabled



