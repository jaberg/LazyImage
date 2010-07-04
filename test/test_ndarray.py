import numpy
from lazyimage import ndarray, compute, function, Closure, Symbol

tanh = ndarray.tanh
NDS = ndarray.NdarraySymbol

print 'nonlazy'
assert numpy.allclose(tanh(5), numpy.tanh(5)), tanh(5)

print 'lazy'
assert isinstance(tanh(NDS()), Symbol)

s = NDS()
print 'compute'
assert numpy.allclose(compute(tanh(s), {s:5}), numpy.tanh(5))

print 'compute adding'
assert numpy.allclose(compute(tanh(6 - s), {s:5}), numpy.tanh(1))

closure = Closure()
s =  closure.symbol(value=5, ctor=NDS)
lazy_output = tanh(6 - s)
assert numpy.allclose(compute(lazy_output, closure), numpy.tanh(1))
#print closure
closure[s] = 8
assert numpy.allclose(compute(lazy_output, closure), numpy.tanh(-2))

f = function([s], lazy_output )
assert numpy.allclose(f(1), numpy.tanh(5))
assert numpy.allclose(f(3), numpy.tanh(3))
assert numpy.allclose(f(7), numpy.tanh(-1))


a1 = numpy.random.randn(4).astype('int32')

from lazyimage import numpy_opencl

cl_tanh = numpy_opencl.impl_from_fnname('tanh')
cl_tanh(a1)
assert numpy.allclose(cl_tanh(a1), numpy.tanh(a1))



