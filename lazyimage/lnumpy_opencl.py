
import numpy
from .structures import Symbol, Impl
from . import lnumpy, transform

import pyopencl as cl


#
# TODO: How to handle contexts?   Some ideas:
#  - global
#  - attached to closures (including _default_closure)
#  - attached to Symbols or Expr
#

_cpu_context = cl.Context(dev_type=cl.device_type.CPU)
_cpu_queue   = cl.CommandQueue(_cpu_context)


def register_transform(priority, tags=None):
    if tags is None:
        tags = set()
    tags.add('lnumpy_opencl')
    tags.add('default')
    return transform.register_transform(priority,tags)

class UnaryElemwiseCpu(Impl):
    def __init__(self, fallback_f, code_fragment, name):
        self.fallback_f = fallback_f
        self.code_fragment = code_fragment
        self.name=name
        self.cache = {}

    def __str__(self):
        return 'OpenCL_CPU_%s'%self.name

    def _spec_from_arg(self, arg):
        #TODO: return argument memory alignment
        return arg.dtype, arg.shape, arg.strides

    def build_fn(self, specs):
        spec, = specs
        dtype,dims,strides = spec
        try:
            ctype = {
                    numpy.dtype('float32'):'float', 
                    numpy.dtype('float64'):'double',
                    }[dtype]
            if len(dims) != 1:
                raise NotImplementedError()
            if strides[0] != {
                    numpy.dtype('float32'):4,
                    numpy.dtype('float64'):8}[dtype]:
                raise NotImplementedError()
        except KeyError:
            raise NotImplementedError(specs)
        print 'BUILDING FN'
        fragment = self.code_fragment(inputs=['a[i]'], outputs=['z[i]'])
        
        prg = cl.Program(_cpu_context, """
            __kernel void elemwise(const int N, 
            __global const %(ctype)s *a,
            __global %(ctype)s *z)
            {
              int gid = get_global_id(0);
              for (int i = N*gid; i < N*gid+N; ++i) %(fragment)s;
            }
            """ % locals()).build()

        def rval(a):
            nthreads=1 #TODO: heuristic to choose how many threads
            z = numpy.zeros_like(a) #empty_like is faster, but can hide errors
            a_buf = cl.Buffer(_cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=a)
            z_buf = cl.Buffer(_cpu_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=z)
            rval = prg.elemwise(_cpu_queue, (nthreads,), None, 
                    numpy.int64(len(a)/nthreads), a_buf, z_buf)
            rval.wait()
            return z
        return rval

    def fn(self, *args):
        specs = tuple(self._spec_from_arg(a) for a in args)
        try:
            f = self.cache[specs]
        except KeyError:
            try:
                f = self.cache[specs] = self.build_fn(specs)
            except NotImplementedError:
                f = self.cache[specs] = self.fallback_f
        return f(*args)

def impl_from_fnname(fnname):
    return UnaryElemwiseCpu(getattr(numpy,fnname),
            lambda inputs, outputs: '%s = %s(%s);'%(outputs[0],fnname,inputs[0]),
            fnname)


@register_transform(priority=1 )
def replace_numpy_with_opencl(expr_graph):
    for expr in expr_graph.expr_iter():
        if isinstance(expr.impl, lnumpy.NumpyElemwise)\
                and expr.n_inputs==1:
            try:
                cl_impl = impl_from_fnname(expr.impl.name)
            except NotImplementedError:
                continue
            expr_graph.replace_impl(expr, cl_impl)


if 0:
    class PowCL(theano.gof.Op):
        def __eq__(self, other):
            return type(self) == type(other)
        def __hash__(self):
            return hash(type(self))
        def make_node(self, a, b):
            return theano.Apply(self,[a,b], [a.type()])

        def make_thunk(self, node,
                input_computed, output_computed,
                input_registers, output_registers):

            def rval():
                a,b = input_registers
                output_registers[0] = numpy.pow(a,b)

            ctx,queue = node.env.opencl_feature.context_queue("CPU")

            prg = cl.Program(ctx, """
                __kernel void sum(const int N, __global const float4 *a,
                __global const float4 *b, __global float4 *c)
                {
                  int gid = get_global_id(0);
                  for (int i = N*gid; i < N*gid+N; ++i) c[i] = pow(a[i] , b[i]);
                }
                """).build()
            mf = cl.mem_flags
            def rval():
                #print 'running OpenCL version'
                a = input_registers[0]
                b = input_registers[1]
                #output_registers[0] = a+b
                #return
                output_registers[0] = z = numpy.zeros_like(a)
                a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a)
                b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b)
                z_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=z)
                rval = prg.sum(queue, (2,), None, numpy.int64(len(a)/2/4), 
                        a_buf, b_buf, z_buf)
                #print rval
                rval.wait()
                #cl.enqueue_read_buffer(queue, z_buf, z).wait()

            rval.lazy=False
            return rval
    pow_cl = PowCL()

    swap_impls = True

    @theano.tensor.opt.register_specialize
    @theano.gof.local_optimizer([])
    def add_to_addcl(node):
        if swap_impls:
            if node.op == T.pow:
                return [pow_cl(*node.inputs)]

