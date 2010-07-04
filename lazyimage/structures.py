"""
Data structures to support lazy-evaluation decorators.

"""

class Metadata(object):
    """
    Properties of data.
    A Metadata instances can be attached to a Symbol, and Impls can modify the metadata of
    their outputs to communicate with other Impls and graph-transformation code.

    """
    def __init__(self):
        pass
    def is_conformant(self, obj):
        """
        Return True iff object is consistent with this Metadata.

        This function can be used in self-verifying execution modes,
        to ensure that Impls return the sort of objects they promised to return.
        """
        return True
    def make_conformant(self, obj):
        """
        Return an object that is equal to obj, but conformant to this metadata.

        This may involve casting integers to floating-point numbers, 
        padding with broadcastable dimensions, changing string encodings, etc.
        """
        return obj
    def disjoint_from(self, other):
        """
        Return True iff no data could be conformant to self and other.

        This function is useful for detecting errors in
        """
        return False

class Symbol(object):
    """A data node in an expression graph.
    """
    Metadata = Metadata
    @classmethod
    def blank(cls, name=None):
        return cls(expr=None, meta=cls.Metadata(), name=name)
    def __init__(self, expr, meta, name):
        self.expr = expr
        self.meta = meta
        self.name = name

    def clone(self):
        return self.__class__(expr=None, meta=self.meta, name=self.name)

class Closure(object):
    """A closure encapsulates data that is available for lazy evaluation

    A Closure subclass implements the policy for where data should be stored,
    and whether it should be copied to a private area.
    """
    # This is like the shared constructor
    def __init__(self):
        self.store = {}
    def symbol(self, value, ctor=Symbol.blank, *args, **kwargs):
        rval = ctor(*args,**kwargs)
        self.store[rval] = value
        return rval

    def constant(self, value):
        #TODO: Mark this symbol's meta-data as being constant
        return self.symbol(value)

    def as_input(self, obj):
        if isinstance(obj, Symbol):
            return obj
        return self.constant(obj)

    def get_value(self, s):
        return self.store[s]
    def init_value(self, s, v):
        self.store[s] = v
    def set_value(self, s, v):
        if s not in self.store:
            raise KeyError(s)
        self.store[s] = s.meta.make_conformant(v)

# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.
default_closure = Closure()

class Impl(object):
    """

    Attributes:
      fn - a normal [non-symbolic] function that does the computations
    """
    @staticmethod
    def new(*args, **kwargs):
        def deco(fn):
            return Impl(fn=fn,*args, **kwargs)
        return deco

    def __init__(self, fn, n_outputs=1, name=None, closure=default_closure):
        self.n_outputs=n_outputs
        self.fn = fn
        self.name = name
        self.closure = closure

    def Expr(self, args, outputs):
        rval = Expr(self, args, outputs)
        for o in rval.outputs:
            o.expr = rval
        return rval

    def _args_has_symbol(self, args):
        for a in args:
            if isinstance(a, Symbol):
                return True
        return False

    def __call__(self, *args):
        if self._args_has_symbol(args):
            inputs = [self.closure.as_input(a) for a in args]
            outputs = [Symbol.blank() for o in range(self.n_outputs)]
            expr = self.Expr(inputs, outputs)
            if self.n_outputs>1:
                return outputs
            else:
                return outputs[0]
        else:
            return self.fn(*args)

class Expr(object):
    """An implementation node in a expression graph.  """
    def __init__(self, impl, inputs, outputs):
        self.impl = impl
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        #assert all Inputs and outputs are symbols
        bad_inputs = [i for i in inputs if not isinstance(i, Symbol)]
        bad_outputs =[i for i in outputs if not isinstance(i, Symbol)] 
        assert not bad_inputs, bad_inputs
        assert not bad_outputs, bad_outputs
    def __str__(self):
        return 'Expr{%s}'%str(self.impl)
    n_inputs = property(lambda self: len(self.inputs))
    n_outputs = property(lambda self: len(self.outputs))

