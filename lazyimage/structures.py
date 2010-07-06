"""
Data structures to support lazy-evaluation decorators.

"""
import copy

class Metadata(object):
    """
    Properties of data.
    A Metadata instances can be attached to a Symbol, and Impls can modify the metadata of
    their outputs to communicate with other Impls and graph-transformation code.

    """
    def __init__(self):
        changed = False
    def is_conformant(self, obj):
        """
        Return True iff object is consistent with this Metadata.

        This function can be used in self-verifying execution modes,
        to ensure that Impls return the sort of objects they promised to return.
        """
        return True
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
        return self.__class__(
                expr=None, 
                meta=copy.deepcopy(self.meta),
                name=self.name)

class DataInterface(object):
    def __init__(self, constant):
        self.constant = constant
    def can_handle(self, value):
        return True
    def update_metadata(self, value, closure_attribs, meta, ):
        """
        Add / modify `meta` to describe `value` given `closure_attribs`.
        """
        return

    def make_conformant(self, obj):
        """
        Return an object that is equal to obj, but conformant to this metadata.

        This may involve casting integers to floating-point numbers, 
        padding with broadcastable dimensions, changing string encodings, etc.
        """
        return obj

    def cmp(self, v0, v1):
        return cmp(v0,v1)

    def approx_cmp(self, v0, v1):
        return cmp(v0,v1)

data_interfaces = [DataInterface()]

def guess_data_interface(value):
    return [di for di in data_interfaces if di.can_handle(value)][-1]

class ClosureElement(object):

    # Does not contain a backpointer to the symbol
    # because many symbols could be associated to one ClosureElement
    # This permits the Closure to merge constants.
    def __init__(self, value, data_interface, constant):
        self.constant = constant
        self.value = value
        self.data_interface = data_interface

    def get(self):
        return self.value
    def set(self, v):
        if (not self.constant) or (v is self.value):
            self.value = v
        self.value = self.data_interface.make_conformant(v) 
    def update_metadata(self, meta):
        self.data_interface.update_metadata(meta, self.value, self.constant)


class Closure(object):
    """A closure encapsulates data that is available for lazy evaluation

    A Closure subclass implements the policy for where data should be stored,
    and whether it should be copied to a private area.

    """
    # This is like the shared constructor
    def __init__(self, automerge=True):
        self.elements = {}

    def new_elem(self, symbol, value, data_interface, constant,):
        rval = ClosureElement(value, data_interfaces,constant)
        self.elements[s] = rval
        return rval

    def new_symbol(self, value, ctor=Symbol.blank, data_interface=None, constant=False,
            allow_merge=False):
        """
        Returns a new symbol created with `ctor`.
        Sets a value for it in this closure before returning.
        """
        if data_interfaces is None:
            data_interface = guess_data_interface(value)
        return self.new_elem(ctor(), value, data_interfaces, constant=constant)

    def get_value(self, s):
        return self.elements[s].get()
    def get_element(self, s):
        return self.elements[s]
    def set_value(self, s, v):
        return self.elements[s].set(v)

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

    def __str__(self):
        return 'Impl_%s'%self.name

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

    def infer_metadata(self, expr):
        """
        Update the meta-data of inputs and outputs.

        Explicitly mark .meta attributes as being modified by setting
        <symbol>.meta.changed = True
        """

    def as_input(self, obj):
        """Convenience method - it's the default constructor for lazy Impl __call__ methods to
        use to easily turn all inputs into symbols.
        """
        if isinstance(obj, Symbol):
            return obj
        return self.closure.new_symbol(obj, constant=True, allow_merge=True)


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

