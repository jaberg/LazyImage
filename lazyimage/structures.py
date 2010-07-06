"""
Data structures to support lazy-evaluation decorators.

"""
import copy

class Type(object):
    """
    Class to represent a set of possible data values.

    A Type is attached to a Symbol.
    Properties and attributes of a Type instance parametrize the kinds of data that a Symbol
    might represent.

    """
    def __init__(self):
        pass

    def is_conformant(self, obj):
        """
        Return True iff object is consistent with this Type.

        This function can be used in self-verifying execution modes,
        to ensure that Impls return the sort of objects they promised to return.
        """
        return True

    def make_conformant(self, obj):
        """
        Return an object that is equal to obj, but conformant to this metadata.

        This may involve casting integers to floating-point numbers, 
        padding with broadcastable dimensions, changing string encodings, etc.

        This method raises TypeError if `obj` cannot be made conformant.
        """
        return obj

    def eq(self, v0, v1, approx=False):
        """Return True iff v0 and v1 are [`approx`] equal in the context of this Type. 

        The return value when neither v0 nor v1 is conformant is undefined.
        
        """
        return v0 == v1

    def disjoint_from(self, othertype):
        """
        Return True iff no data could be conformant to self and other.

        This function is useful for detecting errors in graph transformations.
        """
        return False

    def get_clear_value(self):
        return None

    def as_constant(self, value):
        return Constant(value)


class Constant(Type):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.constant = True
        self.value = value
        # Subclasses can set meta-data at this point

    def is_conformant(self, obj):
        return obj is self.value
    def make_conformant(self, obj):
        if obj is self.value:
            return obj
        raise TypeError(obj)
    def eq(self, v0, v1, approx=False):
        return v0 is v1
    def disjoint_from(self, othertype):
        return not ( getattr(othertype, 'constant', False)\
                        and othertype.value is self.value)
    def get_clear_value(self):
        return self.value

class Symbol(object):
    """A value node in an expression graph.
    """
    Type=Type
    @classmethod
    def new(cls, closure, expr=None, type=None,value=None,name=None):
        if type is None:
            type = cls.Type()
        rval = cls(None, expr,type,value,name)
        return closure.add_symbol(rval)
    def __init__(self, closure, expr, type, value, name):
        self.closure = closure
        self.expr = expr
        self.type = type
        self._value = value
        self.name = name

        self._value_version = 0 # changing self._value increments this

    def clone(self, as_constant=False ):
        newvalue = self.type.get_clear_value()
        if as_constant:
            type = self.type.as_constant()
        else:
            type = copy.deepcopy(self.type)
        return self.__class__(
                expr=None, 
                type=type,
                value=newvalue,
                name=self.name)

    def compute(self):
        return self.closure.compute_value(self)

class Closure(object):
    """A closure encapsulates a set of symbols that can be connected by Expressions and lazy evaluation

    A Closure subclass implements the policy for where data should be stored,
    and whether it should be copied to a private area.

    """
    def __init__(self, allocation_policy):
        self.elements = set()
        self.allocation_policy = allocation_policy
        self.computed = {}

    def clone(self, orig_symbol, replacement_table):
        """Return the clone of orig_symbol

        replacement_table is modified by side-effect

        replacement_table[orig_symbol] -> new_symbol
        """
        if orig_symbol in replacement_table:
            return replacement_table[orig_symbol]
        if getattr(orig_symbol, 'expr', None):
            input_clones=[self.clone(i_s, replacement_table) 
                    for i_s in orig_symbol.expr.inputs]
            if orig_symbol.expr.n_outputs == 1:
                rval = orig_symbol.expr.impl(*input_copies)
                replacement_table[orig_symbol] = rval
            else:
                rval = orig_symbol.expr.impl(*input_copies)
                for s,r in zip(orig_symbol.expr.outputs, rval):
                    replacement_table[s] = r
            return replacement_table[orig_symbol]
        else:
            # this is a leaf node, so we make it a constant
            new_symbol = self.add_obj(orig_symbol.clone(as_constant=True))
            print 'CLONED VALUE', new_symbol._value
        return new_symbol

    def add_symbol(self, symbol):
        """Add a symbol to this closure
        """
        if not isinstance(symbol, Symbol):
            raise TypeError(symbol)
        if symbol.closure is self:
            assert symbol in self.elements
            return symbol
        if symbol.closure:
            raise ValueError('symbol already has closure', symbol.closure)
        symbol.closure = self
        self.elements.add(symbol)
        symbol._computed = False
        return symbol
    def add_obj(self, obj, constant=True,name=None):
        symbol = self.allocation_policy.as_symbol(self, obj, constant,name)
        return self.add_symbol(symbol)

    def get_value(self, symbol):
        if symbol.closure is not self:
            raise ValueError('symbol not in closure', symbol)
        assert symbol.type.is_conformant(symbol._value)
        return symbol._value
    def set_value(self, symbol, v):
        if symbol.closure is not self:
            raise ValueError('symbol not in closure', symbol)
        symbol._value = symbol.type.make_conformant(v) 
    def clear_value(self, symbol):
        if symbol.closure is not self:
            raise ValueError('symbol not in closure', symbol)
        symbol._value = symbol.type.get_clear_value()

    def compute_value(self, symbol):
        if symbol not in self.elements:
            raise ValueError('Symbol not in this closure')
        if symbol.expr is None:
            symbol._computed = True
        if symbol._computed:
            return self.get_value(symbol)
        expr = symbol.expr
        args = [self.compute_value(i) for i in expr.inputs]
        results = expr.impl.fn(*args)
        if expr.n_outputs>1:
            for s,r in zip(expr.outputs, results):
                self.set_value(s,r)
        else:
            # symbol must be the only output of expr
            self.set_value(symbol, results)
        return self.get_value(symbol)

    def clear_values(self):
        """ Clear the values of all non-constant Symbols in the Closure """
        for symbol in self.elements:
            self.clear_value(symbol)

class AllocationPolicy(object):
    def as_symbol(self, closure, obj, constant=True, name=None):
        return Symbol(closure=closure,
                expr=None, 
                type=Constant(obj) if constant else Type(),
                value=obj, 
                name=name)

    def guess_type(obj):
        """Return the Type [subclass] instance most suitable for `obj`
        """
        return [di for di in data_interfaces if di.can_handle(obj)][-1]


# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.
default_closure = Closure(AllocationPolicy())

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

    @staticmethod
    def closure_from_args(args):
        for a in args:
            if isinstance(a, Symbol):
                return a.closure

    def __call__(self, *args):
        closure = self.closure_from_args(args)

        if closure:
            inputs = [self.as_input(a) for a in args]
            outputs = [Symbol.new(closure) for o in range(self.n_outputs)]
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
        return self.closure.add_obj(obj, constant=True)


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

