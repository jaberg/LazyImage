"""
Data structures to support lazy-evaluation decorators.

"""
import copy

import exprgraph
import transform

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

    def clone(self, new_closure, as_constant=False ):
        if as_constant:
            newvalue = self._value # TODO: Copy here?
            type = self.type.as_constant(self._value)
        else:
            newvalue = self.type.get_clear_value()
            type = copy.deepcopy(self.type)
        return self.__class__(
                closure=new_closure,
                expr=None, 
                type=type,
                value=newvalue,
                name=self.name)

    def compute(self):
        return self.closure.compute_value(self)
exprgraph.Symbol=Symbol
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
                rval = orig_symbol.expr.impl(*input_clones)
                replacement_table[orig_symbol] = rval
            else:
                rval = orig_symbol.expr.impl(*input_copies)
                for s,r in zip(orig_symbol.expr.outputs, rval):
                    replacement_table[s] = r
            return replacement_table[orig_symbol]
        else:
            # this is a leaf node, so we make it a constant
            new_symbol = self.add_symbol(orig_symbol.clone(
                new_closure = self,
                as_constant=True))
        return new_symbol

    def add_symbol(self, symbol):
        """Add a symbol to this closure
        """
        if not isinstance(symbol, Symbol):
            raise TypeError(symbol)
        if symbol.closure and symbol.closure is not self:
            raise ValueError('symbol already has closure', symbol.closure)
        self.elements.add(symbol)
        symbol.closure = self
        symbol._computed = False
        return symbol
    def add_obj(self, obj, constant=True,name=None):
        symbol = self.add_symbol(self.allocation_policy.as_symbol(self, obj, constant,name))
        symbol._computed = True
        return symbol

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
        rval =self.get_value(symbol)
        return rval

    def compute_values(self, symbols):
        return [self.compute_value(s) for s in symbols]

    def clear_values(self):
        """ Clear the values of all non-constant Symbols in the Closure """
        for symbol in self.elements:
            self.clear_value(symbol)

    def change_input(self, expr, position, new_symbol):
        #PRE-HOOK
        raise NotImplementedError()
        #POST-HOOK
    def replace_impl(self, expr, new_impl):
        #PRE-HOOK
        expr.impl = new_impl
        #POST-HOOK

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

class CallableClosure(Closure):
    def __init__(self, allocation_policy, transform_policy):
        super(CallableClosure, self).__init__(allocation_policy)
        self._iterating = False
        self._modified_since_iterating = False
        self.transform_policy = transform_policy

        #TODO: install a change_input post-hook to set modified_since_iterating to True
        # call pre-hooks
        if self._iterating:
            self._modified_since_iterating = True
        # call post-hooks

    def set_io(self, inputs, outputs, updates, unpack_single_output):
        if updates:
            #TODO: translate the updates into the cloned graph
            raise NotImplementedError('updates arg is not implemented yet')
        for o in inputs+outputs:
            if o.closure is not self:
                raise ValueError('output not in closure', o)
        self.inputs = inputs
        self.outputs = outputs
        self.unpack = unpack_single_output and len(outputs)==1
        self.transform_policy(self)

    def printprog(self):
        for i,impl in enumerate(self.expr_iter()):
            print i,impl

    def expr_iter(self):
        """Yield expr nodes in arbitrary order.

        Raises an exception if you try to continue iterating after
        modifying the expression graph.
        """
        exprs = [e for e in exprgraph.io_toposort(self.inputs, self.outputs) if isinstance(e,Expr)]
        self._iterating = True
        for e in exprs:
            if self._modified_since_iterating:
                raise Exception('Modified since iterating')
            yield e
        self._iterating = False
        self._modified_since_iterating = False
    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise TypeError('Wrong number of inputs')
        for i, a in zip(self.inputs, args):
            self.set_value(i,a)
        if self.unpack:
            return self.compute_value(self.outputs[0])
        else:
            return self.compute_values(self.outputs)

# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.
def default_closure_ctor():
    return CallableClosure(AllocationPolicy(), transform.TransformPolicy.new())
default_closure = default_closure_ctor()

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
exprgraph.Expr=Expr

def function(inputs, outputs, closure_ctor=default_closure_ctor,
        givens=None, updates=None):
    if isinstance(outputs, Symbol):
        outputs = [outputs]
        return_outputs0 = True
    else:
        return_outputs0 = False

    if givens:
        #TODO: use the givens to modify the clone operation
        raise NotImplementedError('givens arg is not implemented yet')
    if updates:
        #TODO: clone the updates
        raise NotImplementedError('updates arg is not implemented yet')

    closure = closure_ctor()

    cloned_inputs = [closure.add_symbol(i.clone(closure,as_constant=False)) for i in inputs]
    replacements = dict(zip(inputs, cloned_inputs))
    cloned_outputs = [closure.clone(o, replacements) for o in outputs]
    closure.set_io(cloned_inputs, cloned_outputs, updates, return_outputs0)
    return closure

