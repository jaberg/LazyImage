
# Symbol can have multiple exprs during optimization
# Types would ideally be handled as sets

class Symbol(object):
    """A data node in an expression graph.
    """
    def __init__(self, expr=None, type=None, name=None):
        self.expr = expr
        self.type = type
        self.name = name

class Expr(object):
    """An implementation node in a expression graph.  """
    def __init__(self, impl, inputs, outputs):
        self.impl = impl
        self.inputs = inputs
        self.outputs = outputs

class Closure(dict):
    def symbol(self, value, ctor=Symbol, *args, **kwargs):
        rval = ctor(*args,**kwargs)
        self[rval] = value
        return rval

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

    def __init__(self, fn, n_outputs=1, name=None):
        self.n_outputs=n_outputs
        self.fn = fn
        self.name = name

    def Expr(self, args):
        return Expr(self, args, None)

    def _args_has_symbol(self, args):
        for a in args:
            if isinstance(a, Symbol):
                return True
        return False

    def __call__(self, *args):
        if self._args_has_symbol(args):
            expr = self.Expr(args)
            if self.n_outputs>1:
                return [Symbol(expr=expr) for o in self.outputs]
            else:
                return Symbol(expr=expr)
        else:
            return self.fn(*args)

class ExprGraph(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

class TransformPolicy(object):
    def __call__(self, expr_graph):
        pass

class VirtualMachine(object):
    def __init__(self, inputs, outputs, closure, expr_graph, updates, return_outputs0):
        self.inputs = inputs
        self.outputs = outputs
        self.closure = closure
        self.updates = updates
        self.return_outputs0 = return_outputs0
        self.expr_graph = expr_graph

    def __call__(self, *args):
        results = dict(self.closure)
        for a, s in zip(args, self.inputs):
            results[s] = a
        def rec_eval(s):
            try:
                return results[s]
            except KeyError:
                pass
            if isinstance(s, Symbol) and s.expr:
                #TODO: support multi-output Impls
                vargs = [rec_eval(i) for i in s.expr.inputs]
                rval = results[s] = s.expr.impl(*vargs)
                return rval
            return s
        for s in self.outputs:
            rec_eval(s)
        #TODO: support multiple outputs
        if self.return_outputs0:
            return [results[o] for o in self.outputs]
        else:
            return results[self.outputs[0]]

def function_driver(inputs, outputs, closure, givens, updates, 
        VM, TP, return_outputs0):

    if givens:
        #TODO: use the givens to modify the clone operation
        raise NotImplementedError('givens arg is not implemented yet')

    if updates:
        #TODO: translate the updates into the cloned graph
        raise NotImplementedError('updates arg is not implemented yet')

    arg_symbols = [Symbol() for i in inputs]
    lookup_tbl = dict(zip(inputs, arg_symbols))
    output_symbols = [clone(o, lookup_tbl) for o in outputs]

    expr_graph = ExprGraph(arg_symbols, output_symbols)
    TP(expr_graph)
    return VM(inputs, outputs, closure, expr_graph, updates, return_outputs0)

def function(inputs, outputs, closure=None, givens=None, updates=None,
        VM=None, TP=None):
    if closure is None:
        closure = Closure()
    if VM is None:
        VM = VirtualMachine
    if TP is None:
        TP = lambda expr_graph: expr_graph

    if isinstance(outputs, Symbol):
        outputs = [outputs]
        return_outputs0 = True
    else:
        return_outputs0 = False

    return function_driver(inputs, outputs, closure, givens, updates, VM, TP,
            return_outputs0=return_outputs0)

def compute(outputs, closure=None, givens=None, VM=None, TP=None):
    return function(
            inputs=[], 
            outputs=outputs,
            closure=closure,
            updates=None,
            givens=None,
            VM=VM,
            TP=TP)()

def clone(s, dct):
    """Copy an expression graph"""
    if s in dct:
        return dct[s]
    if getattr(s, 'expr', None):
        input_copies=[clone(i_s, dct) for i_s in s.expr.inputs]
        rval = s.expr.impl(*input_copies)
        dct[s] = rval
        return rval
    return s



