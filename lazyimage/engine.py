
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


# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.

_default_closure = Closure()
def symbol(*args, **kwargs):
    return _default_closure.symbol(*args, **kwargs)

def compute(outputs, closure=_default_closure, givens=None, VM=None, TP=None):
    return function(
            inputs=[], 
            outputs=outputs,
            closure=closure,
            updates=None,
            givens=None,
            VM=VM,
            TP=TP)()

