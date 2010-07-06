from .structures import Closure, Symbol, Expr, default_closure
from .exprgraph import clone, ExprGraph
from .transform import TransformPolicy

class MissingValue(Exception):pass

class VirtualMachine(object):
    def __init__(self, inputs, outputs, closure, expr_graph, updates, return_outputs0):
        self.inputs = inputs
        self.outputs = outputs
        self.closure = closure
        self.updates = updates
        self.return_outputs0 = return_outputs0
        self.expr_graph = expr_graph

    def printprog(self):
        for i,impl in enumerate(self.expr_graph.expr_iter()):
            print i,impl

    def __call__(self, *args):
        results = dict(self.closure.values)
        for a, s in zip(args, self.inputs):
            results[s] = a
        def rec_eval(s):
            try:
                return results[s]
            except KeyError:
                pass
            if not s.expr:
                raise MissingValue(s)
            #TODO: support multi-output Impls
            vargs = [rec_eval(i) for i in s.expr.inputs]
            rval = results[s] = s.expr.impl(*vargs)
            return rval
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

    arg_symbols = [i.clone() for i in inputs]

    #lookup_tbl maps the user's  symbols to the cloned symbols
    # used in the VirtualMachine
    # some of the cloned symbols may be optimized away, 
    # and not be present in the VirtualMachine's program.
    lookup_tbl = dict(zip(inputs, arg_symbols))
    output_symbols = [clone(o, lookup_tbl) for o in outputs]

    #
    # Bootstrap the meta-data of the cloned inputs according to the restrictions
    # that have been set in the closure.
    for orig, cloned in lookup_tbl.iteritems():
        closure.update_metadata(orig, cloned.meta)

    expr_graph = ExprGraph(arg_symbols, output_symbols)
    TP(expr_graph)
    return VM(inputs, outputs, closure, expr_graph, updates, return_outputs0)

def function(inputs, outputs, closure=default_closure, givens=None, updates=None,
        VM=None, TP=None):
    if VM is None:
        VM = VirtualMachine
    if TP is None:
        TP = TransformPolicy.new()

    if isinstance(outputs, Symbol):
        outputs = [outputs]
        return_outputs0 = True
    else:
        return_outputs0 = False

    return function_driver(inputs, outputs, closure, givens, updates, VM, TP,
            return_outputs0=return_outputs0)

def symbol(*args, **kwargs):
    return default_closure.symbol(*args, **kwargs)
def set_value(s, v, closure=default_closure):
    closure.set_value(s, v)
def get_value(s, closure=default_closure):
    closure.get_value(s)

def compute(outputs, closure=default_closure, givens=None, VM=None, TP=None):
    return function(
            inputs=[], 
            outputs=outputs,
            closure=closure,
            updates=None,
            givens=None,
            VM=VM,
            TP=TP)()

