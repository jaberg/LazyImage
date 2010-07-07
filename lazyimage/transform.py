
class TransformHandle(object):
    def __init__(self, name, position, tags, transform_fn):
        self.name = name
        self.position=position
        self.tags=tags
        self.transform_fn = transform_fn
        self.enabled = True
    def __str__(self):
        return self.name
    def __repr__(self):
        return 'TransformHandle{%s}'%self.name
    def transform(self, expr_graph):
        if self.enabled:
            return self.transform_fn(expr_graph)

transform_db = set()

def register_transform(position, tags=[]):
    def deco(f):
        handle = TransformHandle(f.__name__,position,tags,f)
        transform_db.add(handle)
        return handle
    return deco

class TransformPolicy(object):
    @classmethod
    def new(cls, filter=lambda handle:True):
        handles = [(h.position,h) for h in transform_db if filter(h)]
        handles.sort()
        return cls([h[1] for h in handles])

    def __init__(self, handles):
        self.handles = handles

    def __call__(self, expr_graph):
        for h in self.handles:
            h.transform(expr_graph)

if 0:
    #from .exprgraph import clone, ExprGraph
    #from .transform import TransformPolicy

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
                rval = results[s] = s.expr.impl.fn(*vargs)
                return rval
            for s in self.outputs:
                rec_eval(s)
            #TODO: support multiple outputs
            if self.return_outputs0:
                return [results[o] for o in self.outputs]
            else:
                return results[self.outputs[0]]
