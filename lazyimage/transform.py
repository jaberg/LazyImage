
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
