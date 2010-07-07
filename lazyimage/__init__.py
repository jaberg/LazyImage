import structures, transform, engine

from structures import Type, Symbol, Closure, Impl, Expr, function

from transform import register_transform, transform_db

def set_default_closure(dc):
    structures.default_closure = dc
def get_default_closure():
    return structures.default_closure

