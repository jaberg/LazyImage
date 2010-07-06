"""
Sorting and traversing of expression graphs.
"""
from collections import deque

from .structures import Expr, Symbol, Impl

# Symbol can have multiple exprs during optimization
class ExprGraph(object):
    """
    Object to permit event-driven programming during expression graph transformation.
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self._iterating = False
        self._modified_since_iterating = False

        all_symbols = io_toposort(inputs, outputs)

    def expr_iter(self):
        """Yield expr nodes in arbitrary order.

        Raises an exception if you try to continue iterating after
        modifying the expression graph.
        """
        exprs = [e for e in io_toposort(self.inputs, self.outputs) if isinstance(e,Expr)]
        self._iterating = True
        for e in exprs:
            if self._modified_since_iterating:
                raise Exception('Modified since iterating')
            yield e
        self._iterating = False
        self._modified_since_iterating = False

    def replace_symbol(self, current_symbol, new_symbol):
        # call pre-hooks
        raise NotImplementedError()
        if self._iterating:
            self._modified_since_iterating = True
        # call post-hooks

    def replace_impl(self, symbol, new_impl):
        # call pre-hooks
        symbol.impl = new_impl
        # call post-hooks


##
#
# TODO: Review the following code
#
# TODO: optionally use NetworkX for these algorithms
#
##

def stack_search(start, expand, mode='bfs', build_inv = False):
    """Search through a graph, either breadth- or depth-first

    :type start: deque
    :param start: search from these nodes
    :type expand: callable
    :param expand: 
        when we get to a node, add expand(node) to the list of nodes to visit.  This function
        should return a list, or None
    :rtype: list of `Symbol` or `Expr` instances (depends on `expend`)
    :return: the list of nodes in order of traversal.
    
    :note:
        a node will appear at most once in the return value, even if it appears multiple times
        in the start parameter.  

    :postcondition: every element of start is transferred to the returned list.
    :postcondition: start is empty.

    """

    if mode not in ('bfs', 'dfs'):
        raise ValueError('mode should be bfs or dfs', mode)
    rval_set = set()
    rval_list = list()
    if mode is 'bfs': start_pop = start.popleft
    else: start_pop = start.pop
    expand_inv = {}
    while start:
        l = start_pop()
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            expand_l = expand(l)
            if expand_l:
                if build_inv:
                    for r in expand_l:
                        expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
    assert len(rval_list) == len(rval_set)
    if build_inv:
        return rval_list, expand_inv
    return rval_list

def inputs(variable_list, blockers = None):
    """Return the inputs required to compute the given Symbol.

    :type variable_list: list of `Symbol` instances
    :param variable_list:
        output `Symbol` instances from which to search backward through expr
    :rtype: list of `Symbol` instances
    :returns: 
        input nodes with no expr, in the order found by a left-recursive depth-first search
        started at the nodes in `variable_list`.

    """
    def expand(r):
        if r.expr and (not blockers or r not in blockers):
            l = list(r.expr.inputs)
            l.reverse()
            return l
    dfs_variables = stack_search(deque(variable_list), expand, 'dfs')
    rval = [r for r in dfs_variables if r.expr is None]
    #print rval, _orig_inputs(o)
    return rval

def io_variables_and_orphans(i, o):
    """WRITEME
    """
    def expand(r):
        if r.expr and r not in i:
            l = list(r.expr.inputs) + list(r.expr.outputs)
            l.reverse()
            return l
    variables = stack_search(deque(o), expand, 'dfs')
    orphans = [r for r in variables if r.expr is None and r not in i]
    return variables, orphans

def io_expr_list(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Symbol}s
    :type o: list
    :param o: output L{Symbol}s

    :returns:
        the set of ops that are contained within the subgraph that lies between i and o,
        including the expr of the L{Symbol}s in o and intermediary ops between i and o, but
        not the expr of the L{Symbol}s in i.
    """
    rval = []
    variables, orphans = io_variables_and_orphans(i, o)
    for r in variables:
        if r not in i and r not in orphans:
            if r.expr is not None:
                rval.append(r.expr)
    return rval

def variables(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Symbol}s
    :type o: list
    :param o: output L{Symbol}s

    :returns:
        the set of Symbol that are involved in the subgraph that lies between i and o. This
        includes i, o, orphans(i, o) and all values of all intermediary steps from i to o.
    """
    return variables_and_orphans(i, o)[0]

def orphans(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Symbol}s
    :type o: list
    :param o: output L{Symbol}s

    :returns:
        the set of Symbol which one or more Symbol in o depend on but are neither in i nor in
        the subgraph that lies between i and o.

    e.g. orphans([x], [(x+y).out]) => [y]
    """
    return variables_and_orphans(i, o)[1]

def old_clone(i, o, copy_inputs = True):
    """ WRITEME

    :type i: list
    :param i: input L{Symbol}s
    :type o: list
    :param o: output L{Symbol}s
    :type copy_inputs: bool
    :param copy_inputs: if True, the inputs will be copied (defaults to False)

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o).
    """
    equiv = clone_get_equiv(i, o, copy_inputs)
    return [equiv[input] for input in i], [equiv[output] for output in o]

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



def clone_get_equiv(i, o, copy_inputs_and_orphans = True):
    """ WRITEME

    :type i: list
    :param i: input L{Symbol}s
    :type o: list
    :param o: output L{Symbol}s
    :type copy_inputs_and_orphans: bool
    :param copy_inputs_and_orphans: 
        if True, the inputs and the orphans will be replaced in the cloned graph by copies
        available in the equiv dictionary returned by the function (copy_inputs_and_orphans
        defaults to True)

    :rtype: a dictionary
    :return:
        equiv mapping each L{Symbol} and L{Op} in the graph delimited by i and o to a copy
        (akin to deepcopy's memo).
    """

    d = {}
    for input in i:
        if copy_inputs_and_orphans:
            cpy = input.clone()
            cpy.expr = None
            cpy.index = None
            d[input] = cpy
        else:
            d[input] = input

    for apply in io_toposort(i, o):
        for input in apply.inputs:
            if input not in d:
                if copy_inputs_and_orphans:
                    cpy = input.clone()
                    d[input] = cpy
                else:
                    d[input] = input

        new_apply = apply.clone_with_new_inputs([d[i] for i in apply.inputs])
        d[apply] = new_apply
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            d[output] = new_output

    for output in o:
        if output not in d:
            d[output] = output.clone()

    return d

def general_toposort(r_out, deps, debug_print = False):
    """WRITEME

    :note: 
        deps(i) should behave like a pure function (no funny business with internal state)

    :note: 
        deps(i) will be cached by this function (to be fast)

    :note:
        The order of the return value list is determined by the order of nodes returned by the deps() function.
    """
    deps_cache = {}
    def _deps(io):
        if io not in deps_cache:
            d = deps(io)
            if d:
                deps_cache[io] = list(d)
            else:
                deps_cache[io] = d
            return d
        else:
            return deps_cache[io]

    assert isinstance(r_out, (tuple, list, deque))

    reachable, clients = stack_search( deque(r_out), _deps, 'dfs', True)
    sources = deque([r for r in reachable if not deps_cache.get(r, None)])

    rset = set()
    rlist = []
    while sources:
        node = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in clients.get(node, []):
                deps_cache[client] = [a for a in deps_cache[client] if a is not node]
                if not deps_cache[client]:
                    sources.append(client)

    if len(rlist) != len(reachable):
        if debug_print:
            print ''
            print reachable
            print rlist
        raise ValueError('graph contains cycles')

    return rlist

def io_toposort(i, o, orderings = {}):
    """Return a topological ordering of Symbols and Expressions in a graph
    """
    #the inputs are used only here in the function that decides what 'predecessors' to explore
    iset = set(i)
    def deps(obj): 
        rval = []
        if obj not in iset:
            if isinstance(obj, Symbol): 
                if obj.expr:
                    rval = [obj.expr]
            if isinstance(obj, Expr):
                rval = list(obj.inputs)
            rval.extend(orderings.get(obj, []))
        else:
            assert not orderings.get(obj, [])
        return rval
    return general_toposort(o, deps)

