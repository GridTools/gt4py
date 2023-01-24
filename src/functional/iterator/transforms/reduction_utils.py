from collections.abc import Iterable, Iterator
from typing import TypeGuard

from functional import common
from functional.iterator import ir as itir


def _is_shifted(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "shift"
    )


def _is_applied_lift(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and isinstance(arg.fun.fun, itir.SymRef)
        and arg.fun.fun.id == "lift"
    )


def _is_shifted_or_lifted_and_shifted(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def get_shifted_args(reduce_args: Iterable[itir.Expr]) -> Iterator[itir.FunCall]:
    return filter(
        _is_shifted_or_lifted_and_shifted,
        reduce_args,
    )


def _is_list_of_funcalls(lst: list) -> TypeGuard[list[itir.FunCall]]:
    return all(isinstance(f, itir.FunCall) for f in lst)


def _get_partial_offset_tag(arg: itir.FunCall) -> str:
    if _is_shifted(arg):
        assert isinstance(arg.fun, itir.FunCall)
        offset = arg.fun.args[-1]
        assert isinstance(offset, itir.OffsetLiteral)
        assert isinstance(offset.value, str)
        return offset.value
    else:
        assert _is_applied_lift(arg)
        assert _is_list_of_funcalls(arg.args)
        partial_offsets = [_get_partial_offset_tag(arg) for arg in arg.args]
        assert all(o == partial_offsets[0] for o in partial_offsets)
        return partial_offsets[0]


def _get_partial_offsets(reduce_args: Iterable[itir.Expr]) -> Iterable[str]:
    return [_get_partial_offset_tag(arg) for arg in get_shifted_args(reduce_args)]


def is_reduce(node: itir.FunCall) -> TypeGuard[itir.FunCall]:
    return (
        isinstance(node.fun, itir.FunCall)
        and isinstance(node.fun.fun, itir.SymRef)
        and node.fun.fun.id == "reduce"
    )


def get_connectivity(
    applied_reduce_node: itir.FunCall,
    offset_provider: dict[str, common.Dimension | common.Connectivity],
) -> common.Connectivity:
    """Return single connectivity that is compatible with the arguments of the reduce."""
    if not is_reduce(applied_reduce_node):
        raise ValueError("Expected a call to a `reduce` object, i.e. `reduce(...)(...)`.")

    connectivities: list[common.Connectivity] = []
    for o in _get_partial_offsets(applied_reduce_node.args):
        conn = offset_provider[o]
        assert isinstance(conn, common.Connectivity)
        connectivities.append(conn)

    if not connectivities:
        raise RuntimeError("Couldn't detect partial shift in any arguments of reduce.")

    if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
        # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
        raise RuntimeError("Arguments to reduce have incompatible partial shifts.")
    return connectivities[0]
