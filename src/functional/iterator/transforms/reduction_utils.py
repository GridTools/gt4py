import abc
import types
from collections.abc import Iterable, Iterator
from typing import TypeGuard

from functional import common


class _Expr(abc.ABC):
    ...


class _SymRef(_Expr):
    @property
    @abc.abstractmethod
    def id(self):  # noqa: A003
        ...


class _FunCall(_Expr):
    @property
    @abc.abstractmethod
    def fun(self):
        ...

    @property
    @abc.abstractmethod
    def args(self):
        ...


class _OffsetLiteral(_Expr):
    @property
    @abc.abstractmethod
    def value(self):
        ...


def _is_shifted(arg: _Expr) -> TypeGuard[_FunCall]:
<<<<<<< Updated upstream
    return (
        isinstance(arg, _FunCall)
        and isinstance(arg.fun, _FunCall)
        and isinstance(arg.fun.fun, _SymRef)
        and arg.fun.fun.id == "shift"
    )
=======
    return isinstance(arg, _FunCall) and isinstance(arg.fun, _SymRef) and arg.fun.id == "shift"
>>>>>>> Stashed changes


def _is_applied_lift(arg: _Expr) -> TypeGuard[_FunCall]:
    return (
        isinstance(arg, _FunCall)
        and isinstance(arg.fun, _FunCall)
        and isinstance(arg.fun.fun, _SymRef)
        and arg.fun.fun.id == "lift"
    )


def _is_shifted_or_lifted_and_shifted(arg: _Expr) -> TypeGuard[_FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def _get_shifted_args(reduce_args: Iterable[_Expr]) -> Iterator[_FunCall]:
    return filter(
        _is_shifted_or_lifted_and_shifted,
        reduce_args,
    )


def _is_list_of_funcalls(lst: list) -> TypeGuard[list[_FunCall]]:
    return all(isinstance(f, _FunCall) for f in lst)


def _get_partial_offset_tag(arg: _FunCall) -> str:
    if _is_shifted(arg):
<<<<<<< Updated upstream
        assert isinstance(arg.fun, _FunCall)
        offset = arg.fun.args[-1]
=======
        assert isinstance(arg.fun, _SymRef)
        offset = arg.args[-1]
>>>>>>> Stashed changes
        assert isinstance(offset, _OffsetLiteral)
        assert isinstance(offset.value, str)
        return offset.value
    else:
        assert _is_applied_lift(arg)
        assert _is_list_of_funcalls(arg.args)
        partial_offsets = [_get_partial_offset_tag(arg) for arg in arg.args]
        assert all(o == partial_offsets[0] for o in partial_offsets)
        return partial_offsets[0]


def _get_partial_offsets(reduce_args: Iterable[_Expr]) -> Iterable[str]:
    return [_get_partial_offset_tag(arg) for arg in _get_shifted_args(reduce_args)]


def _is_reduce(node: _FunCall):
    return (
        isinstance(node.fun, _FunCall)
        and isinstance(node.fun.fun, _SymRef)
        and node.fun.fun.id == "reduce"
    )


def register_ir(ir: types.ModuleType) -> None:
    """
    Register an IR (a module containing eve nodes) to work with the functions of this module.

    They work on IRs with the same Syntax (tree structure) and same semantics as Iterator IR.
    """
    _Expr.register(ir.Expr)
    _FunCall.register(ir.FunCall)
    _SymRef.register(ir.SymRef)
    _OffsetLiteral.register(ir.OffsetLiteral)


def get_connectivity(applied_reduce_node: _FunCall, offset_provider) -> common.Connectivity:
    """Return single connectivity that is compatible with the arguments of the reduce."""
    if not isinstance(applied_reduce_node, _FunCall):
        raise TypeError(
            f"{applied_reduce_node=} is not a `FunCall` of a registered IR. Did you forget to call `.register_ir()`?"
        )
    if not _is_reduce(applied_reduce_node):
        raise ValueError("Expected a call to a `reduce` object, i.e. `reduce(...)(...)`.")

    connectivities = []
    for o in _get_partial_offsets(applied_reduce_node.args):
        connectivities.append(offset_provider[o])

    if not connectivities:
        raise RuntimeError("Couldn't detect partial shift in any arguments of reduce.")

    if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
        # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
        raise RuntimeError("Arguments to reduce have incompatible partial shifts.")
    return connectivities[0]
