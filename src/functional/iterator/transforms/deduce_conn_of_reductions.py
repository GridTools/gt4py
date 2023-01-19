from collections.abc import Iterable, Iterator
from typing import TypeGuard

from eve import NodeTranslator
from functional.iterator import ir


def _is_shifted(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(arg, ir.FunCall)
        and isinstance(arg.fun, ir.FunCall)
        and arg.fun.fun == ir.SymRef(id="shift")
    )


def _is_applied_lift(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(arg, ir.FunCall)
        and isinstance(arg.fun, ir.FunCall)
        and arg.fun.fun == ir.SymRef(id="lift")
    )


def _is_shifted_or_lifted_and_shifted(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def _get_shifted_args(reduce_args: Iterable[ir.Expr]) -> Iterator[ir.FunCall]:
    return filter(
        _is_shifted_or_lifted_and_shifted,
        reduce_args,
    )


def _is_list_of_funcalls(lst: list) -> TypeGuard[list[ir.FunCall]]:
    return all(isinstance(f, ir.FunCall) for f in lst)


def _get_partial_offset_tag(arg: ir.FunCall) -> str:
    if _is_shifted(arg):
        assert isinstance(arg.fun, ir.FunCall)
        offset = arg.fun.args[-1]
        assert isinstance(offset, ir.OffsetLiteral)
        assert isinstance(offset.value, str)
        return offset.value
    else:
        assert _is_applied_lift(arg)
        assert _is_list_of_funcalls(arg.args)
        partial_offsets = [_get_partial_offset_tag(arg) for arg in arg.args]
        assert all(o == partial_offsets[0] for o in partial_offsets)
        return partial_offsets[0]


def _get_partial_offsets(reduce_args: Iterable[ir.Expr]) -> Iterable[str]:
    return [_get_partial_offset_tag(arg) for arg in _get_shifted_args(reduce_args)]


def _is_reduce(node: ir.FunCall):
    return isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="reduce")


class DeduceConnOfReductions(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node, **kwargs):
        return cls().visit(node, **kwargs)

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if not _is_reduce(node):
            return node

        annexed_node = ir.FunCall(fun=node.fun, args=node.args)
        annexed_node.annex.reduction_offsets = _get_partial_offsets(node.args)
        return annexed_node
