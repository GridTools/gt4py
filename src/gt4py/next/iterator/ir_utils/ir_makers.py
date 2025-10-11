# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing
from typing import Any, Callable, Iterable, Optional, TypeAlias, Union

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.type_system import type_specifications as ts, type_translation


ExprLike: TypeAlias = Union[str, core_defs.Scalar, common.Dimension, itir.Expr]


def sym(sym_or_name: Union[str, itir.Sym], type_: str | ts.TypeSpec | None = None) -> itir.Sym:
    """
    Convert to Sym if necessary.

    Examples
    --------
    >>> sym("a")
    Sym(id=SymbolName('a'))

    >>> sym(itir.Sym(id="b"))
    Sym(id=SymbolName('b'))

    >>> a = sym("a", "float32")
    >>> a.id, a.type
    (SymbolName('a'), ScalarType(kind=<ScalarKind.FLOAT32: 10>, shape=None))
    """
    if isinstance(sym_or_name, itir.Sym):
        assert not type_
        return sym_or_name
    return itir.Sym(id=sym_or_name, type=ensure_type(type_))


def ref(
    ref_or_name: Union[str, itir.SymRef],
    type_: str | ts.TypeSpec | None = None,
    annex: dict[str, Any] | None = None,
) -> itir.SymRef:
    """
    Convert to SymRef if necessary.

    Examples
    --------
    >>> ref("a")
    SymRef(id=SymbolRef('a'))

    >>> ref(itir.SymRef(id="b"))
    SymRef(id=SymbolRef('b'))

    >>> a = ref("a", "float32")
    >>> a.id, a.type
    (SymbolRef('a'), ScalarType(kind=<ScalarKind.FLOAT32: 10>, shape=None))
    """
    if isinstance(ref_or_name, itir.SymRef):
        assert not type_
        assert not annex
        return ref_or_name
    ref = itir.SymRef(id=ref_or_name, type=ensure_type(type_))
    if annex is not None:
        for key, value in annex.items():
            setattr(ref.annex, key, value)
    return ref


def ensure_expr(expr_like: ExprLike) -> itir.Expr:
    """
    Convert literals into a SymRef or Literal and let expressions pass unchanged.

    Examples
    --------
    >>> ensure_expr("a")
    SymRef(id=SymbolRef('a'))

    >>> ensure_expr(3)
    Literal(value='3', type=ScalarType(kind=<ScalarKind.INT32: 6>, shape=None))

    >>> ensure_expr(itir.OffsetLiteral(value="i"))
    OffsetLiteral(value='i')
    """
    if isinstance(expr_like, str):
        return ref(expr_like)
    elif core_defs.is_scalar_type(expr_like):
        return literal_from_value(expr_like)
    elif expr_like is None:
        return itir.NoneLiteral()
    elif isinstance(expr_like, common.Dimension):
        return axis_literal(expr_like)
    assert isinstance(expr_like, itir.Expr), expr_like
    return expr_like


def ensure_offset(str_or_offset: Union[str, int, itir.OffsetLiteral]) -> itir.OffsetLiteral:
    """
    Convert Python literals into an OffsetLiteral and let OffsetLiterals pass unchanged.

    Examples
    --------
    >>> ensure_offset("V2E")
    OffsetLiteral(value='V2E')

    >>> ensure_offset(itir.OffsetLiteral(value="J"))
    OffsetLiteral(value='J')
    """
    if isinstance(str_or_offset, (str, int)):
        return itir.OffsetLiteral(value=str_or_offset)
    return str_or_offset


def ensure_type(type_: str | ts.TypeSpec | None) -> ts.TypeSpec | None:
    if isinstance(type_, str):
        return ts.ScalarType(kind=getattr(ts.ScalarKind, type_.upper()))
    assert isinstance(type_, ts.TypeSpec) or type_ is None
    return type_


class lambda_:
    """
    Create a lambda from params and an expression.

    Examples
    --------
    >>> lambda_("a")(deref("a"))  # doctest: +ELLIPSIS
    Lambda(params=[Sym(id=SymbolName('a'))], expr=FunCall(fun=SymRef(id=SymbolRef('deref')), args=[SymRef(id=SymbolRef('a'))]))
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, expr):
        return itir.Lambda(params=[sym(arg) for arg in self.args], expr=ensure_expr(expr))


class call:
    """
    Create a FunCall from an expression and arguments.

    Examples
    --------
    >>> call("plus")(1, 1)
    FunCall(fun=SymRef(id=SymbolRef('plus')), args=[Literal(value='1', type=ScalarType(kind=<ScalarKind.INT32: 6>, shape=None)), Literal(value='1', type=ScalarType(kind=<ScalarKind.INT32: 6>, shape=None))])
    """

    def __init__(self, expr):
        self.fun = ensure_expr(expr)

    def __call__(self, *exprs):
        return itir.FunCall(fun=self.fun, args=[ensure_expr(expr) for expr in exprs])


def deref(expr):
    """Create a deref FunCall, shorthand for ``call("deref")(expr)``."""
    return call("deref")(expr)


def plus(left, right):
    """Create a plus FunCall, shorthand for ``call("plus")(left, right)``."""
    return call("plus")(left, right)


def minus(left, right):
    """Create a minus FunCall, shorthand for ``call("minus")(left, right)``."""
    return call("minus")(left, right)


def multiplies_(left, right):
    """Create a multiplies FunCall, shorthand for ``call("multiplies")(left, right)``."""
    return call("multiplies")(left, right)


def divides_(left, right):
    """Create a divides FunCall, shorthand for ``call("divides")(left, right)``."""
    return call("divides")(left, right)


def and_(left, right):
    """Create an and_ FunCall, shorthand for ``call("and_")(left, right)``."""
    return call("and_")(left, right)


def or_(left, right):
    """Create an or_ FunCall, shorthand for ``call("or_")(left, right)``."""
    return call("or_")(left, right)


def xor_(left, right):
    """Create an xor_ FunCall, shorthand for ``call("xor_")(left, right)``."""
    return call("xor_")(left, right)


def greater(left, right):
    """Create a greater FunCall, shorthand for ``call("greater")(left, right)``."""
    return call("greater")(left, right)


def less(left, right):
    """Create a less FunCall, shorthand for ``call("less")(left, right)``."""
    return call("less")(left, right)


def less_equal(left, right):
    """Create a less_equal FunCall, shorthand for ``call("less_equal")(left, right)``."""
    return call("less_equal")(left, right)


def greater_equal(left, right):
    """Create a greater_equal FunCall, shorthand for ``call("greater_equal")(left, right)``."""
    return call("greater_equal")(left, right)


def not_eq(left, right):
    """Create a not_eq FunCall, shorthand for ``call("not_eq")(left, right)``."""
    return call("not_eq")(left, right)


def eq(left, right):
    """Create a eq FunCall, shorthand for ``call("eq")(left, right)``."""
    return call("eq")(left, right)


def not_(expr):
    """Create a not_ FunCall, shorthand for ``call("not_")(expr)``."""
    return call("not_")(expr)


def make_tuple(*args):
    """Create a make_tuple FunCall, shorthand for ``call("make_tuple")(*args)``."""
    return call("make_tuple")(*args)


def tuple_get(index: str | int | itir.Literal, tuple_expr):
    """Create a tuple_get FunCall, shorthand for ``call("tuple_get")(index, tuple_expr)``."""
    if not isinstance(index, itir.Literal):
        index = literal(str(index), builtins.INTEGER_INDEX_BUILTIN)
    return call("tuple_get")(index, tuple_expr)


def if_(cond, true_val, false_val):
    """Create a if_ FunCall, shorthand for ``call("if_")(expr)``."""
    return call("if_")(cond, true_val, false_val)


def concat_where(cond, true_field, false_field):
    """Create a concat_where FunCall, shorthand for ``call("concat_where")(expr)``."""

    return call("concat_where")(cond, true_field, false_field)


def lift(expr):
    """Create a lift FunCall, shorthand for ``call(call("lift")(expr))``."""
    return call(call("lift")(expr))


class let:
    """
    Create a lambda expression that works as a let.

    Examples
    --------
    >>> str(let("a", "b")("a"))  # doctest: +ELLIPSIS
    '(λ(a) → a)(b)'
    >>> str(let(("a", 1), ("b", 2))(plus("a", "b")))
    '(λ(a, b) → a + b)(1, 2)'
    """

    @typing.overload
    def __init__(self, var: str | itir.Sym, init_form: itir.Expr | str): ...

    @typing.overload
    def __init__(self, *args: tuple[str | itir.Sym, itir.Expr | str]): ...

    def __init__(self, *args):
        if all(isinstance(arg, tuple) and len(arg) == 2 for arg in args):
            assert isinstance(args, tuple)
            assert all(isinstance(arg, tuple) and len(arg) == 2 for arg in args)
            self.vars = [var for var, _ in args]
            self.init_forms = [init_form for _, init_form in args]
        elif len(args) == 2:
            self.vars = [args[0]]
            self.init_forms = [args[1]]
        else:
            raise TypeError(
                "Invalid arguments: expected a variable name and an init form or a list thereof."
            )

    def __call__(self, form) -> itir.FunCall:
        return call(lambda_(*self.vars)(form))(*self.init_forms)


def shift(offset, value=None):
    """
    Create a shift call.

    Examples
    --------
    >>> shift("i", 0)("a")
    FunCall(fun=FunCall(fun=SymRef(id=SymbolRef('shift')), args=[OffsetLiteral(value='i'), OffsetLiteral(value=0)]), args=[SymRef(id=SymbolRef('a'))])

    >>> shift("V2E")("b")
    FunCall(fun=FunCall(fun=SymRef(id=SymbolRef('shift')), args=[OffsetLiteral(value='V2E')]), args=[SymRef(id=SymbolRef('b'))])
    """
    offset = ensure_offset(offset)
    args = [offset]
    if value is not None:
        if isinstance(value, int):
            value = ensure_offset(value)
        if isinstance(value, itir.Literal) and value.type.kind in (
            ts.ScalarKind.INT32,
            ts.ScalarKind.INT64,
        ):
            value = itir.OffsetLiteral(value=int(value.value))
        args.append(value)
    return call(call("shift")(*args))


def literal(value: str, type_: str | ts.TypeSpec) -> itir.Literal:
    return itir.Literal(value=value, type=ensure_type(type_))


def literal_from_value(val: core_defs.Scalar) -> itir.Literal:
    """
    Make a literal node from a value.

    >>> literal_from_value(1.0)
    Literal(value='1.0', type=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None))
    >>> literal_from_value(1)
    Literal(value='1', type=ScalarType(kind=<ScalarKind.INT32: 6>, shape=None))
    >>> literal_from_value(2147483648)
    Literal(value='2147483648', type=ScalarType(kind=<ScalarKind.INT64: 8>, shape=None))
    >>> literal_from_value(True)
    Literal(value='True', type=ScalarType(kind=<ScalarKind.BOOL: 1>, shape=None))
    """
    if not isinstance(val, core_defs.Scalar):  # type: ignore[arg-type] # mypy bug #11673
        raise ValueError(f"Value must be a scalar, got '{type(val).__name__}'.")

    # At the time this has been written the iterator module has its own type system that is
    # uncoupled from the one used in the frontend. However since we decided to eventually replace
    # it with the frontend type system we already use it here (avoiding unnecessary code
    # duplication).
    type_spec = type_translation.from_value(val)
    assert isinstance(type_spec, ts.ScalarType)

    typename = type_spec.kind.name.lower()
    assert typename in builtins.TYPE_BUILTINS

    return literal(str(val), typename)


def literal_from_tuple_value(
    val: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
) -> itir.FunCall | itir.Literal:
    """
    Create a `make_tuple` with literals from a tuple of values.

    >>> str(literal_from_tuple_value((1.0, (2.0, 3.0))))
    '{1.0, {2.0, 3.0}}'
    """
    if isinstance(val, tuple):
        return make_tuple(*(literal_from_tuple_value(v) for v in val))
    return literal_from_value(val)


def neighbors(offset, it):
    offset = ensure_offset(offset)
    return call("neighbors")(offset, it)


def lifted_neighbors(offset, it) -> itir.Expr:
    """
    Create a lifted neighbors call.

    Examples
    --------
    >>> str(lifted_neighbors("off", "a"))
    '(↑(λ(it) → neighbors(offₒ, it)))(a)'
    """
    return lift(lambda_("it")(neighbors(offset, "it")))(it)


def as_fieldop_neighbors(
    offset: str | itir.OffsetLiteral, field: str | itir.Expr, domain: Optional[itir.FunCall] = None
) -> itir.Expr:
    """
    Create a fieldop for neighbors call.

    Examples
    --------
    >>> str(as_fieldop_neighbors("off", "a"))
    '(⇑(λ(it) → neighbors(offₒ, it)))(a)'
    """
    return as_fieldop(lambda_("it")(neighbors(offset, "it")), domain)(field)


def as_fieldop_deref_list_get(
    list_idx: str | itir.Expr, local_field: str | itir.Expr, domain: Optional[itir.FunCall] = None
) -> itir.Expr:
    """
    Create a fieldop for list_get call.

    Examples
    --------
    >>> str(as_fieldop_deref_list_get("idx", "lst"))
    '(⇑(λ(it) → list_get(idx, ·it)))(lst)'
    """
    return as_fieldop(lambda_("it")(list_get(list_idx, deref("it"))), domain)(local_field)


def promote_to_const_iterator(expr: str | itir.Expr) -> itir.Expr:
    """
    Create a lifted nullary lambda that captures `expr`.

    Examples
    --------
    >>> str(promote_to_const_iterator("foo"))
    '(↑(λ() → foo))()'
    """
    return lift(lambda_()(expr))()


def promote_to_lifted_stencil(op: str | itir.SymRef | Callable) -> Callable[..., itir.FunCall]:
    """
    Promotes a function `op` from values to iterators.

    `op` is a function from values to value.

    Returns:
        A lifted stencil, a function from iterators to iterator.

    Examples
    --------
    >>> str(promote_to_lifted_stencil("op")("a", "b"))
    '(↑(λ(__arg0, __arg1) → op(·__arg0, ·__arg1)))(a, b)'
    """
    if isinstance(op, (str, itir.SymRef, itir.Lambda)):
        op = call(op)

    def _impl(*its: itir.Expr) -> itir.FunCall:
        args = [
            f"__arg{i}" for i in range(len(its))
        ]  # TODO: `op` must not contain `SymRef(id="__argX")`
        return lift(lambda_(*args)(op(*[deref(arg) for arg in args])))(*its)

    return _impl


def domain(
    grid_type: Union[common.GridType, str],
    ranges: dict[common.Dimension, tuple[itir.Expr, itir.Expr]],
) -> itir.FunCall:
    """
    >>> IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
    >>> JDim = common.Dimension(value="JDim", kind=common.DimensionKind.HORIZONTAL)
    >>> str(domain(common.GridType.CARTESIAN, {IDim: (0, 10), JDim: (0, 20)}))
    'c⟨ IDimₕ: [0, 10[, JDimₕ: [0, 20[ ⟩'
    >>> str(domain(common.GridType.UNSTRUCTURED, {IDim: (0, 10), JDim: (0, 20)}))
    'u⟨ IDimₕ: [0, 10[, JDimₕ: [0, 20[ ⟩'
    """
    if isinstance(grid_type, common.GridType):
        grid_type = f"{grid_type!s}_domain"
    expr = call(grid_type)(
        *[
            named_range(
                d,
                r[0],
                r[1],
            )
            for d, r in ranges.items()
        ]
    )
    expr.type = ts.DomainType(dims=list(ranges.keys()))
    return expr


def get_field_domain(
    grid_type: Union[common.GridType, str],
    field: str | itir.Expr,
    dims: Iterable[common.Dimension] | None = None,
) -> itir.Expr:
    if isinstance(field, itir.Expr) and isinstance(field.type, ts.FieldType):
        assert dims is None or all(d1 == d2 for d1, d2 in zip(field.type.dims, dims, strict=True))
        dims = field.type.dims
    else:
        if dims is None:
            raise ValueError("Field expression must be typed if 'dims' is not given.")

    return domain(
        grid_type,
        {
            dim: (
                tuple_get(0, call("get_domain_range")(field, dim)),
                tuple_get(1, call("get_domain_range")(field, dim)),
            )
            for dim in dims
        },
    )


def named_range(
    dim: itir.AxisLiteral | common.Dimension, start: itir.Expr, stop: itir.Expr
) -> itir.FunCall:
    return call("named_range")(dim, start, stop)


def as_fieldop(expr: itir.Expr | str, domain: Optional[itir.Expr] = None) -> Callable:
    """
    Create an `as_fieldop` call.

    Examples
    --------
    >>> str(as_fieldop(lambda_("it1", "it2")(plus(deref("it1"), deref("it2"))))("field1", "field2"))
    '(⇑(λ(it1, it2) → ·it1 + ·it2))(field1, field2)'
    """
    from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils

    result = call(
        call("as_fieldop")(
            *(
                (
                    expr,
                    domain,
                )
                if domain
                else (expr,)
            )
        )
    )

    def _populate_domain_annex_wrapper(*args, **kwargs):
        node = result(*args, **kwargs)
        # note: if the domain is not a direct construction, e.g. because it is only a reference
        # to a domain defined in a let, don't populate the annex, since we can not create a
        # symbolic domain for it.
        if domain and cpm.is_call_to(domain, ("cartesian_domain", "unstructured_domain")):
            node.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
        return node

    return _populate_domain_annex_wrapper


def op_as_fieldop(
    op: str | itir.SymRef | itir.Lambda | Callable, domain: Optional[itir.FunCall] = None
) -> Callable[..., itir.FunCall]:
    """
    Promotes a function `op` to a field_operator.

    Args:
        op: a function from values to value.
        domain: the domain of the returned field.

    Returns:
        A function from Fields to Field.

    Examples:
        >>> str(op_as_fieldop("op")("a", "b"))
        '(⇑(λ(__arg0, __arg1) → op(·__arg0, ·__arg1)))(a, b)'
    """
    if isinstance(op, (str, itir.SymRef, itir.Lambda)):
        op = call(op)

    def _impl(*its: itir.Expr) -> itir.FunCall:
        args = [
            f"__arg{i}" for i in range(len(its))
        ]  # TODO: `op` must not contain `SymRef(id="__argX")`
        return as_fieldop(lambda_(*args)(op(*[deref(arg) for arg in args])), domain)(*its)

    return _impl


def axis_literal(dim: common.Dimension) -> itir.AxisLiteral:
    return itir.AxisLiteral(value=dim.value, kind=dim.kind)


def cast_as_fieldop(type_: str, domain: Optional[itir.FunCall] = None):
    """
    Promotes the function `cast_` to a field_operator.

    Args:
        type_: the target type to be passed as argument to `cast_` function.
        domain: the domain of the returned field.

    Returns:
        A function from Fields to Field.

    Examples:
        >>> str(cast_as_fieldop("float32")("a"))
        '(⇑(λ(__arg0) → cast_(·__arg0, float32)))(a)'
    """

    def _impl(it: itir.Expr) -> itir.FunCall:
        return op_as_fieldop(lambda v: call("cast_")(v, type_), domain)(it)

    return _impl


def index(dim: common.Dimension) -> itir.FunCall:
    """
    Create a call to the `index` builtin, shorthand for `call("index")(axis)`,
    after converting the given dimension to `itir.AxisLiteral`.

    Args:
        dim: the dimension corresponding to the index axis.

    Returns:
        A function that constructs a Field of indices in the given dimension.
    """
    return call("index")(itir.AxisLiteral(value=dim.value, kind=dim.kind))


def map_(op):
    """Create a `map_` call."""
    return call(call("map_")(op))


def reduce(op, expr):
    """Create a `reduce` call."""
    return call(call("reduce")(op, expr))


def scan(expr, forward, init):
    """Create a `scan` call."""
    return call("scan")(expr, forward, init)


def list_get(list_idx, list_):
    """Create a `list_get` call."""
    return call("list_get")(list_idx, list_)


def maximum(expr1, expr2):
    """Create a `maximum` call."""
    return call("maximum")(expr1, expr2)


def minimum(expr1, expr2):
    """Create a `minimum` call."""
    return call("minimum")(expr1, expr2)


def cast_(expr, dtype: ts.ScalarType | str):
    """Create a `cast_` call."""
    if isinstance(dtype, ts.ScalarType):
        dtype = dtype.kind.name.lower()
    return call("cast_")(expr, dtype)


def can_deref(expr):
    """Create a `can_deref` call."""
    return call("can_deref")(expr)


def compose(a: itir.SymRef | itir.Lambda, b: itir.SymRef | itir.Lambda) -> itir.Lambda:
    # TODO(havogt): `a`, `b` must not contain `SymRef(id="_comp")` for a `Sym` in a parent scope
    return lambda_("__comp")(call(a)(call(b)("__comp")))
