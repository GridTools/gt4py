# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Find fields that a statement writes and also reads at a non-zero offset.

ADR 18 (rule 3) allows a field to be both input and output of the same statement only if the
output depends elementwise on the input, which rules out statements like `A[0:10] = A[1:11]`.
Backends rely on this: the DaCe backend, for example, removes what looks like a self-copy, and
the GTFN backend writes all outputs of a statement in the same loop that reads its inputs. This
module detects violations on the field-view GTIR of a statement, i.e. before any inlining, fusion
or temporary extraction, so that the result does not depend on the backend.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Optional, TypeAlias

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import inline_fundefs, trace_shifts
from gt4py.next.type_system import type_specifications as ts


#: A program parameter, or an element of a tuple-typed parameter, given by its path.
_FieldRef: TypeAlias = tuple[str, tuple[int, ...]]


def _overlaps(a: _FieldRef, b: _FieldRef) -> bool:
    (a_name, a_path), (b_name, b_path) = a, b
    common_length = min(len(a_path), len(b_path))
    return a_name == b_name and a_path[:common_length] == b_path[:common_length]


@dataclasses.dataclass(frozen=True)
class _Offset:
    """Position of a read relative to the point being computed."""

    #: Displacement along non-staggered dimensions, zero entries removed, sorted by dimension.
    cartesian: tuple[tuple[str, float], ...] = ()
    #: Connectivities the read goes through, in order of appearance.
    unstructured: tuple[str, ...] = ()
    #: Whether the read is at a data-dependent offset (e.g. 'as_offset').
    dynamic: bool = False

    @property
    def is_zero(self) -> bool:
        return not (self.cartesian or self.unstructured or self.dynamic)

    def then(self, other: _Offset) -> _Offset:
        displacement = dict(self.cartesian)
        for dim, value in other.cartesian:
            displacement[dim] = displacement.get(dim, 0) + value
        return _Offset(
            cartesian=tuple(sorted((dim, value) for dim, value in displacement.items() if value)),
            unstructured=tuple(dict.fromkeys(self.unstructured + other.unstructured)),
            dynamic=self.dynamic or other.dynamic,
        )

    def __str__(self) -> str:
        parts = [f"shifted by {value:+g} along '{dim}'" for dim, value in self.cartesian]
        parts += [f"through '{tag}'" for tag in self.unstructured]
        if self.dynamic:
            parts.append("at a data-dependent offset")
        return " and ".join(parts) if parts else "at the same point"


_ZERO = _Offset()


@dataclasses.dataclass(frozen=True)
class _Access:
    offset: _Offset
    #: The `_Reads.token` of the selection (see `_Reads.selection_of`) the read went through.
    via: Optional[int] = None


_tokens = itertools.count()


@dataclasses.dataclass(frozen=True)
class _Reads:
    """Abstract value of a non-tuple field expression: the parameters it reads and where."""

    accesses: Mapping[_FieldRef, frozenset[_Access]] = dataclasses.field(default_factory=dict)
    #: Set if the expression is (an element of) a parameter itself, i.e. a copy of it.
    identity_of: Optional[_FieldRef] = None
    #: Set if, at every point, the expression is either the parameter (element) at that point or
    #: does not read it at all, e.g. `b if flag else a` or `concat_where(KDim < n, b, a)` for
    #: parameter `a` and `b` not reading `a`.
    selection_of: Optional[_FieldRef] = None
    token: int = dataclasses.field(default_factory=lambda: next(_tokens), compare=False)

    def consumed(self) -> dict[_FieldRef, frozenset[_Access]]:
        """The accesses of this value, attributed to it if it is a selection."""
        if self.selection_of is None:
            return dict(self.accesses)
        return {
            ref: frozenset(_Access(a.offset, self.token) for a in accesses)
            if ref == self.selection_of
            else accesses
            for ref, accesses in self.accesses.items()
        }


#: Abstract value of a field expression, structured like the expression's tuple type.
_Value: TypeAlias = _Reads | tuple["_Value", ...]


@dataclasses.dataclass(frozen=True)
class _Closure:
    """A `let`-bound lambda, e.g. the continuation of an `if` that returns in one branch only."""

    lambda_: itir.Lambda
    env: Mapping[str, _Binding]


_Binding: TypeAlias = _Value | _Closure


def _param(ref: _FieldRef) -> _Reads:
    return _Reads({ref: frozenset({_Access(_ZERO)})}, identity_of=ref, selection_of=ref)


def _union(
    *accesses: Mapping[_FieldRef, frozenset[_Access]],
) -> dict[_FieldRef, frozenset[_Access]]:
    result: dict[_FieldRef, frozenset[_Access]] = {}
    for el in accesses:
        for ref, ref_accesses in el.items():
            result[ref] = result.get(ref, frozenset()) | ref_accesses
    return result


def _leaves(value: _Value) -> Iterator[_Reads]:
    if isinstance(value, tuple):
        for el in value:
            yield from _leaves(el)
    else:
        yield value


def _flatten(value: _Value) -> _Reads:
    """All reads of a (possibly tuple-valued) expression, as an opaque pointwise computation."""
    return _Reads(_union(*(leaf.consumed() for leaf in _leaves(value))))


def _tuple_get(index: int, value: _Value) -> _Value:
    if isinstance(value, tuple):
        return value[index]
    if value.identity_of is not None:
        name, path = value.identity_of
        return _param((name, (*path, index)))
    # an element of a tuple computed by a single stencil: assume it reads what the whole does
    return _Reads(value.consumed())


def _merge(a: _Value, b: _Value) -> _Value:
    """Abstract value of an expression evaluating to either `a` or `b` (pointwise)."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        size = len(a) if isinstance(a, tuple) else len(b)  # type: ignore[arg-type]  # one is a tuple
        if isinstance(a, tuple) and isinstance(b, tuple) and len(a) != len(b):
            return _flatten((a, b))
        return tuple(_merge(_tuple_get(i, a), _tuple_get(i, b)) for i in range(size))

    selection_of = None
    if candidates := {v.selection_of for v in (a, b) if v.selection_of is not None}:
        (ref, *others) = candidates
        if not others and all(
            v.selection_of == ref or not any(_overlaps(r, ref) for r in v.accesses) for v in (a, b)
        ):
            selection_of = ref
    return _Reads(
        _union(a.accesses, b.accesses),
        identity_of=a.identity_of if a.identity_of == b.identity_of else None,
        selection_of=selection_of,
    )


def _is_identity_stencil(stencil: itir.Expr) -> bool:
    return cpm.is_ref_to(stencil, "deref") or (
        isinstance(stencil, itir.Lambda)
        and len(stencil.params) == 1
        and stencil.expr == im.deref(stencil.params[0].id)
    )


def _dimension(axis: itir.AxisLiteral) -> common.Dimension:
    return common.Dimension(axis.value, axis.kind)


def _is_cartesian_offset_type(offset_type: ts.OffsetType) -> bool:
    return len(offset_type.target) == 1 and offset_type.source == offset_type.target[0]


@dataclasses.dataclass
class _FieldReadTracer:
    function_definitions: Mapping[str, itir.FunctionDefinition]
    offset_types: Mapping[str, ts.OffsetType]
    _stencil_shifts: dict[int, list[set[tuple]]] = dataclasses.field(default_factory=dict)

    def offset(self, shift: tuple) -> _Offset:
        """Convert a shift recorded by `trace_shifts` into an `_Offset`."""
        if len(shift) % 2:
            return _Offset(dynamic=True)
        result = _ZERO
        for tag, index in zip(shift[0::2], shift[1::2]):
            is_static = isinstance(index, itir.OffsetLiteral) and isinstance(index.value, int)
            if isinstance(tag, itir.CartesianOffset):
                if not is_static:
                    result = result.then(_Offset(dynamic=True))
                    continue
                domain, codomain = _dimension(tag.domain), _dimension(tag.codomain)
                # a staggered index sits half a cell below its base index (see ADR 26)
                value = (
                    index.value + (common.is_staggered(domain) - common.is_staggered(codomain)) / 2
                )
                dim = common.as_non_staggered(domain).value
                result = result.then(_Offset(cartesian=((dim, value),)))
            elif isinstance(tag, itir.OffsetLiteral) and isinstance(tag.value, str):
                offset_type = self.offset_types.get(tag.value)
                if offset_type is None and is_static and index.value == 0:
                    pass  # unknown offset: assume it is Cartesian, i.e. a no-op at index 0
                elif offset_type is not None and _is_cartesian_offset_type(offset_type):
                    if is_static:
                        dim = common.as_non_staggered(offset_type.source).value
                        result = result.then(_Offset(cartesian=((dim, index.value),)))
                    else:
                        result = result.then(_Offset(dynamic=True))
                else:
                    result = result.then(_Offset(unstructured=(tag.value,)))
            else:
                result = result.then(_Offset(dynamic=True))
        return result

    def stencil_shifts(self, stencil: itir.Expr, num_args: int) -> list[set[tuple]]:
        # Function definitions are traced once per call, not inlined, so their stencils are the
        # same nodes in every call and the expensive shift tracing can be cached by identity.
        key = id(stencil)
        if key not in self._stencil_shifts:
            # a stencil can call function definitions, e.g. a scan pass calling a field operator
            stencil = inline_fundefs.InlineFundefs().visit(
                stencil, symtable=self.function_definitions
            )
            self._stencil_shifts[key] = trace_shifts.trace_stencil(stencil, num_args=num_args)
        return self._stencil_shifts[key]

    def apply(self, closure: _Closure, args: Sequence[_Value]) -> _Value:
        params = (str(param.id) for param in closure.lambda_.params)
        return self.visit(
            closure.lambda_.expr, {**closure.env, **dict(zip(params, args, strict=True))}
        )

    def visit(self, expr: itir.Expr, env: Mapping[str, _Binding]) -> _Value:
        # bind `let`s iteratively, they can be nested very deeply
        while cpm.is_let(expr):
            env = {
                **env,
                **{
                    param.id: _Closure(arg, env)
                    if isinstance(arg, itir.Lambda)
                    else self.visit(arg, env)
                    for param, arg in zip(expr.fun.params, expr.args, strict=True)
                },
            }
            expr = expr.fun.expr

        if isinstance(expr, itir.SymRef):
            if expr.id not in env:
                # not bound: a parameter of the program (or a builtin, which is harmless)
                return _param((str(expr.id), ()))
            if isinstance(value := env[expr.id], _Closure):
                # a function that is not called here: count what it reads by itself
                return _flatten(self.apply(value, [_Reads()] * len(value.lambda_.params)))
            return value
        if not isinstance(expr, itir.FunCall):
            return _Reads()

        if cpm.is_applied_as_fieldop(expr):
            return self.visit_as_fieldop(expr, env)
        if isinstance(expr.fun, itir.SymRef) and isinstance(
            closure := env.get(expr.fun.id), _Closure
        ):
            return self.apply(closure, [self.visit(arg, env) for arg in expr.args])
        if (
            isinstance(expr.fun, itir.SymRef)
            and expr.fun.id not in env
            and expr.fun.id in self.function_definitions
        ):
            fundef = self.function_definitions[expr.fun.id]
            args = [self.visit(arg, env) for arg in expr.args]
            return self.visit(
                fundef.expr, {p.id: a for p, a in zip(fundef.params, args, strict=True)}
            )
        if cpm.is_call_to(expr, "make_tuple"):
            return tuple(self.visit(arg, env) for arg in expr.args)
        if cpm.is_call_to(expr, "tuple_get"):
            index = expr.args[0]
            assert isinstance(index, itir.Literal)
            return _tuple_get(int(index.value), self.visit(expr.args[1], env))
        if cpm.is_call_to(expr, ("if_", "concat_where")):
            # the condition is a scalar or a domain, it does not read fields
            return _merge(self.visit(expr.args[1], env), self.visit(expr.args[2], env))
        # any other builtin, e.g. `broadcast` or scalar arithmetic: pointwise in its arguments
        return _flatten(tuple(self.visit(arg, env) for arg in expr.args))

    def visit_as_fieldop(self, expr: itir.FunCall, env: Mapping[str, _Binding]) -> _Value:
        assert cpm.is_applied_as_fieldop(expr)
        stencil = expr.fun.args[0]
        args = [self.visit(arg, env) for arg in expr.args]
        if _is_identity_stencil(stencil) and len(args) == 1:
            return args[0]

        accesses: dict[_FieldRef, frozenset[_Access]] = {}
        for shifts, arg in zip(self.stencil_shifts(stencil, len(args)), args, strict=True):
            if not shifts:
                continue
            offsets = {self.offset(shift) for shift in shifts}
            for ref, arg_accesses in _flatten(arg).accesses.items():
                accesses[ref] = accesses.get(ref, frozenset()) | frozenset(
                    _Access(offset.then(a.offset), a.via)
                    for offset in offsets
                    for a in arg_accesses
                )
        return _Reads(accesses)


def _written(target: _Value, value: _Value) -> Iterator[tuple[_FieldRef, _Reads]]:
    """Pair each field written by a statement with the abstract value written to it."""
    if not isinstance(target, tuple) and isinstance(value, tuple):
        assert target.identity_of is not None
        target = tuple(_tuple_get(i, target) for i in range(len(value)))
    if isinstance(target, tuple):
        for i, el in enumerate(target):
            yield from _written(el, _tuple_get(i, value))
    else:
        assert target.identity_of is not None, "Targets must be (elements of) parameters."
        assert not isinstance(value, tuple)
        yield target.identity_of, value


def _offending_offsets(
    ref: _FieldRef, written_values: Sequence[_Reads], reads: Mapping[_FieldRef, Iterable[_Access]]
) -> set[_Offset]:
    if all(v.identity_of == ref for v in written_values):
        return set()  # written with its own value, i.e. memory does not change
    # Reading a selection of `ref` that is also what is written to `ref` is harmless at any
    # offset: where the selection reads `ref` it writes the same value, elsewhere it does not
    # read `ref`. This is the case for `a = f(...) if flag else a` followed by a shifted use.
    harmless = (
        {written_values[0].token}
        if len(written_values) == 1 and written_values[0].selection_of == ref
        else set()
    )
    return {
        access.offset
        for read_ref, accesses in reads.items()
        if _overlaps(read_ref, ref)
        for access in accesses
        if not access.offset.is_zero and access.via not in harmless
    }


@dataclasses.dataclass(frozen=True)
class InOutFieldReadWithOffset:
    """A field written by a statement that the same statement also reads at a non-zero offset."""

    #: Name of the program parameter.
    name: str
    #: Path of the field into the parameter if the parameter is a tuple, else `()`.
    path: tuple[int, ...]
    #: Human-readable descriptions of the offsets at which the field is read.
    offsets: tuple[str, ...]

    @property
    def field(self) -> str:
        return self.name + "".join(f"[{i}]" for i in self.path)


def find_inout_fields_read_with_offset(
    expr: itir.Expr,
    target: itir.Expr,
    *,
    function_definitions: Sequence[itir.FunctionDefinition] = (),
    offset_types: Mapping[str, ts.OffsetType],
) -> list[InOutFieldReadWithOffset]:
    """
    Find the fields written by the statement `target ← expr` that it also reads at an offset.

    A statement may read a field it also writes only at the point being written, see ADR 18,
    rule 3: executing it in place must give the same result as computing it into a buffer first.
    This applies to all reads of the statement, including those contributing to another output
    of a multi-output statement, since backends may write one output before another output reads
    the field. Two kinds of writes are exempt, because they never change a value that is read:

    - a field written with its own value, e.g. returned untouched by the field operator;
    - a field written with a value that, at every point, is either the field's own value or does
      not depend on the field, as in `a = f(b) if flag else a`, if all reads of the field at an
      offset go through this value.

    The analysis knows nothing about domains, so it also reports reads and writes that touch
    disjoint points at runtime.

    Args:
        expr: The field-view expression of the statement, i.e. as lowered from the frontend,
            before inlining or fusion. Symbol references that are neither parameters of a
            function definition nor `let`-bound are considered parameters of the program.
        target: The statement's target, a parameter, an element of a tuple parameter, or a
            `make_tuple` thereof.
        function_definitions: The function definitions `expr` may call.
        offset_types: Type of each offset tag used in `expr`, to distinguish Cartesian offsets
            from connectivities. An unknown offset is treated as a connectivity, except at index
            0 where it is assumed to be Cartesian, i.e. pointwise.

    Returns:
        The written fields that are read at an offset, in order of appearance in `target`.

    Examples:
        >>> from gt4py.next.iterator.ir_utils import ir_makers as im
        >>> KDim = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
        >>> offset_types = {"Koff": ts.OffsetType(source=KDim, target=(KDim,))}
        >>> shifted = im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Koff", 1)("it"))))
        >>> find_inout_fields_read_with_offset(shifted("f"), im.ref("f"), offset_types=offset_types)
        [InOutFieldReadWithOffset(name='f', path=(), offsets=("shifted by +1 along 'K'",))]
        >>> find_inout_fields_read_with_offset(shifted("f"), im.ref("g"), offset_types=offset_types)
        []
    """
    tracer = _FieldReadTracer(
        function_definitions={str(f.id): f for f in function_definitions},
        offset_types=offset_types,
    )
    value = tracer.visit(expr, {})
    reads = _flatten(value).accesses

    written: dict[_FieldRef, list[_Reads]] = {}
    for ref, written_value in _written(tracer.visit(target, {}), value):
        written.setdefault(ref, []).append(written_value)

    return [
        InOutFieldReadWithOffset(name=ref[0], path=ref[1], offsets=tuple(sorted(map(str, offsets))))
        for ref, written_values in written.items()
        if (offsets := _offending_offsets(ref, written_values, reads))
    ]
