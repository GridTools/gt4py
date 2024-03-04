# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import typing
from collections import abc
from typing import Optional

import gt4py.eve as eve
import gt4py.next as gtx
from gt4py.next.common import Connectivity
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.global_tmps import FencilWithTemporaries
from gt4py.next.type_inference import Type, TypeVar, freshen, reindex_vars, unify


"""Constraint-based inference for the iterator IR."""

T = typing.TypeVar("T", bound="Type")

# list of nodes that have a type
TYPED_IR_NODES: typing.Final = (
    ir.Expr,
    ir.FunctionDefinition,
    ir.StencilClosure,
    ir.FencilDefinition,
    ir.Sym,
)


class UnsatisfiableConstraintsError(Exception):
    unsatisfiable_constraints: list[tuple[Type, Type]]

    def __init__(self, unsatisfiable_constraints: list[tuple[Type, Type]]):
        self.unsatisfiable_constraints = unsatisfiable_constraints
        msg = "Type inference failed: Can not satisfy constraints:"
        for lhs, rhs in unsatisfiable_constraints:
            msg += f"\n  {lhs} ≡ {rhs}"
        super().__init__(msg)


class EmptyTuple(Type):
    def __iter__(self) -> abc.Iterator[Type]:
        return
        yield

    def __len__(self) -> int:
        return 0


class Tuple(Type):
    """Tuple type with arbitrary number of elements."""

    front: Type
    others: Type

    @classmethod
    def from_elems(cls: typing.Type[T], *elems: Type) -> typing.Union[T, EmptyTuple]:
        tup: typing.Union[T, EmptyTuple] = EmptyTuple()
        for e in reversed(elems):
            tup = cls(front=e, others=tup)
        return tup

    def __iter__(self) -> abc.Iterator[Type]:
        yield self.front
        if not isinstance(self.others, (Tuple, EmptyTuple)):
            raise ValueError(f"Can not iterate over partially defined tuple '{self}'.")
        yield from self.others

    @property
    def has_known_length(self):
        return isinstance(self.others, EmptyTuple) or (
            isinstance(self.others, Tuple) and self.others.has_known_length
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


class FunctionType(Type):
    """Function type.

    Note: the type inference algorithm always infers a tuple-like type for
    `args`, even for single-argument functions.
    """

    args: Type = eve.field(default_factory=TypeVar.fresh)
    ret: Type = eve.field(default_factory=TypeVar.fresh)


class Location(Type):
    """Location type."""

    name: str


ANYWHERE = Location(name="ANYWHERE")


class Val(Type):
    """The main type for representing values and iterators.

    Each `Val` consists of the following three things:
    - A `kind` which is either `Value()`, `Iterator()`, or a variable
    - A `dtype` which is either a `Primitive` or a variable
    - A `size` which is either `Scalar()`, `Column()`, or a variable
    """

    kind: Type = eve.field(default_factory=TypeVar.fresh)
    dtype: Type = eve.field(default_factory=TypeVar.fresh)
    size: Type = eve.field(default_factory=TypeVar.fresh)
    current_loc: Type = ANYWHERE
    defined_loc: Type = ANYWHERE


class ValTuple(Type):
    """A tuple of `Val` where all items have the same `kind` and `size`, but different dtypes."""

    kind: Type = eve.field(default_factory=TypeVar.fresh)
    dtypes: Type = eve.field(default_factory=TypeVar.fresh)
    size: Type = eve.field(default_factory=TypeVar.fresh)
    current_loc: Type = eve.field(default_factory=TypeVar.fresh)
    defined_locs: Type = eve.field(default_factory=TypeVar.fresh)

    def __eq__(self, other: typing.Any) -> bool:
        if (
            isinstance(self.dtypes, Tuple)
            and isinstance(self.defined_locs, Tuple)
            and isinstance(other, Tuple)
        ):
            dtypes: Type = self.dtypes
            defined_locs: Type = self.defined_locs
            elems: Type = other
            while (
                isinstance(dtypes, Tuple)
                and isinstance(defined_locs, Tuple)
                and isinstance(elems, Tuple)
                and Val(
                    kind=self.kind,
                    dtype=dtypes.front,
                    size=self.size,
                    current_loc=self.current_loc,
                    defined_loc=defined_locs.front,
                )
                == elems.front
            ):
                dtypes = dtypes.others
                defined_locs = defined_locs.others
                elems = elems.others
            return dtypes == defined_locs == elems == EmptyTuple()

        return (
            isinstance(other, ValTuple)
            and self.kind == other.kind
            and self.dtypes == other.dtypes
            and self.size == other.size
            and self.current_loc == other.current_loc
            and self.defined_locs == other.defined_locs
        )

    def handle_constraint(
        self, other: Type, add_constraint: abc.Callable[[Type, Type], None]
    ) -> bool:
        if isinstance(other, Tuple):
            dtypes = [TypeVar.fresh() for _ in other]
            defined_locs = [TypeVar.fresh() for _ in other]
            expanded = [
                Val(
                    kind=self.kind,
                    dtype=dtype,
                    size=self.size,
                    current_loc=self.current_loc,
                    defined_loc=defined_loc,
                )
                for dtype, defined_loc in zip(dtypes, defined_locs)
            ]
            add_constraint(self.dtypes, Tuple.from_elems(*dtypes))
            add_constraint(self.defined_locs, Tuple.from_elems(*defined_locs))
            add_constraint(Tuple.from_elems(*expanded), other)
            return True
        if isinstance(other, EmptyTuple):
            add_constraint(self.dtypes, EmptyTuple())
            add_constraint(self.defined_locs, EmptyTuple())
            return True
        return False


class ValListTuple(Type):
    """
    A tuple of `Val` that contains `List`s.

    All items have:
      - the same `kind` and `size`;
      - `dtype` is `List` with different `list_dtypes`, but same `max_length`, and `has_skip_values`.
    """

    kind: Type = eve.field(default_factory=TypeVar.fresh)
    list_dtypes: Type = eve.field(default_factory=TypeVar.fresh)
    max_length: Type = eve.field(default_factory=TypeVar.fresh)
    has_skip_values: Type = eve.field(default_factory=TypeVar.fresh)
    size: Type = eve.field(default_factory=TypeVar.fresh)

    def __eq__(self, other: typing.Any) -> bool:
        if isinstance(self.list_dtypes, Tuple) and isinstance(other, Tuple):
            list_dtypes: Type = self.list_dtypes
            elems: Type = other
            while (
                isinstance(list_dtypes, Tuple)
                and isinstance(elems, Tuple)
                and Val(
                    kind=self.kind,
                    dtype=List(
                        dtype=list_dtypes.front,
                        max_length=self.max_length,
                        has_skip_values=self.has_skip_values,
                    ),
                    size=self.size,
                )
                == elems.front
            ):
                list_dtypes = list_dtypes.others
                elems = elems.others
            return list_dtypes == elems == EmptyTuple()

        return (
            isinstance(other, ValListTuple)
            and self.kind == other.kind
            and self.list_dtypes == other.list_dtypes
            and self.max_length == other.max_length
            and self.has_skip_values == other.has_skip_values
            and self.size == other.size
        )

    def handle_constraint(
        self, other: Type, add_constraint: abc.Callable[[Type, Type], None]
    ) -> bool:
        if isinstance(other, Tuple):
            list_dtypes = [TypeVar.fresh() for _ in other]
            expanded = [
                Val(
                    kind=self.kind,
                    dtype=List(
                        dtype=dtype,
                        max_length=self.max_length,
                        has_skip_values=self.has_skip_values,
                    ),
                    size=self.size,
                )
                for dtype in list_dtypes
            ]
            add_constraint(self.list_dtypes, Tuple.from_elems(*list_dtypes))
            add_constraint(Tuple.from_elems(*expanded), other)
            return True
        if isinstance(other, EmptyTuple):
            add_constraint(self.list_dtypes, EmptyTuple())
            return True
        return False


class Column(Type):
    """Marker for column-sized values/iterators."""

    ...


class Scalar(Type):
    """Marker for scalar-sized values/iterators."""

    ...


class Primitive(Type):
    """Primitive type used in values/iterators."""

    name: str

    def handle_constraint(
        self, other: Type, add_constraint: abc.Callable[[Type, Type], None]
    ) -> bool:
        if not isinstance(other, Primitive):
            return False

        if self.name != other.name:
            raise TypeError(
                f"Can not satisfy constraint on primitive types: '{self.name}' ≡ '{other.name}'."
            )
        return True


class UnionPrimitive(Type):
    """Union of primitive types."""

    names: tuple[str, ...]

    def handle_constraint(
        self, other: Type, add_constraint: abc.Callable[[Type, Type], None]
    ) -> bool:
        if isinstance(other, UnionPrimitive):
            raise AssertionError("'UnionPrimitive' may only appear on one side of a constraint.")
        if not isinstance(other, Primitive):
            return False

        return other.name in self.names


class Value(Type):
    """Marker for values."""

    ...


class Iterator(Type):
    """Marker for iterators."""

    ...


class Length(Type):
    length: int


class BoolType(Type):
    value: bool


class List(Type):
    dtype: Type = eve.field(default_factory=TypeVar.fresh)
    max_length: Type = eve.field(default_factory=TypeVar.fresh)
    has_skip_values: Type = eve.field(default_factory=TypeVar.fresh)


class Closure(Type):
    """Stencil closure type."""

    output: Type
    inputs: Type


class FunctionDefinitionType(Type):
    """Function definition type."""

    name: str
    fun: FunctionType


class FencilDefinitionType(Type):
    """Fencil definition type."""

    name: str
    fundefs: Type
    params: Type


class LetPolymorphic(Type):
    """
    Wrapper for let-polymorphic types.

    Used for fencil-level function definitions.
    """

    dtype: Type


def _default_constraints():
    return {
        (FLOAT_DTYPE, UnionPrimitive(names=("float32", "float64"))),
        (INT_DTYPE, UnionPrimitive(names=("int32", "int64"))),
    }


BOOL_DTYPE = Primitive(name="bool")
INT_DTYPE = TypeVar.fresh()
FLOAT_DTYPE = TypeVar.fresh()
AXIS_DTYPE = Primitive(name="axis")
NAMED_RANGE_DTYPE = Primitive(name="named_range")
DOMAIN_DTYPE = Primitive(name="domain")
OFFSET_TAG_DTYPE = Primitive(name="offset_tag")

# Some helpers to define the builtins' types
T0 = TypeVar.fresh()
T1 = TypeVar.fresh()
T2 = TypeVar.fresh()
T3 = TypeVar.fresh()
T4 = TypeVar.fresh()
T5 = TypeVar.fresh()
Val_T0_T1 = Val(kind=Value(), dtype=T0, size=T1)
Val_T0_Scalar = Val(kind=Value(), dtype=T0, size=Scalar())
Val_BOOL_T1 = Val(kind=Value(), dtype=BOOL_DTYPE, size=T1)

BUILTIN_CATEGORY_MAPPING = (
    (
        ir.UNARY_MATH_FP_BUILTINS,
        FunctionType(
            args=Tuple.from_elems(Val(kind=Value(), dtype=FLOAT_DTYPE, size=T0)),
            ret=Val(kind=Value(), dtype=FLOAT_DTYPE, size=T0),
        ),
    ),
    (
        ir.UNARY_MATH_NUMBER_BUILTINS,
        FunctionType(
            args=Tuple.from_elems(Val_T0_T1),
            ret=Val_T0_T1,
        ),
    ),
    (
        {"power"},
        FunctionType(
            args=Tuple.from_elems(Val_T0_T1, Val(kind=Value(), dtype=T2, size=T1)), ret=Val_T0_T1
        ),
    ),
    (
        ir.BINARY_MATH_NUMBER_BUILTINS,
        FunctionType(args=Tuple.from_elems(Val_T0_T1, Val_T0_T1), ret=Val_T0_T1),
    ),
    (
        ir.UNARY_MATH_FP_PREDICATE_BUILTINS,
        FunctionType(
            args=Tuple.from_elems(Val(kind=Value(), dtype=FLOAT_DTYPE, size=T0)),
            ret=Val(kind=Value(), dtype=BOOL_DTYPE, size=T0),
        ),
    ),
    (
        ir.BINARY_MATH_COMPARISON_BUILTINS,
        FunctionType(args=Tuple.from_elems(Val_T0_T1, Val_T0_T1), ret=Val_BOOL_T1),
    ),
    (
        ir.BINARY_LOGICAL_BUILTINS,
        FunctionType(args=Tuple.from_elems(Val_BOOL_T1, Val_BOOL_T1), ret=Val_BOOL_T1),
    ),
    (
        ir.UNARY_LOGICAL_BUILTINS,
        FunctionType(
            args=Tuple.from_elems(
                Val_BOOL_T1,
            ),
            ret=Val_BOOL_T1,
        ),
    ),
)

BUILTIN_TYPES: dict[str, Type] = {
    **{builtin: type_ for category, type_ in BUILTIN_CATEGORY_MAPPING for builtin in category},
    "deref": FunctionType(
        args=Tuple.from_elems(
            Val(kind=Iterator(), dtype=T0, size=T1, current_loc=T2, defined_loc=T2)
        ),
        ret=Val_T0_T1,
    ),
    "can_deref": FunctionType(
        args=Tuple.from_elems(
            Val(kind=Iterator(), dtype=T0, size=T1, current_loc=T2, defined_loc=T3)
        ),
        ret=Val_BOOL_T1,
    ),
    "if_": FunctionType(
        args=Tuple.from_elems(Val_BOOL_T1, T2, T2),
        ret=T2,
    ),
    "lift": FunctionType(
        args=Tuple.from_elems(
            FunctionType(
                args=ValTuple(kind=Iterator(), dtypes=T2, size=T1, current_loc=T3, defined_locs=T4),
                ret=Val_T0_T1,
            )
        ),
        ret=FunctionType(
            args=ValTuple(kind=Iterator(), dtypes=T2, size=T1, current_loc=T5, defined_locs=T4),
            ret=Val(kind=Iterator(), dtype=T0, size=T1, current_loc=T5, defined_loc=T3),
        ),
    ),
    "map_": FunctionType(
        args=Tuple.from_elems(
            FunctionType(
                args=ValTuple(kind=Value(), dtypes=T2, size=T1),
                ret=Val_T0_T1,
            ),
        ),
        ret=FunctionType(
            args=ValListTuple(kind=Value(), list_dtypes=T2, size=T1),
            ret=Val(kind=Value(), dtype=List(dtype=T0, max_length=T4, has_skip_values=T5), size=T1),
        ),
    ),
    "reduce": FunctionType(
        args=Tuple.from_elems(
            FunctionType(
                args=Tuple(front=Val_T0_T1, others=ValTuple(kind=Value(), dtypes=T2, size=T1)),
                ret=Val_T0_T1,
            ),
            Val_T0_T1,
        ),
        ret=FunctionType(
            args=ValListTuple(
                kind=Value(), list_dtypes=T2, max_length=T4, has_skip_values=T5, size=T1
            ),
            ret=Val_T0_T1,
        ),
    ),
    "make_const_list": FunctionType(
        args=Tuple.from_elems(Val_T0_T1),
        ret=Val(kind=Value(), dtype=List(dtype=T0, max_length=T2, has_skip_values=T3), size=T1),
    ),
    "list_get": FunctionType(
        args=Tuple.from_elems(
            Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
            Val(kind=Value(), dtype=List(dtype=T0, max_length=T2, has_skip_values=T3), size=T1),
        ),
        ret=Val_T0_T1,
    ),
    "scan": FunctionType(
        args=Tuple.from_elems(
            FunctionType(
                args=Tuple(
                    front=Val_T0_Scalar,
                    others=ValTuple(
                        kind=Iterator(), dtypes=T2, size=Scalar(), current_loc=T3, defined_locs=T4
                    ),
                ),
                ret=Val_T0_Scalar,
            ),
            Val(kind=Value(), dtype=BOOL_DTYPE, size=Scalar()),
            Val_T0_Scalar,
        ),
        ret=FunctionType(
            args=ValTuple(
                kind=Iterator(), dtypes=T2, size=Column(), current_loc=T3, defined_locs=T4
            ),
            ret=Val(kind=Value(), dtype=T0, size=Column()),
        ),
    ),
    "named_range": FunctionType(
        args=Tuple.from_elems(
            Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar()),
            Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
            Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
        ),
        ret=Val(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar()),
    ),
}


del T0, T1, T2, T3, T4, T5, Val_T0_T1, Val_T0_Scalar, Val_BOOL_T1


def _infer_shift_location_types(shift_args, offset_provider, constraints):
    current_loc_in = TypeVar.fresh()
    if offset_provider:
        current_loc_out = current_loc_in
        for arg in shift_args:
            if not isinstance(arg, ir.OffsetLiteral):
                # probably some dynamically computed offset, thus we assume it's a number not an axis and just ignore it (see comment below)
                continue
            offset = arg.value
            if isinstance(offset, int):
                continue  # ignore 'application' of (partial) shifts
            else:
                assert isinstance(offset, str)
                axis = offset_provider[offset]
                if isinstance(axis, gtx.Dimension):
                    continue  # Cartesian shifts don't change the location type
                elif isinstance(axis, Connectivity):
                    assert (
                        axis.origin_axis.kind
                        == axis.neighbor_axis.kind
                        == gtx.DimensionKind.HORIZONTAL
                    )
                    constraints.add((current_loc_out, Location(name=axis.origin_axis.value)))
                    current_loc_out = Location(name=axis.neighbor_axis.value)
                else:
                    raise NotImplementedError()
    elif not shift_args:
        current_loc_out = current_loc_in
    else:
        current_loc_out = TypeVar.fresh()
    return current_loc_in, current_loc_out


@dataclasses.dataclass
class _TypeInferrer(eve.traits.VisitorWithSymbolTableTrait, eve.NodeTranslator):
    """
    Visit the full iterator IR tree, convert nodes to respective types and generate constraints.

    Attributes:
        collected_types: Mapping from the (Python) id of a node to its type.
        constraints: Set of constraints, where a constraint is a pair of types that need to agree.
            See `unify` for more information.
    """

    offset_provider: Optional[dict[str, Connectivity | gtx.Dimension]]
    collected_types: dict[int, Type] = dataclasses.field(default_factory=dict)
    constraints: set[tuple[Type, Type]] = dataclasses.field(default_factory=_default_constraints)

    def visit(self, node, **kwargs) -> typing.Any:
        result = super().visit(node, **kwargs)
        if isinstance(node, TYPED_IR_NODES):
            assert isinstance(result, Type)
            if not (
                id(node) not in self.collected_types or self.collected_types[id(node)] == result
            ):
                # using the same node in multiple places is fine as long as the type is the same
                # for all occurences
                self.constraints.add((result, self.collected_types[id(node)]))
            self.collected_types[id(node)] = result

        return result

    def visit_Sym(self, node: ir.Sym, **kwargs) -> Type:
        result = TypeVar.fresh()
        if node.kind:
            kind = {"Iterator": Iterator(), "Value": Value()}[node.kind]
            self.constraints.add((
                Val(kind=kind, current_loc=TypeVar.fresh(), defined_loc=TypeVar.fresh()),
                result,
            ))
        if node.dtype:
            assert node.dtype is not None
            dtype: Primitive | List = Primitive(name=node.dtype[0])
            if node.dtype[1]:
                dtype = List(dtype=dtype)
            self.constraints.add((
                Val(
                    dtype=dtype,
                    current_loc=TypeVar.fresh(),
                    defined_loc=TypeVar.fresh(),
                ),
                result,
            ))
        return result

    def visit_SymRef(self, node: ir.SymRef, *, symtable, **kwargs) -> Type:
        if node.id in ir.BUILTINS:
            if node.id in BUILTIN_TYPES:
                return freshen(BUILTIN_TYPES[node.id])
            elif node.id in ir.GRAMMAR_BUILTINS:
                raise TypeError(
                    f"Builtin '{node.id}' is only allowed as applied/called function by the type "
                    "inference."
                )
            elif node.id in ir.TYPEBUILTINS:
                # TODO(tehrengruber): Implement propagating types of values referring to types, e.g.
                #   >>> my_int = int64
                #   ... cast_(expr, my_int)
                #  One way to support this is by introducing a "type of type" similar to pythons
                #  `typing.Type`.
                raise NotImplementedError(
                    f"Type builtin '{node.id}' is only supported as literal argument by the "
                    "type inference."
                )
            else:
                raise NotImplementedError(f"Missing type definition for builtin '{node.id}'.")
        elif node.id in symtable:
            sym_decl = symtable[node.id]
            assert isinstance(sym_decl, TYPED_IR_NODES)
            res = self.collected_types[id(sym_decl)]
            if isinstance(res, LetPolymorphic):
                return freshen(res.dtype)
            return res

        return TypeVar.fresh()

    def visit_Literal(self, node: ir.Literal, **kwargs) -> Val:
        return Val(kind=Value(), dtype=Primitive(name=node.type))

    def visit_AxisLiteral(self, node: ir.AxisLiteral, **kwargs) -> Val:
        return Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar())

    def visit_OffsetLiteral(self, node: ir.OffsetLiteral, **kwargs) -> TypeVar:
        return TypeVar.fresh()

    def visit_Lambda(
        self,
        node: ir.Lambda,
        **kwargs,
    ) -> FunctionType:
        ptypes = {p.id: self.visit(p, **kwargs) for p in node.params}
        ret = self.visit(node.expr, **kwargs)
        return FunctionType(args=Tuple.from_elems(*(ptypes[p.id] for p in node.params)), ret=ret)

    def _visit_make_tuple(self, node: ir.FunCall, **kwargs) -> Type:
        # Calls to `make_tuple` are handled as being part of the grammar, not as function calls.
        argtypes = self.visit(node.args, **kwargs)
        kind = (
            TypeVar.fresh()
        )  # `kind == Iterator()` means zipping iterators into an iterator of tuples
        size = TypeVar.fresh()
        dtype = Tuple.from_elems(*(TypeVar.fresh() for _ in argtypes))
        for d, a in zip(dtype, argtypes):
            self.constraints.add((Val(kind=kind, dtype=d, size=size), a))
        return Val(kind=kind, dtype=dtype, size=size)

    def _visit_tuple_get(self, node: ir.FunCall, **kwargs) -> Type:
        # Calls to `tuple_get` are handled as being part of the grammar, not as function calls.
        if len(node.args) != 2:
            raise TypeError("'tuple_get' requires exactly two arguments.")
        if (
            not isinstance(node.args[0], ir.Literal)
            or node.args[0].type != ir.INTEGER_INDEX_BUILTIN
        ):
            raise TypeError(
                f"The first argument to 'tuple_get' must be a literal of type '{ir.INTEGER_INDEX_BUILTIN}'."
            )
        self.visit(node.args[0], **kwargs)  # visit index so that its type is collected
        idx = int(node.args[0].value)
        tup = self.visit(node.args[1], **kwargs)
        kind = TypeVar.fresh()  # `kind == Iterator()` means splitting an iterator of tuples
        elem = TypeVar.fresh()
        size = TypeVar.fresh()

        dtype = Tuple(front=elem, others=TypeVar.fresh())
        for _ in range(idx):
            dtype = Tuple(front=TypeVar.fresh(), others=dtype)

        val = Val(
            kind=kind,
            dtype=dtype,
            size=size,
        )
        self.constraints.add((tup, val))
        return Val(kind=kind, dtype=elem, size=size)

    def _visit_neighbors(self, node: ir.FunCall, **kwargs) -> Type:
        if len(node.args) != 2:
            raise TypeError("'neighbors' requires exactly two arguments.")
        if not (isinstance(node.args[0], ir.OffsetLiteral) and isinstance(node.args[0].value, str)):
            raise TypeError("The first argument to 'neighbors' must be an 'OffsetLiteral' tag.")

        # Visit arguments such that their type is also inferred
        self.visit(node.args, **kwargs)

        max_length: Type = TypeVar.fresh()
        has_skip_values: Type = TypeVar.fresh()
        if self.offset_provider:
            connectivity = self.offset_provider[node.args[0].value]
            assert isinstance(connectivity, Connectivity)
            max_length = Length(length=connectivity.max_neighbors)
            has_skip_values = BoolType(value=connectivity.has_skip_values)
        current_loc_in, current_loc_out = _infer_shift_location_types(
            [node.args[0]], self.offset_provider, self.constraints
        )
        dtype_ = TypeVar.fresh()
        size = TypeVar.fresh()
        it = self.visit(node.args[1], **kwargs)
        self.constraints.add((
            it,
            Val(
                kind=Iterator(),
                dtype=dtype_,
                size=size,
                current_loc=current_loc_in,
                defined_loc=current_loc_out,
            ),
        ))
        lst = List(
            dtype=dtype_,
            max_length=max_length,
            has_skip_values=has_skip_values,
        )
        return Val(kind=Value(), dtype=lst, size=size)

    def _visit_cast_(self, node: ir.FunCall, **kwargs) -> Type:
        if len(node.args) != 2:
            raise TypeError("'cast_' requires exactly two arguments.")
        val_arg_type = self.visit(node.args[0], **kwargs)
        type_arg = node.args[1]
        if not isinstance(type_arg, ir.SymRef) or type_arg.id not in ir.TYPEBUILTINS:
            raise TypeError("The second argument to 'cast_' must be a type literal.")

        size = TypeVar.fresh()

        self.constraints.add((
            val_arg_type,
            Val(
                kind=Value(),
                dtype=TypeVar.fresh(),
                size=size,
            ),
        ))

        return Val(
            kind=Value(),
            dtype=Primitive(name=type_arg.id),
            size=size,
        )

    def _visit_shift(self, node: ir.FunCall, **kwargs) -> Type:
        # Calls to shift are handled as being part of the grammar, not
        # as function calls, as the type depends on the offset provider.

        # Visit arguments such that their type is also inferred (particularly important for
        # dynamic offsets)
        self.visit(node.args)

        current_loc_in, current_loc_out = _infer_shift_location_types(
            node.args, self.offset_provider, self.constraints
        )
        defined_loc = TypeVar.fresh()
        dtype_ = TypeVar.fresh()
        size = TypeVar.fresh()
        return FunctionType(
            args=Tuple.from_elems(
                Val(
                    kind=Iterator(),
                    dtype=dtype_,
                    size=size,
                    current_loc=current_loc_in,
                    defined_loc=defined_loc,
                ),
            ),
            ret=Val(
                kind=Iterator(),
                dtype=dtype_,
                size=size,
                current_loc=current_loc_out,
                defined_loc=defined_loc,
            ),
        )

    def _visit_domain(self, node: ir.FunCall, **kwargs) -> Type:
        for arg in node.args:
            self.constraints.add((
                Val(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar()),
                self.visit(arg, **kwargs),
            ))
        return Val(kind=Value(), dtype=DOMAIN_DTYPE, size=Scalar())

    def _visit_cartesian_domain(self, node: ir.FunCall, **kwargs) -> Type:
        return self._visit_domain(node, **kwargs)

    def _visit_unstructured_domain(self, node: ir.FunCall, **kwargs) -> Type:
        return self._visit_domain(node, **kwargs)

    def visit_FunCall(
        self,
        node: ir.FunCall,
        **kwargs,
    ) -> Type:
        if isinstance(node.fun, ir.SymRef) and node.fun.id in ir.GRAMMAR_BUILTINS:
            # builtins that are treated as part of the grammar are handled in `_visit_<builtin_name>`
            return getattr(self, f"_visit_{node.fun.id}")(node, **kwargs)
        elif isinstance(node.fun, ir.SymRef) and node.fun.id in ir.TYPEBUILTINS:
            return Val(kind=Value(), dtype=Primitive(name=node.fun.id))

        fun = self.visit(node.fun, **kwargs)
        args = Tuple.from_elems(*self.visit(node.args, **kwargs))
        ret = TypeVar.fresh()
        self.constraints.add((fun, FunctionType(args=args, ret=ret)))
        return ret

    def visit_FunctionDefinition(
        self,
        node: ir.FunctionDefinition,
        **kwargs,
    ) -> LetPolymorphic:
        fun = ir.Lambda(params=node.params, expr=node.expr)

        # Since functions defined in a function definition are let-polymorphic we don't want
        # their parameters to inherit the constraints of the arguments in a call to them. A simple
        # way to do this is to run the type inference on the function itself and reindex its type
        # vars when referencing the function, i.e. in a `SymRef`.
        collected_types = infer_all(fun, offset_provider=self.offset_provider, reindex=False)
        fun_type = LetPolymorphic(dtype=collected_types.pop(id(fun)))
        assert not set(self.collected_types.keys()) & set(collected_types.keys())
        self.collected_types = {**self.collected_types, **collected_types}

        return fun_type

    def visit_StencilClosure(
        self,
        node: ir.StencilClosure,
        **kwargs,
    ) -> Closure:
        domain = self.visit(node.domain, **kwargs)
        stencil = self.visit(node.stencil, **kwargs)
        output = self.visit(node.output, **kwargs)
        output_dtype = TypeVar.fresh()
        output_loc = TypeVar.fresh()
        self.constraints.add((
            domain,
            Val(kind=Value(), dtype=Primitive(name="domain"), size=Scalar()),
        ))
        self.constraints.add((
            output,
            Val(
                kind=Iterator(),
                dtype=output_dtype,
                size=Column(),
                defined_loc=output_loc,
            ),
        ))

        inputs: list[Type] = self.visit(node.inputs, **kwargs)
        stencil_params = []
        for input_ in inputs:
            stencil_param = Val(current_loc=output_loc, defined_loc=TypeVar.fresh())
            self.constraints.add((
                input_,
                Val(
                    kind=stencil_param.kind,
                    dtype=stencil_param.dtype,
                    size=stencil_param.size,
                    # closure input and stencil param differ in `current_loc`
                    current_loc=ANYWHERE,
                    # TODO(tehrengruber): Seems to break for scalars. Use `TypeVar.fresh()`?
                    defined_loc=stencil_param.defined_loc,
                ),
            ))
            stencil_params.append(stencil_param)

        self.constraints.add((
            stencil,
            FunctionType(
                args=Tuple.from_elems(*stencil_params),
                ret=Val(kind=Value(), dtype=output_dtype, size=Column()),
            ),
        ))
        return Closure(output=output, inputs=Tuple.from_elems(*inputs))

    def visit_FencilWithTemporaries(self, node: FencilWithTemporaries, **kwargs):
        return self.visit(node.fencil, **kwargs)

    def visit_FencilDefinition(
        self,
        node: ir.FencilDefinition,
        **kwargs,
    ) -> FencilDefinitionType:
        ftypes = []
        # Note: functions have to be ordered according to Lisp/Scheme `let*`
        # statements; that is, functions can only reference other functions
        # that are defined before
        for fun_def in node.function_definitions:
            fun_type: LetPolymorphic = self.visit(fun_def, **kwargs)
            ftype = FunctionDefinitionType(name=fun_def.id, fun=fun_type.dtype)
            ftypes.append(ftype)

        params = [self.visit(p, **kwargs) for p in node.params]
        self.visit(node.closures, **kwargs)
        return FencilDefinitionType(
            name=str(node.id),
            fundefs=Tuple.from_elems(*ftypes),
            params=Tuple.from_elems(*params),
        )


def _save_types_to_annex(node: ir.Node, types: dict[int, Type]) -> None:
    for child_node in node.pre_walk_values().if_isinstance(*TYPED_IR_NODES):
        try:
            child_node.annex.type = types[id(child_node)]
        except KeyError as ex:
            if not (
                isinstance(child_node, ir.SymRef)
                and child_node.id in ir.GRAMMAR_BUILTINS | ir.TYPEBUILTINS
            ):
                raise AssertionError(
                    f"Expected a type to be inferred for node '{child_node}', but none was found."
                ) from ex


def infer_all(
    node: ir.Node,
    *,
    offset_provider: Optional[dict[str, Connectivity | gtx.Dimension]] = None,
    reindex: bool = True,
    save_to_annex=False,
) -> dict[int, Type]:
    """
    Infer the types of the child expressions of a given iterator IR expression.

    The result is a dictionary mapping the (Python) id of child nodes to their type.

    The `save_to_annex` flag should only be used as a last resort when the  return dictionary is
    not enough.
    """
    # Collect preliminary types of all nodes and constraints on them
    inferrer = _TypeInferrer(offset_provider=offset_provider)
    inferrer.visit(node)

    # Ensure dict order is pre-order of the tree
    collected_types = dict(reversed(inferrer.collected_types.items()))

    # Compute the most general type that satisfies all constraints
    unified_types, unsatisfiable_constraints = unify(
        list(collected_types.values()), inferrer.constraints
    )

    if reindex:
        unified_types, unsatisfiable_constraints = reindex_vars((
            unified_types,
            unsatisfiable_constraints,
        ))

    result = {
        id_: unified_type
        for id_, unified_type in zip(collected_types.keys(), unified_types, strict=True)
    }

    if save_to_annex:
        _save_types_to_annex(node, result)

    if unsatisfiable_constraints:
        raise UnsatisfiableConstraintsError(unsatisfiable_constraints)

    return result


def infer(
    expr: ir.Node,
    offset_provider: typing.Optional[dict[str, typing.Any]] = None,
    save_to_annex: bool = False,
) -> Type:
    """Infer the type of the given iterator IR expression."""
    inferred_types = infer_all(expr, offset_provider=offset_provider, save_to_annex=save_to_annex)
    return inferred_types[id(expr)]


class PrettyPrinter(eve.NodeTranslator):
    """Pretty-printer for type expressions."""

    @staticmethod
    def _subscript(i: int) -> str:
        return "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(i))

    @staticmethod
    def _superscript(i: int) -> str:
        return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(i))

    def _fmt_size(self, size: Type) -> str:
        if size == Column():
            return "ᶜ"
        if size == Scalar():
            return "ˢ"
        assert isinstance(size, TypeVar)
        return self._superscript(size.idx)

    def _fmt_dtype(
        self,
        kind: Type,
        dtype_str: str,
        current_loc: typing.Optional[str] = None,
        defined_loc: typing.Optional[str] = None,
    ) -> str:
        if kind == Value():
            return dtype_str
        if kind == Iterator():
            if current_loc == defined_loc == "ANYWHERE" or current_loc is defined_loc is None:
                locs = ""
            else:
                assert isinstance(current_loc, str) and isinstance(defined_loc, str)
                locs = current_loc + ", " + defined_loc + ", "
            return "It[" + locs + dtype_str + "]"
        assert isinstance(kind, TypeVar)
        return "ItOrVal" + self._subscript(kind.idx) + "[" + dtype_str + "]"

    def visit_EmptyTuple(self, node: EmptyTuple) -> str:
        return "()"

    def visit_Tuple(self, node: Tuple) -> str:
        s = "(" + self.visit(node.front)
        while isinstance(node.others, Tuple):
            node = node.others
            s += ", " + self.visit(node.front)
        s += ")"
        if not isinstance(node.others, EmptyTuple):
            s += ":" + self.visit(node.others)
        return s

    def visit_Location(self, node: Location):
        return node.name

    def visit_FunctionType(self, node: FunctionType) -> str:
        return self.visit(node.args) + " → " + self.visit(node.ret)

    def visit_Val(self, node: Val) -> str:
        return self._fmt_dtype(
            node.kind,
            self.visit(node.dtype) + self._fmt_size(node.size),
            self.visit(node.current_loc),
            self.visit(node.defined_loc),
        )

    def visit_Primitive(self, node: Primitive) -> str:
        return node.name

    def visit_List(self, node: List) -> str:
        return f"L[{self.visit(node.dtype)}, {self.visit(node.max_length)}, {self.visit(node.has_skip_values)}]"

    def visit_FunctionDefinitionType(self, node: FunctionDefinitionType) -> str:
        return node.name + " :: " + self.visit(node.fun)

    def visit_Closure(self, node: Closure) -> str:
        return self.visit(node.inputs) + " ⇒ " + self.visit(node.output)

    def visit_FencilDefinitionType(self, node: FencilDefinitionType) -> str:
        assert isinstance(node.fundefs, (Tuple, EmptyTuple))
        assert isinstance(node.params, (Tuple, EmptyTuple))
        return (
            "{"
            + "".join(self.visit(f) + ", " for f in node.fundefs)
            + node.name
            + "("
            + ", ".join(self.visit(p) for p in node.params)
            + ")}"
        )

    def visit_ValTuple(self, node: ValTuple) -> str:
        if isinstance(node.dtypes, TypeVar):
            assert isinstance(node.defined_locs, TypeVar)
            return (
                "("
                + self._fmt_dtype(
                    node.kind,
                    "T" + self._fmt_size(node.size),
                    self.visit(node.current_loc),
                    "…" + self._subscript(node.defined_locs.idx),
                )
                + ", …)"
                + self._subscript(node.dtypes.idx)
            )
        assert isinstance(node.dtypes, (Tuple, EmptyTuple))
        if isinstance(node.defined_locs, (Tuple, EmptyTuple)):
            defined_locs = node.defined_locs
        else:
            defined_locs = Tuple.from_elems(*(Location(name="_") for _ in node.dtypes))
        return (
            "("
            + ", ".join(
                self.visit(
                    Val(
                        kind=node.kind,
                        dtype=dtype,
                        size=node.size,
                        current_loc=node.current_loc,
                        defined_loc=defined_loc,
                    )
                )
                for dtype, defined_loc in zip(node.dtypes, defined_locs)
            )
            + ")"
        )

    def visit_ValListTuple(self, node: ValListTuple) -> str:
        if isinstance(node.list_dtypes, TypeVar):
            return f"(L[…{self._subscript(node.list_dtypes.idx)}, {self.visit(node.max_length)}, {self.visit(node.has_skip_values)}]{self._fmt_size(node.size)}, …)"
        assert isinstance(node.list_dtypes, (Tuple, EmptyTuple))
        return (
            "("
            + ", ".join(
                self.visit(
                    Val(
                        kind=Value(),
                        dtype=List(
                            dtype=dtype,
                            max_length=node.max_length,
                            has_skip_values=node.has_skip_values,
                        ),
                        size=node.size,
                    )
                )
                for dtype in node.list_dtypes
            )
            + ")"
        )

    def visit_TypeVar(self, node: TypeVar) -> str:
        return "T" + self._subscript(node.idx)

    def visit_Type(self, node: Type) -> str:
        return (
            node.__class__.__name__
            + "("
            + ", ".join(f"{k}={v}" for k, v in node.iter_children_items())
            + ")"
        )


pformat = PrettyPrinter().visit


def pprint(x: Type) -> None:
    print(pformat(x))
