import typing
from collections.abc import Callable

import eve
from eve.utils import noninstantiable
from functional.iterator import ir


"""Constraint-based inference for the iterator IR.

Based on the classical constraint-based two-pass type consisting of the following passes:
    1. Constraint collection
    2. Type unification
"""

V = typing.TypeVar("V", bound="TypeVar")
T = typing.TypeVar("T", bound="Type")


@noninstantiable
class Type(eve.Node, unsafe_hash=True):  # type: ignore[call-arg]
    """Base class for all types.

    The initial type constraint collection pass treats all instances of Type as hashable frozen
    nodes, that is, no in-place modification is used.

    In the type unification phase however, in-place modifications are used for efficient
    renaming/node replacements and special care is taken to handle hash values that change due to
    those modifications.
    """

    def __str__(self) -> str:
        return pformat(self)


class TypeVar(Type):
    """Basic type variable.

    Also used as baseclass for specialized type variables.
    """

    idx: int

    _counter: typing.ClassVar[int] = 0

    @staticmethod
    def fresh_index() -> int:
        TypeVar._counter += 1
        return TypeVar._counter

    @classmethod
    def fresh(cls: type[V], **kwargs: typing.Any) -> V:
        """Create a type variable with a previously unused index."""
        return cls(idx=cls.fresh_index(), **kwargs)


class Tuple(Type):
    """Tuple type with arbitrary number of elements."""

    elems: tuple[Type, ...]


class PartialTupleVar(TypeVar):
    """Type variable representing a partially defined tuple.

    `elem_indices` are the indices of the known elements; `elem_values` are the values of the known
    elements. Both tuples need to have the same length.

    E.g., if `elem_indices` is `(0, 2)` and `elem_values` are two type variables `(T₀, T₁)`, then
    the form of the tuple type is `(T₀, ?, T₁, …)`. That is, the types of the elements at indices 0
    and 2 are known and T₀, respectively T₁. But the total number of elements and types of other
    elements are unknown.
    """

    elem_indices: tuple[int, ...]
    elem_values: tuple[Type, ...]


class PrefixTuple(Type):
    """A tuple type consisting of a prefix (first element) and other elements.

    This type is similar to the cons operator for tuple types. Not that this type can be replaced by
    an instance of `Tuple` as soon as the type of `others` is a `Tuple`.
    """

    prefix: Type
    others: Type


class FunctionType(Type):
    """Function type.

    Note: the type inference algorithm always infers a tuple-like type for
    `args`, even for single-argument functions.
    """

    args: Type = eve.field(default_factory=TypeVar.fresh)
    ret: Type = eve.field(default_factory=TypeVar.fresh)


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


class ValTuple(Type):
    """A tuple of `Val` where all items have the same `kind` and `size`, but different dtypes."""

    kind: Type = eve.field(default_factory=TypeVar.fresh)
    dtypes: Type = eve.field(default_factory=TypeVar.fresh)
    size: Type = eve.field(default_factory=TypeVar.fresh)


class UniformValTupleVar(TypeVar):
    """A tuple of `Val` with unknown length, but common `kind`, `size`, and `dtype` of all items."""

    kind: Type = eve.field(default_factory=TypeVar.fresh)
    dtype: Type = eve.field(default_factory=TypeVar.fresh)
    size: Type = eve.field(default_factory=TypeVar.fresh)


class Column(Type):
    """Marker for column-sized values/iterators."""

    ...


class Scalar(Type):
    """Marker for scalar-sized values/iterators."""

    ...


class Primitive(Type):
    """Primitive type used in values/iterators."""

    name: str


class Value(Type):
    """Marker for values."""

    ...


class Iterator(Type):
    """Marker for iterators."""

    ...


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
    fundefs: tuple[Type, ...]
    params: tuple[Type, ...]


class LetPolymorphic(Type):
    """Wrapper for let-polymorphic types.

    Used for fencil-level function definitions.
    """

    dtype: Type


BOOL_DTYPE = Primitive(name="bool")  # type: ignore [call-arg]
INT_DTYPE = Primitive(name="int")  # type: ignore [call-arg]
FLOAT_DTYPE = Primitive(name="float")  # type: ignore [call-arg]
AXIS_DTYPE = Primitive(name="axis")  # type: ignore [call-arg]
NAMED_RANGE_DTYPE = Primitive(name="named_range")  # type: ignore [call-arg]
DOMAIN_DTYPE = Primitive(name="domain")  # type: ignore [call-arg]


class _TypeVarReindexer(eve.NodeTranslator):
    """Reindex type variables in a type tree."""

    def __init__(self, indexer: Callable[[dict[int, int]], int]):
        super().__init__()
        self.indexer = indexer

    def visit_TypeVar(self, node: V, *, index_map: dict[int, int]) -> V:
        node = self.generic_visit(node, index_map=index_map)
        new_index = index_map.setdefault(node.idx, self.indexer(index_map))
        new_values = {
            typing.cast(str, k): (new_index if k == "idx" else v)
            for k, v in node.iter_children_items()
        }
        return node.__class__(**new_values)


def _freshen(dtype: T) -> T:
    """Re-instantiate `dtype` with fresh type variables."""

    def indexer(index_map: dict[int, int]) -> int:
        return TypeVar.fresh_index()

    index_map = dict[int, int]()
    return _TypeVarReindexer(indexer).visit(dtype, index_map=index_map)


def _builtin_type(builtin: str) -> Type:
    """Generate type definition for the given builtin function."""
    assert builtin in ir.BUILTINS
    if builtin == "deref":
        dtype = TypeVar.fresh()
        size = TypeVar.fresh()
        return FunctionType(
            args=Tuple(elems=(Val(kind=Iterator(), dtype=dtype, size=size),)),
            ret=Val(kind=Value(), dtype=dtype, size=size),
        )
    if builtin == "can_deref":
        dtype = TypeVar.fresh()
        size = TypeVar.fresh()
        return FunctionType(
            args=Tuple(elems=(Val(kind=Iterator(), dtype=dtype, size=size),)),
            ret=Val(kind=Value(), dtype=BOOL_DTYPE, size=size),
        )
    if builtin in ("plus", "minus", "multiplies", "divides"):
        v = Val(kind=Value())
        return FunctionType(args=Tuple(elems=(v, v)), ret=v)
    if builtin in ("eq", "less", "greater"):
        v = Val(kind=Value())
        ret: Type = Val(kind=Value(), dtype=BOOL_DTYPE, size=v.size)
        return FunctionType(args=Tuple(elems=(v, v)), ret=ret)
    if builtin == "not_":
        v = Val(kind=Value(), dtype=BOOL_DTYPE)
        return FunctionType(args=Tuple(elems=(v,)), ret=v)
    if builtin in ("and_", "or_"):
        v = Val(kind=Value(), dtype=BOOL_DTYPE)
        return FunctionType(args=Tuple(elems=(v, v)), ret=v)
    if builtin == "if_":
        v = Val(kind=Value())
        c = Val(kind=Value(), dtype=BOOL_DTYPE, size=v.size)
        return FunctionType(args=Tuple(elems=(c, v, v)), ret=v)
    if builtin == "lift":
        size = TypeVar.fresh()
        args: Type = ValTuple(kind=Iterator(), dtypes=TypeVar.fresh(), size=size)
        dtype = TypeVar.fresh()
        stencil_ret = Val(kind=Value(), dtype=dtype, size=size)
        lifted_ret = Val(kind=Iterator(), dtype=dtype, size=size)
        return FunctionType(
            args=Tuple(elems=(FunctionType(args=args, ret=stencil_ret),)),
            ret=FunctionType(args=args, ret=lifted_ret),
        )
    if builtin == "reduce":
        dtypes = TypeVar.fresh()
        size = TypeVar.fresh()
        acc = Val(kind=Value(), dtype=TypeVar.fresh(), size=size)
        f_args = PrefixTuple(prefix=acc, others=ValTuple(kind=Value(), dtypes=dtypes, size=size))
        ret_args = ValTuple(kind=Iterator(), dtypes=dtypes, size=size)
        f = FunctionType(args=f_args, ret=acc)
        ret = FunctionType(args=ret_args, ret=acc)
        return FunctionType(args=Tuple(elems=(f, acc)), ret=ret)
    if builtin == "scan":
        dtypes = TypeVar.fresh()
        fwd = Val(kind=Value(), dtype=BOOL_DTYPE, size=Scalar())
        acc = Val(kind=Value(), dtype=TypeVar.fresh(), size=Scalar())
        f_args = PrefixTuple(
            prefix=acc, others=ValTuple(kind=Iterator(), dtypes=dtypes, size=Scalar())
        )
        ret_args = ValTuple(kind=Iterator(), dtypes=dtypes, size=Column())
        f = FunctionType(args=f_args, ret=acc)
        ret = FunctionType(args=ret_args, ret=Val(kind=Value(), dtype=acc.dtype, size=Column()))
        return FunctionType(args=Tuple(elems=(f, fwd, acc)), ret=ret)
    if builtin == "domain":
        args = UniformValTupleVar.fresh(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar())
        ret = Val(kind=Value(), dtype=DOMAIN_DTYPE, size=Scalar())
        return FunctionType(args=args, ret=ret)
    if builtin == "named_range":
        args = Tuple(
            elems=(
                Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar()),
                Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
                Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
            )
        )
        ret = Val(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar())
        return FunctionType(args=args, ret=ret)
    if builtin in ("make_tuple", "tuple_get", "shift"):
        raise TypeError(
            f"Builtin '{builtin}' is only supported as applied/called function by the type checker"
        )
    raise NotImplementedError(f"Missing type definition of builtin '{builtin}'")


class _TypeInferrer(eve.NodeTranslator):
    """Visit the full iterator IR tree, convert nodes to respective types and generate constraints."""

    def visit_SymRef(
        self, node: ir.SymRef, constraints: set[tuple[Type, Type]], symtypes: dict[str, Type]
    ) -> Type:
        if node.id in symtypes:
            res = symtypes[node.id]
            if isinstance(res, LetPolymorphic):
                return _freshen(res.dtype)
            return res
        if node.id in ir.BUILTINS:
            return _builtin_type(node.id)

        return TypeVar.fresh()

    def visit_Literal(
        self, node: ir.Literal, constraints: set[tuple[Type, Type]], symtypes: dict[str, Type]
    ) -> Val:
        return Val(kind=Value(), dtype=Primitive(name=node.type))

    def visit_AxisLiteral(
        self,
        node: ir.AxisLiteral,
        constraints: set[tuple[Type, Type]],
        symtypes: dict[str, Type],
    ) -> Val:
        return Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar())

    def visit_OffsetLiteral(
        self,
        node: ir.OffsetLiteral,
        constraints: set[tuple[Type, Type]],
        symtypes: dict[str, Type],
    ) -> TypeVar:
        return TypeVar.fresh()

    def visit_Lambda(
        self, node: ir.Lambda, constraints: set[tuple[Type, Type]], symtypes: dict[str, Type]
    ) -> FunctionType:
        ptypes = {p.id: TypeVar.fresh() for p in node.params}
        ret = self.visit(node.expr, constraints=constraints, symtypes=symtypes | ptypes)
        return FunctionType(args=Tuple(elems=tuple(ptypes[p.id] for p in node.params)), ret=ret)

    def visit_FunCall(
        self, node: ir.FunCall, constraints: set[tuple[Type, Type]], symtypes: dict[str, Type]
    ) -> Type:
        if isinstance(node.fun, ir.SymRef):
            if node.fun.id == "make_tuple":
                # Calls to make_tuple are handled as being part of the grammar,
                # not as function calls
                argtypes = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                kind = TypeVar.fresh()
                size = TypeVar.fresh()
                dtype = Tuple(elems=tuple(TypeVar.fresh() for _ in argtypes))
                for d, a in zip(dtype.elems, argtypes):
                    constraints.add((Val(kind=kind, dtype=d, size=size), a))
                return Val(kind=kind, dtype=dtype, size=size)
            if node.fun.id == "tuple_get":
                # Calls to tuple_get are handled as being part of the grammar,
                # not as function calls
                if len(node.args) != 2:
                    raise TypeError("tuple_get requires exactly two arguments")
                if not isinstance(node.args[0], ir.Literal) or node.args[0].type != "int":
                    raise TypeError("The first argument to tuple_get must be a literal int")
                idx = int(node.args[0].value)
                tup = self.visit(node.args[1], constraints=constraints, symtypes=symtypes)
                kind = TypeVar.fresh()
                elem = TypeVar.fresh()
                size = TypeVar.fresh()
                val = Val(
                    kind=kind,
                    dtype=PartialTupleVar.fresh(elem_indices=(idx,), elem_values=(elem,)),
                    size=size,
                )
                constraints.add((tup, val))
                return Val(kind=kind, dtype=elem, size=size)
            if node.fun.id == "shift":
                # Calls to shift are handled as being part of the grammar, not
                # as function calls; that is, the offsets are completely
                # ignored by the type inference algorithm
                it = Val(kind=Iterator())
                return FunctionType(args=Tuple(elems=(it,)), ret=it)

        fun = self.visit(node.fun, constraints=constraints, symtypes=symtypes)
        args = Tuple(elems=tuple(self.visit(node.args, constraints=constraints, symtypes=symtypes)))
        ret = TypeVar.fresh()
        constraints.add((fun, FunctionType(args=args, ret=ret)))
        return ret

    def visit_FunctionDefinition(
        self,
        node: ir.FunctionDefinition,
        constraints: set[tuple[Type, Type]],
        symtypes: dict[str, Type],
    ) -> FunctionDefinitionType:
        if node.id in symtypes:
            raise TypeError(f"Multiple definitions of symbol {node.id}")

        fun = self.visit(
            ir.Lambda(params=node.params, expr=node.expr),
            constraints=constraints,
            symtypes=symtypes,
        )
        constraints.add((fun, FunctionType()))
        return FunctionDefinitionType(name=node.id, fun=fun)

    def visit_StencilClosure(
        self,
        node: ir.StencilClosure,
        constraints: set[tuple[Type, Type]],
        symtypes: dict[str, Type],
    ) -> Closure:
        domain = self.visit(node.domain, constraints=constraints, symtypes=symtypes)
        stencil = self.visit(node.stencil, constraints=constraints, symtypes=symtypes)
        output = self.visit(node.output, constraints=constraints, symtypes=symtypes)
        inputs = Tuple(
            elems=tuple(self.visit(node.inputs, constraints=constraints, symtypes=symtypes))
        )
        output_dtype = TypeVar.fresh()
        constraints.add((domain, Val(kind=Value(), dtype=Primitive(name="domain"), size=Scalar())))
        constraints.add((output, Val(kind=Iterator(), dtype=output_dtype, size=Column())))
        constraints.add(
            (
                stencil,
                FunctionType(args=inputs, ret=Val(kind=Value(), dtype=output_dtype, size=Column())),
            )
        )
        return Closure(output=output, inputs=inputs)

    def visit_FencilDefinition(
        self,
        node: ir.FencilDefinition,
        constraints: set[tuple[Type, Type]],
        symtypes: dict[str, Type],
    ) -> FencilDefinitionType:
        ftypes = []
        fmap = dict[str, LetPolymorphic]()
        # Note: functions have to be ordered according to Lisp/Scheme `let*`
        # statements; that is, functions can only reference other functions
        # that are defined before
        for f in node.function_definitions:
            c = set[tuple[Type, Type]]()
            ftype: FunctionDefinitionType = self.visit(f, constraints=c, symtypes=symtypes | fmap)
            ftype = typing.cast(FunctionDefinitionType, unify(ftype, c))
            ftypes.append(ftype)
            fmap[ftype.name] = LetPolymorphic(dtype=ftype.fun)

        params = {p.id: TypeVar.fresh() for p in node.params}
        self.visit(node.closures, constraints=constraints, symtypes=symtypes | fmap | params)
        return FencilDefinitionType(
            name=node.id,
            fundefs=tuple(ftypes),
            params=tuple(params[p.id] for p in node.params),
        )


class _FreeVariables(eve.NodeVisitor):
    """Collect type variables within a type expression."""

    def visit_TypeVar(self, node: TypeVar, *, free_variables: set[TypeVar]) -> None:
        self.generic_visit(node, free_variables=free_variables)
        free_variables.add(node)


def _free_variables(x: Type) -> set[TypeVar]:
    """Collect type variables within a type expression."""
    fv = set[TypeVar]()
    _FreeVariables().visit(x, free_variables=fv)
    return fv


class _Dedup(eve.NodeTranslator):
    """Deduplicate nodes that have the same value but a different `id`."""

    def visit(self, node: T, *, memo: dict[T, T]) -> T:  # type: ignore[override]
        node = super().visit(node, memo=memo)
        return memo.setdefault(node, node)


class _Renamer:
    """Efficiently rename (that is, replace) nodes in a type expression.

    Works by collecting all parent nodes of all nodes in a tree. If a node should be replaced by
    another, all referencing parent nodes can be found efficiently and modified in place.

    Note that all types have to be registered before they can be used in a `rename` call.

    Besides basic renaming, this also resolves `ValTuple` and `PrefixTuple` to full `Tuple` if
    possible after renaming.
    """

    def __init__(self) -> None:
        self._parents = dict[Type, list[tuple[Type, str, typing.Optional[int]]]]()

    def register(self, dtype: Type) -> None:
        """Register a type for possible future renaming.

        Collects the parent nodes of all nodes in the type tree.
        """

        def collect_parents(node: Type) -> None:
            for field, child in node.iter_children_items():
                if isinstance(child, Type):
                    self._parents.setdefault(child, []).append(
                        (node, typing.cast(str, field), None)
                    )
                    collect_parents(child)
                elif isinstance(child, tuple):
                    for i, c in enumerate(child):
                        if isinstance(c, Type):
                            self._parents.setdefault(c, []).append(
                                (node, typing.cast(str, field), i)
                            )
                            collect_parents(c)
                else:
                    assert isinstance(child, (int, str))

        collect_parents(dtype)

    @staticmethod
    def _val_tuple_will_resolve_to_tuple(node: Type, field: str, replacement: Type) -> bool:
        """Check if a `ValTuple` instance can be resolved to a `Tuple` after renaming."""
        return isinstance(node, ValTuple) and field == "dtypes" and isinstance(replacement, Tuple)

    @staticmethod
    def _resolve_val_tuple(node: ValTuple, replacement: Tuple) -> Tuple:
        """Resolve a fully defined `ValTuple` instance to a `Tuple`."""
        return Tuple(
            elems=tuple(Val(kind=node.kind, dtype=d, size=node.size) for d in replacement.elems)
        )

    @staticmethod
    def _prefix_tuple_will_resolve_to_tuple(node: Type, field: str, replacement: Type) -> bool:
        """Check if a `PrefixTuple` instance can be resolved to a `Tuple` after renaming."""
        return (
            isinstance(node, PrefixTuple) and field == "others" and isinstance(replacement, Tuple)
        )

    @staticmethod
    def _resolve_prefix_tuple(node: PrefixTuple, replacement: Tuple) -> Tuple:
        """Resolve a fully defined `PrefixTuple` instance to a `Tuple`."""
        return Tuple(elems=(node.prefix,) + replacement.elems)

    def _update_node(
        self, node: Type, field: str, index: typing.Optional[int], replacement: Type
    ) -> None:
        """Replace a field of a node by some other value.

        If `index` is `None`, basically performs `setattr(node, field, replacement)`. Otherwise,
        assumes that the given field is a tuple field and replaces only the tuple element that
        matches the given index.

        Further, updates the mapping of node parents and handles the possibly changing hash value of
        the updated node.
        """
        # Pop the node out of the parents dict as its hash could change after modification
        popped = self._parents.pop(node, None)

        # Update the node’s field or field element
        if index is None:
            setattr(node, field, replacement)
        else:
            field_list = list(getattr(node, field))
            field_list[index] = replacement
            setattr(node, field, tuple(field_list))

        # Register `node` to be the new parent of `replacement`
        self._parents.setdefault(replacement, []).append((node, field, index))

        # Put back possible previous entries to the parents dict after possible hash change
        if popped:
            self._parents[node] = popped

    def rename(self, node: Type, replacement: Type) -> None:
        """Rename/replace all occurrences of `node` to/by `replacement`."""
        try:
            # Find parent nodes
            nodes = self._parents.pop(node)
        except KeyError:
            return

        follow_up_renames = list[tuple[Type, Type]]()
        for node, field, index in nodes:
            if self._val_tuple_will_resolve_to_tuple(node, field, replacement):
                assert isinstance(node, ValTuple) and isinstance(replacement, Tuple)  # helping mypy
                # Special case 1: a `ValTuple` instance can be resolved to a `Tuple` after renaming
                tup = self._resolve_val_tuple(node, replacement)
                # So just collect the resolved tuple for a following rename
                follow_up_renames.append((node, tup))
            elif self._prefix_tuple_will_resolve_to_tuple(node, field, replacement):
                assert isinstance(node, PrefixTuple) and isinstance(
                    replacement, Tuple
                )  # helping mypy
                # Special case 2: a `PrefixTuple` instance can be resolved to a `Tuple` after renaming
                tup = self._resolve_prefix_tuple(node, replacement)
                follow_up_renames.append((node, tup))
            else:
                # Default case: just update a field value of the node
                self._update_node(node, field, index, replacement)

        # Handle follow-up renames
        for s, d in follow_up_renames:
            self.register(d)
            self.rename(s, d)


class _Box(Type):
    """Simple value holder, used for wrapping root nodes of a type tree.

    This makes sure that all root nodes have a parent node which can be updated by the `_Renamer`.
    """

    value: Type


class _Unifier:
    """A classical type unifier (Robinson, 1971).

    Computes the most general type satisfying all given constraints. Uses a `_Renamer` for efficient
    type variable renaming.
    """

    def __init__(self, dtype: Type, constraints: set[tuple[Type, Type]]) -> None:
        # Wrap the original `dtype` and all `constraints` to make sure they have a parent node and
        # thus the root nodes are correctly handled by the renamer
        self._dtype = _Box(value=dtype)
        self._constraints = [(_Box(value=s), _Box(value=t)) for s, t in constraints]

        # Create a renamer and register `dtype` and all `constraints` types
        self._renamer = _Renamer()
        self._renamer.register(self._dtype)
        for s, t in self._constraints:
            self._renamer.register(s)
            self._renamer.register(t)

    def unify(self) -> Type:
        """Run the unification."""
        while self._constraints:
            constraint = self._constraints.pop()
            handled = self._handle_constraint(constraint)
            if not handled:
                # Try with swapped LHS and RHS
                handled = self._handle_constraint(constraint[::-1])
            if not handled:
                raise TypeError(
                    f"Can not satisfy constraint: {constraint[0].value} ≡ {constraint[1].value}"
                )

        return self._dtype.value

    def _rename(self, x: Type, y: Type) -> None:
        """Type renaming/replacement."""
        self._renamer.register(x)
        self._renamer.register(y)
        self._renamer.rename(x, y)

    def _add_constraint(self, x: Type, y: Type) -> None:
        """Register a new constraint."""
        x = _Box(value=x)
        y = _Box(value=y)
        self._renamer.register(x)
        self._renamer.register(y)
        self._constraints.append((x, y))

    def _handle_constraint(  # noqa: C901  # too complex
        self, constraint: tuple[_Box, _Box]
    ) -> bool:
        """Handle a single constraint."""
        s, t = (c.value for c in constraint)
        if s == t:
            # Constraint is satisfied if LHS equals RHS
            return True

        if type(s) is TypeVar:
            assert s not in _free_variables(t)
            # Just replace LHS by RHS if LHS is a type variable
            self._rename(s, t)
            return True

        if type(s) is type(t) is FunctionType:
            # Make sure that argument and return type matches if LHS and RHS are function types
            self._add_constraint(s.args, t.args)
            self._add_constraint(s.ret, t.ret)
            return True

        if type(s) is type(t) is Val:
            # Make sure that kind, dtype, and size matches if LHS and RHS are `Val` types
            self._add_constraint(s.kind, t.kind)
            self._add_constraint(s.dtype, t.dtype)
            self._add_constraint(s.size, t.size)
            return True

        if type(s) is type(t) is Tuple:
            if len(s.elems) != len(t.elems):
                # If LHS and RHS are tuple types, they must have the same number of elements…
                raise TypeError(f"Can not satisfy constraint: {s} ≡ {t}")
            for lhs, rhs in zip(s.elems, t.elems):
                # …and also the types of all elements must match
                self._add_constraint(lhs, rhs)
            return True

        if type(s) is PartialTupleVar and type(t) is Tuple:
            assert s not in _free_variables(t)
            # Make sure that size and elements in a `PartialTupleVar` LHS match the RHS `Tuple`
            if max(s.elem_indices) >= len(t.elems):
                raise TypeError(f"Can not satisfy constraint: {s} ≡ {t}")
            for i, x in zip(s.elem_indices, s.elem_values):
                self._add_constraint(x, t.elems[i])
            return True

        if type(s) is type(t) is PartialTupleVar:
            assert s not in _free_variables(t) and t not in _free_variables(s)
            # If both, LHS and RHS, are `PartialTupleVar`s, replace both instances by a new,
            # combined instance and make sure that the types of the elements defined in the LHS and
            # RHS match
            se = dict(zip(s.elem_indices, s.elem_values))
            te = dict(zip(t.elem_indices, t.elem_values))
            for i in set(s.elem_indices) & set(t.elem_indices):
                self._add_constraint(se[i], te[i])
            elems = se | te
            combined = PartialTupleVar.fresh(
                elem_indices=tuple(elems.keys()), elem_values=tuple(elems.values())
            )
            self._rename(s, combined)
            self._rename(t, combined)
            return True

        if type(s) is PrefixTuple and type(t) is Tuple:
            assert s not in _free_variables(t)
            # Make sure that all elements match
            self._add_constraint(s.prefix, t.elems[0])
            self._add_constraint(s.others, Tuple(elems=t.elems[1:]))
            return True

        if type(s) is type(t) is PrefixTuple:
            assert s not in _free_variables(t) and t not in _free_variables(s)
            self._add_constraint(s.prefix, t.prefix)
            self._add_constraint(s.others, t.others)
            return True

        if type(s) is ValTuple and type(t) is Tuple:
            # Expand the LHS `ValTuple` to the size of the RHS `Tuple` and make sure they match
            s_elems = tuple(Val(kind=s.kind, dtype=TypeVar.fresh(), size=s.size) for _ in t.elems)
            self._add_constraint(s.dtypes, Tuple(elems=tuple(e.dtype for e in s_elems)))
            self._add_constraint(Tuple(elems=s_elems), t)
            return True

        if type(s) is type(t) is ValTuple:
            assert s not in _free_variables(t) and t not in _free_variables(s)
            self._add_constraint(s.kind, t.kind)
            self._add_constraint(s.dtypes, t.dtypes)
            self._add_constraint(s.size, t.size)
            return True

        if type(s) is UniformValTupleVar and type(t) is Tuple:
            assert s not in _free_variables(t)
            # Replace the LHS `UniformValTupleVar` by the RHS `Tuple` and make sure the types match
            self._rename(s, t)
            elem_dtype = Val(kind=s.kind, dtype=s.dtype, size=s.size)
            for e in t.elems:
                self._add_constraint(e, elem_dtype)
            return True

        if type(s) is type(t) is UniformValTupleVar:
            self._add_constraint(s.kind, t.kind)
            self._add_constraint(s.dtype, t.dtype)
            self._add_constraint(s.size, t.size)
            return True

        # Constraint handling failed
        return False


def unify(dtype: Type, constraints: set[tuple[Type, Type]]) -> Type:
    """Unify all given constraints."""
    # Deduplicate nodes, this can speed up later things a bit
    memo = dict[T, T]()
    dtype = _Dedup().visit(dtype, memo=memo)
    constraints = {_Dedup().visit(c, memo=memo) for c in constraints}
    del memo

    unifier = _Unifier(dtype, constraints)
    return unifier.unify()


def reindex_vars(dtype: T) -> T:
    """Reindex all type variables, to have nice indices starting at zero."""

    def indexer(index_map: dict[int, int]) -> int:
        return len(index_map)

    index_map = dict[int, int]()
    return _TypeVarReindexer(indexer).visit(dtype, index_map=index_map)


def infer(expr: ir.Node, symtypes: typing.Optional[dict[str, Type]] = None) -> Type:
    """Infer the type of the given iterator IR expression."""
    if symtypes is None:
        symtypes = dict()

    # Collect constraints
    constraints = set[tuple[Type, Type]]()
    dtype = _TypeInferrer().visit(expr, constraints=constraints, symtypes=symtypes)
    # Compute the most general type that satisfies all constraints
    unified = unify(dtype, constraints)
    return reindex_vars(unified)


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

    def _fmt_dtype(self, kind: Type, dtype_str: str) -> str:
        if kind == Value():
            return dtype_str
        if kind == Iterator():
            return "It[" + dtype_str + "]"
        assert isinstance(kind, TypeVar)
        return "ItOrVal" + self._subscript(kind.idx) + "[" + dtype_str + "]"

    def visit_Tuple(self, node: Tuple) -> str:
        return "(" + ", ".join(self.visit(e) for e in node.elems) + ")"

    def visit_PartialTupleVar(self, node: PartialTupleVar) -> str:
        s = ""
        if node.elem_indices:
            e = dict(zip(node.elem_indices, node.elem_values))
            for i in range(max(e) + 1):
                s += (self.visit(e[i]) if i in e else "_") + ", "
        return "(" + s + "…)" + self._subscript(node.idx)

    def visit_PrefixTuple(self, node: PrefixTuple) -> str:
        return self.visit(node.prefix) + ":" + self.visit(node.others)

    def visit_FunctionType(self, node: FunctionType) -> str:
        return self.visit(node.args) + " → " + self.visit(node.ret)

    def visit_Val(self, node: Val) -> str:
        return self._fmt_dtype(node.kind, self.visit(node.dtype) + self._fmt_size(node.size))

    def visit_Primitive(self, node: Primitive) -> str:
        return node.name

    def visit_FunctionDefinitionType(self, node: FunctionDefinitionType) -> str:
        return node.name + " :: " + self.visit(node.fun)

    def visit_Closure(self, node: Closure) -> str:
        return self.visit(node.inputs) + " ⇒ " + self.visit(node.output)

    def visit_FencilDefinitionType(self, node: FencilDefinitionType) -> str:
        return (
            "{"
            + "".join(self.visit(f) + ", " for f in node.fundefs)
            + node.name
            + "("
            + ", ".join(self.visit(p) for p in node.params)
            + ")}"
        )

    def visit_ValTuple(self, node: ValTuple) -> str:
        if not isinstance(node.dtypes, TypeVar):
            return self.visit_Type(node)
        return (
            "("
            + self._fmt_dtype(node.kind, "T" + self._fmt_size(node.size))
            + ", …)"
            + self._subscript(node.dtypes.idx)
        )

    def visit_UniformValTupleVar(self, node: UniformValTupleVar) -> str:
        return (
            "("
            + self.visit(Val(kind=node.kind, dtype=node.dtype, size=node.size))
            + " × n"
            + self._subscript(node.idx)
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
