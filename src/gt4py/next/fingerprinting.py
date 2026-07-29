# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fingerprinting infrastructure for GT4Py objects.

Provides deterministic, content-based hashing (fingerprinting) of arbitrary
Python objects, built on a generic catamorphism (generalized tree fold) over
one-level deconstructions of those objects.

Public API:

- `Deconstruction` / `EmptyDeconstruction` / `OrderInsensitiveDeconstruction`
  — results of deconstructing one level of an object.
- `Deconstructor` — type alias for a per-object deconstructor.
- `make_deconstructor` — build a dispatching deconstructor from per-type overrides and a fallback.
- `make_strict_data_deconstructor` / `make_lenient_data_deconstructor` — build the
  by-reference (strict) / structural (lenient) deconstructor, optionally with overrides.
- `make_fingerprinter` — build a fingerprinter from a deconstructor and an aggregator.
- `catabolize` — iterative bottom-up fold over a deconstructed object graph.
- `Aggregator` / `Fingerprinter` — type aliases.
- `fingerprint_aggregator` — the xxhash-based aggregator used by all built-in fingerprinters.
- `strict_fingerprinter` — cross-process deterministic fingerprinter.
- `strict_fingerprint_deconstructor` — the default deconstructor, used by `strict_fingerprinter`.
- `lenient_fingerprinter` — single-process fingerprinter tolerant of non-importable objects.
- `lenient_fingerprint_deconstructor` — deconstructor used by `lenient_fingerprinter`.
- `NodeDeconstructor` — type alias for a per-`eve.Node` deconstructor.
- `skipping_fields_node_deconstructor` — node deconstructor (a `make_*_data_deconstructor`
  override) that skips named node fields.

The per-field opt-out helper `gt4py_metadata` (and its namespace key
`GT4PY_CLASS_METADATA_NS`) used to live here; they now live in
`gt4py.next.utils`.
"""

from __future__ import annotations

import collections
import copyreg
import dataclasses
import enum
import functools
import struct
import sys
import types
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Any, Final, Optional, TypeAlias, TypeVar
from unittest import mock

import xxhash

from gt4py.eve import concepts, datamodels, utils as eve_utils
from gt4py.next import utils as next_utils


_T = TypeVar("_T")
_R = TypeVar("_R")


# -- Generic deconstruction of arbitrary objects --


@dataclasses.dataclass(frozen=True, slots=True)
class Deconstruction(Collection[_T]):
    """
    A container with custom internal state for the pieces resulting from an object deconstruction.
    """

    state: Any
    pieces: tuple[Any, ...]

    def __contains__(self, __x: object) -> bool:
        return __x in self.pieces

    def __iter__(self) -> Iterator[_T]:
        return iter(self.pieces)

    def __len__(self) -> int:
        return len(self.pieces)

    @staticmethod
    def from_id(obj: Any) -> EmptyDeconstruction:
        """
        Builder for objects identified by their `id()`.

        Note that this builder should only be used in special cases, e.g. for mock
        objects in test sessions, because `id()` is not a stable identifier across
        processes or sessions.
        """
        return EmptyDeconstruction(b"id\0" + str(id(obj)).encode())

    @staticmethod
    def from_reference(obj: Any, *, strict: bool = True) -> EmptyDeconstruction:
        """
        Builder for objects identified by their fully qualified name.

        With ``strict`` (the default) non-importable objects are rejected; with
        ``strict=False`` the qualified name is trusted unverified (the lenient
        by-reference variant). See `_reference_by_fully_qualified_name`.
        """
        return EmptyDeconstruction(
            b"ref\0" + _reference_by_fully_qualified_name(obj, check=strict).encode()
        )

    @staticmethod
    def from_typed_value(type_: type, /, value: bytes = b"") -> EmptyDeconstruction:
        """Builder with a domain-separation tag and optional payload."""
        return EmptyDeconstruction(_class_tag(type_) + b"\0" + value)

    @staticmethod
    def from_pieces(*pieces: Any, state: Any = None, ordered: bool = True) -> Deconstruction:
        """Builder for creating a deconstruction with the given pieces."""
        deconstruction_cls = OrderInsensitiveDeconstruction if not ordered else Deconstruction
        return deconstruction_cls(state=state, pieces=pieces)


@dataclasses.dataclass(frozen=True, slots=True)
class EmptyDeconstruction(Deconstruction[_T]):
    state: Any = None
    pieces: tuple[Any, ...] = dataclasses.field(default=(), init=False, repr=False)


@dataclasses.dataclass(frozen=True, slots=True)
class OrderInsensitiveDeconstruction(Deconstruction[_T]):
    """A deconstruction whose pieces are semantically order-insensitive and should be reduced in a canonical order."""

    pass


#: Deconstructs one level of an object.
Deconstructor: TypeAlias = Callable[[Any], Deconstruction]


def _resolves_as_global(obj: Any) -> bool:
    """Return whether resolving ``<module>.<qualname>`` through `sys.modules` yields `obj` itself."""
    if isinstance(obj, types.ModuleType):
        return sys.modules.get(obj.__name__) is obj
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not module or not qualname:
        return False
    target: Any = sys.modules.get(module, None)
    for part in qualname.split("."):
        if (target := getattr(target, part, None)) is None:
            return False
    return target is obj


def _reference_by_fully_qualified_name(obj: Any, check: bool) -> str:
    """
    Return the qualified name of `obj`, optionally rejecting non-importable objects.

    Objects identified by their fully qualified name (import path) can only be
    properly fingerprinted across processes if they are importable, module-level
    objects, whose qualified name uniquely identifies them. When ``check`` is
    set, anything with ``<locals>`` in its qualified name (closures, nested
    functions, classes defined inside a function) and anonymous ``<lambda>``
    callables are rejected by a cheap string check; additionally, the name is
    resolved through `sys.modules` and required to yield `obj` itself, which
    also rejects shadowed, reassigned or deleted names whose qualified name no
    longer identifies the object. When ``check`` is unset (the lenient
    by-reference variant) the name is returned unverified, trusting it as the
    object's identity within a single process.
    """
    fqn = eve_utils.get_fully_qualified_name(obj)
    if check and ("<locals>" in fqn or "<lambda>" in fqn or not _resolves_as_global(obj)):
        raise TypeError(
            f"Objects which are not importable under their qualified name ('{fqn}') -- e.g. "
            "locally defined or anonymous callables, shadowed or reassigned globals -- cannot "
            "be safely referenced since the name does not uniquely identify them."
        )
    return fqn


@functools.cache
def _class_tag(cls: type) -> bytes:
    return eve_utils.get_fully_qualified_name(cls).encode()


def _class_obj_tag(obj: Any) -> bytes:
    cls: type = type(obj)
    return _class_tag(cls)


def _instance_state(obj: Any) -> tuple[Any, ...]:
    """
    Extra instance ``__dict__`` state carried by a builtin-container *subclass*.

    Exact ``dict``/``list``/``set``/... instances have no instance ``__dict__``,
    so this is empty and the fingerprint is unchanged for them. Subclasses that
    set extra attributes (which the items-only deconstruction would otherwise
    drop, yielding false cache hits) contribute them here.
    """
    state = getattr(obj, "__dict__", None)
    return ((b"__dict__\0", state),) if state else ()


_COMMON_DECONSTRUCTORS: Final[dict[type, Deconstructor]] = {
    **{
        # for mixin-based enums (`IntEnum`, `IntFlag`, `StrEnum`, ...) the value-type base
        # precedes `Enum` in the MRO and would win the dispatch.
        base: (
            lambda obj: Deconstruction.from_pieces(
                type(obj), obj.name, obj.value, state=b"enum_member"
            )
        )
        for base in (
            enum.Enum,
            enum.IntEnum,
            enum.IntFlag,
            enum.Flag,
            getattr(enum, "StrEnum", None),
        )
        if base is not None
    },
    enum.EnumMeta: lambda obj: Deconstruction.from_pieces(
        *((member.name, member.value) for member in obj),
        state=b"enum_class\0" + eve_utils.get_fully_qualified_name(obj).encode(),
    ),
    type(None): lambda obj: EmptyDeconstruction.from_typed_value(type(None)),
    bool: lambda obj: EmptyDeconstruction.from_typed_value(bool, b"1" if obj else b"0"),
    int: lambda obj: EmptyDeconstruction.from_typed_value(type(obj), str(int(obj)).encode()),
    float: lambda obj: EmptyDeconstruction.from_typed_value(type(obj), struct.pack(">d", obj)),
    complex: lambda obj: EmptyDeconstruction.from_typed_value(
        type(obj), struct.pack(">dd", obj.real, obj.imag)
    ),
    str: lambda obj: EmptyDeconstruction.from_typed_value(
        type(obj), obj.encode("utf-8", "surrogatepass")
    ),
    bytes: lambda obj: EmptyDeconstruction.from_typed_value(type(obj), bytes(obj)),
    bytearray: lambda obj: EmptyDeconstruction.from_typed_value(type(obj), bytes(obj)),
    tuple: lambda obj: Deconstruction.from_pieces(
        *obj, *_instance_state(obj), state=_class_obj_tag(obj)
    ),
    list: lambda obj: Deconstruction.from_pieces(
        *obj, *_instance_state(obj), state=_class_obj_tag(obj)
    ),
    dict: lambda obj: Deconstruction.from_pieces(
        *obj.items(),
        *_instance_state(obj),
        state=_class_obj_tag(obj),
        ordered=isinstance(obj, collections.OrderedDict),
    ),
    collections.defaultdict: lambda obj: Deconstruction.from_pieces(
        obj.default_factory, dict(obj), *_instance_state(obj), state=_class_obj_tag(obj)
    ),
    set: lambda obj: Deconstruction.from_pieces(
        *obj, *_instance_state(obj), state=_class_obj_tag(obj), ordered=False
    ),
    frozenset: lambda obj: Deconstruction.from_pieces(
        *obj, *_instance_state(obj), state=_class_obj_tag(obj), ordered=False
    ),
    # `Namespace`/`FrozenNamespace` (e.g. frontend-valid constant namespaces in
    # closure vars) are deconstructed by their contents. This both content-hashes
    # the namespace and avoids the `__reduce_ex__` fallback, which would reference
    # the (possibly locally-defined, non-importable) class and raise.
    eve_utils.Namespace: lambda obj: Deconstruction.from_pieces(
        *obj.items(), state=b"namespace\0" + _class_obj_tag(obj), ordered=False
    ),
}

# Pickle protocol used when deconstructing unknown objects via
# `__reduce_ex__`. Pinned so the deconstruction (and therefore the
# fingerprint) does not depend on the running Python's default protocol.
_DECONSTRUCT_PICKLE_REDUCE_PROTOCOL: Final[int] = 2


def object_deconstruct_fallback(obj: Any) -> Deconstruction:
    """
    General deconstructor for objects.

    Dataclasses and datamodels are deconstructed through their fields, everything else is
    delegated to the ``__reduce_ex__`` rules.

    We need to define this workaround instead of dispatching on virtual
    Dataclass / DataModel ABCs, since complicated inheritance / composition
    hierarchies breaks `functools.singledispatch` dispatch.
    """
    cls: type = type(obj)

    if (fields := _data_fields(cls)) is not None:
        return _fields_deconstruction(
            cls,
            ((f.name, getattr(obj, f.name)) for f in fields),
        )

    if issubclass(cls, mock.Mock):
        # Workaround for mock objects, which override their own `__class__` attribute
        # and therefore cannot be correctly dispatched by `functools.singledispatch`.
        return Deconstruction.from_id(obj)

    try:
        # Like `pickle.Pickler`, give precedence to reducers registered in
        # `copyreg.dispatch_table` (e.g. NumPy registers one for `ufunc`s,
        # whose direct `__reduce_ex__` raises).
        if (dispatched_reducer := copyreg.dispatch_table.get(cls)) is not None:
            reduced = dispatched_reducer(obj)
        else:
            reduced = obj.__reduce_ex__(_DECONSTRUCT_PICKLE_REDUCE_PROTOCOL)
    except Exception as error:
        raise TypeError(f"Cannot fingerprint object of type '{cls.__name__}'.") from error
    if isinstance(reduced, str):
        # `__reduce__` may return a bare name to be looked up in the object's module.
        return EmptyDeconstruction(
            b"global\0" + f"{getattr(obj, '__module__', '')}:{reduced}".encode()
        )
    constructor, constructor_args, *rest = reduced
    obj_state = rest[0] if len(rest) > 0 else None
    list_items = tuple(rest[1]) if len(rest) > 1 and rest[1] is not None else ()
    dict_items = tuple(rest[2]) if len(rest) > 2 and rest[2] is not None else ()
    custom_setstate = rest[3] if len(rest) > 3 else None

    return Deconstruction.from_pieces(
        constructor,
        constructor_args,
        obj_state,
        list_items,
        dict_items,
        custom_setstate,
        state=b"reduce",
    )


#: Strict deconstructors map used by `strict_fingerprinter`
STRICT_DECONSTRUCTORS: Final[dict[type, Deconstructor]] = _COMMON_DECONSTRUCTORS | {
    type: EmptyDeconstruction.from_reference,
    types.FunctionType: EmptyDeconstruction.from_reference,
    types.BuiltinFunctionType: EmptyDeconstruction.from_reference,
    types.ModuleType: EmptyDeconstruction.from_reference,
}


def _deconstruct_code(code: types.CodeType) -> Deconstruction:
    """Deconstruct a code object by its behaviorally-relevant, deterministic parts."""
    return Deconstruction.from_pieces(
        code.co_code,
        code.co_consts,  # includes nested code objects (inner functions, comprehensions)
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        state=b"code\0" + code.co_name.encode(),
    )


def _deconstruct_cell(cell: types.CellType) -> Deconstruction:
    """Deconstruct a closure cell by its captured value (an empty cell carries none)."""
    try:
        contents = cell.cell_contents
    except ValueError:
        return EmptyDeconstruction.from_typed_value(types.CellType, b"empty")
    return Deconstruction.from_pieces(contents, state=b"cell")


def _lenient_function_deconstruction(func: types.FunctionType) -> Deconstruction:
    try:
        return EmptyDeconstruction.from_reference(func, strict=True)
    except TypeError:
        # Non-importable callable (lambda / nested / locally-defined): hash its
        # code, defaults and captured closure instead of its (non-identifying)
        # qualified name. Globals are intentionally *not* included -- the
        # bytecode references them by name (in ``co_names``), so the body still
        # affects the fingerprint without dragging in the whole module namespace.
        return Deconstruction.from_pieces(
            func.__code__,
            func.__defaults__,
            func.__kwdefaults__,
            func.__closure__ or (),
            state=b"function\0" + func.__qualname__.encode(),
        )


# Tolerant by-reference deconstructor: identifies a dynamically-created
# type/module (e.g. a ``unittest.mock.Mock`` subclass) by its (unverified)
# qualified name, skipping the resolve check the strict reference requires
# instead of raising.
_lenient_reference = functools.partial(Deconstruction.from_reference, strict=False)

#: Tolerant deconstructors map used by `lenient_fingerprinter`
LENIENT_DECONSTRUCTORS: Final[dict[type, Deconstructor]] = {
    types.FunctionType: _lenient_function_deconstruction,
    types.BuiltinFunctionType: _lenient_reference,
    types.ModuleType: _lenient_reference,
    type: _lenient_reference,
    types.CodeType: _deconstruct_code,
    types.CellType: _deconstruct_cell,
}


@functools.cache
def _data_fields(cls: type) -> Optional[tuple[Any, ...]]:
    # Cached per type, since this is on the fingerprinting hot path and the
    # fields are a pure function of the class.
    if dataclasses.is_dataclass(cls):
        return dataclasses.fields(cls)
    if datamodels.is_datamodel(cls):
        return tuple(datamodels.get_fields(cls).values())
    return None


def _fields_deconstruction(cls: type, items: Iterable[tuple[Any, Any]]) -> Deconstruction:
    """
    Deconstruct a dataclass-like object into its ``(field name, value)`` pieces.

    Shared by every fields-based deconstructor (default, fingerprinting and
    node-field-skipping) so they agree on the domain-separation tag and the
    piece shape.
    """
    return Deconstruction.from_pieces(*items, state=b"fields\0" + _class_tag(cls))


def _is_fingerprinted_field(metadata: Mapping[str, Any]) -> bool:
    gt4py_meta = metadata.get(next_utils.GT4PY_CLASS_METADATA_NS, None)
    return not gt4py_meta or gt4py_meta.get("fingerprint", True)


def metadata_based_deconstructor_fallback(obj: Any) -> Deconstruction:
    """
    Default deconstructor understanding dataclasses and datamodels through their fields.

    Dataclasses and datamodels are deconstructed through their fields, dropping
    those marked with ``gt4py_metadata(fingerprint=False)``; everything else is
    delegated to the ``__reduce_ex__`` rules.

    We need to define this workaround instead of dispatching on virtual
    Dataclass / DataModel ABCs, since complicated inheritance / composition
    hierarchies breaks `singledispatch`.

    Used as the `fallback` of the ``make_*_data_deconstructor`` factories.
    """
    cls: type = type(obj)
    if (fields := _data_fields(cls)) is not None:
        return _fields_deconstruction(
            cls,
            ((f.name, getattr(obj, f.name)) for f in fields if _is_fingerprinted_field(f.metadata)),
        )
    else:
        return object_deconstruct_fallback(obj)


def _caller_module_name() -> Optional[str]:
    # Like `collections.namedtuple`: the frame two levels up is the caller of
    # the maker function this helper is called from.
    try:
        return sys._getframe(2).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        return None


def make_deconstructor(
    overrides: Optional[Mapping[type, Deconstructor]] = None,
    *,
    fallback: Deconstructor,
    name: Optional[str] = None,
    module: Optional[str] = None,
) -> Deconstructor:
    """
    Create a deconstructor, optionally with customized per-type deconstruction.

    The returned deconstructor produces one level of an object as a
    `Deconstruction` (an `EmptyDeconstruction` for terminal
    objects) whose `state` is a byte tag (a domain-separation tag, optionally
    followed by a payload). Per-type deconstructors are dispatched on the
    object's MRO.

    Args:
        overrides: Per-type deconstructors taking precedence over the default
            rules.
        fallback: Handles objects without a matching deconstructor (by default
            deconstructed through their fields for dataclasses and datamodels,
            or via the standard ``__reduce_ex__`` protocol).
        name: Sets the returned dispatcher's ``__name__`` / ``__qualname__``.
            Pass the name under which the deconstructor will be stored to make
            it picklable by reference (e.g. for shipping an executor carrying
            it to a process-pool worker); without it the dispatcher carries the
            identity that ``functools.singledispatch`` copies from `fallback`,
            so pickle would resolve it to the plain fallback function instead.
        module: Sets the returned dispatcher's ``__module__``; defaults to the
            caller's module (determined like in ``collections.namedtuple``).

    Returns:
        A single-dispatch deconstructor with the registered implementations.
    """
    deconstructor = eve_utils.singledispatcher(
        fallback,
        implementations={**_COMMON_DECONSTRUCTORS, **(overrides or {})},
    )
    if name is not None:
        deconstructor.__name__ = deconstructor.__qualname__ = name
        if module is None:
            module = _caller_module_name()
    if module is not None:
        deconstructor.__module__ = module
    return deconstructor


def make_strict_data_deconstructor(
    overrides: Optional[Mapping[type, Deconstructor]] = None,
    *,
    name: Optional[str] = None,
    module: Optional[str] = None,
) -> Deconstructor:
    """Build a strict deconstructor with the standard per-type rules.

    See `make_deconstructor` for the meaning of ``name`` and ``module``; a
    deconstructor created without them is not picklable by reference.
    """
    if name is not None and module is None:
        module = _caller_module_name()
    return make_deconstructor(
        overrides=STRICT_DECONSTRUCTORS | dict(overrides or {}),
        fallback=metadata_based_deconstructor_fallback,
        name=name,
        module=module,
    )


def make_lenient_data_deconstructor(
    overrides: Optional[Mapping[type, Deconstructor]] = None,
    *,
    name: Optional[str] = None,
    module: Optional[str] = None,
) -> Deconstructor:
    """Build a lenient deconstructor with the standard per-type rules.

    See `make_deconstructor` for the meaning of ``name`` and ``module``; a
    deconstructor created without them is not picklable by reference.
    """
    if name is not None and module is None:
        module = _caller_module_name()
    return make_deconstructor(
        overrides=LENIENT_DECONSTRUCTORS | dict(overrides or {}),
        fallback=metadata_based_deconstructor_fallback,
        name=name,
        module=module,
    )


# -- Generic reduction of deconstructed objects --

#: Aggregates a deconstruction into a result; for non-terminal objects, the
#: pieces have already been aggregated into results. This is the algebra of
#: the catamorphism implemented by `catabolize`.
Aggregator: TypeAlias = Callable[[Deconstruction], _R]


class _VisitAction(enum.IntEnum):
    VISIT = 0
    COMBINE = 1


def catabolize(
    obj: Any,
    *,
    deconstructor: Deconstructor,
    aggregator: Aggregator[_R],
    allow_cycles: bool = False,
    memoize: bool = True,
) -> _R:
    """
    Catabolize an object: reduce it bottom-up using a catamorphism (a generalized fold over trees).

    The traversal scheme is fixed: an iterative post-order walk over the
    one-level deconstructions produced by `deconstructor`, so the structure
    depth is bounded neither by the Python recursion limit nor (on Python
    <= 3.10) by the C stack. The reduction logic is supplied as the
    `aggregator` (the algebra of the catamorphism): it aggregates an
    `EmptyDeconstruction` into a result and, for non-terminal objects,
    a `Deconstruction` whose pieces have been replaced by the
    already-aggregated results of the original pieces — in piece order, or in
    canonical sorted order for an `OrderInsensitiveDeconstruction`
    (which requires the results to be orderable).

    Keyword Args:
        deconstructor: Per-object deconstruction of one level of structure.
        aggregator: Aggregation of a deconstruction into a result.
        allow_cycles: Whether to allow cyclic references in the object graph.
            If allowed, a cyclic reference is aggregated as an
            `EmptyDeconstruction` whose state encodes the relative
            depth (in currently open nodes) back up to its target; results
            computed below a cycle target embed context-dependent back
            references and are therefore never memoized. If not allowed,
            cycles raise `ValueError`.
        memoize: Reuse the result of already-reduced subobjects (matched by
            identity), so shared substructure is reduced only once. Requires
            a pure aggregator: results must depend only on the deconstruction,
            not on where the object appears.

    Examples:
        Computing the depth of a nested structure (pieces are aggregated
        before the objects containing them, so the aggregator sees their
        depths):

        >>> def deconstructor(obj):
        ...     if isinstance(obj, (list, tuple)):
        ...         return Deconstruction.from_pieces(*obj)
        ...     return EmptyDeconstruction(obj)
        >>> catabolize(
        ...     [[1, [2]], 3],
        ...     deconstructor=deconstructor,
        ...     aggregator=lambda d: (
        ...         0 if isinstance(d, EmptyDeconstruction) else 1 + max(d.pieces, default=0)
        ...     ),
        ... )
        3
    """
    memo: dict[int, _R] = {}
    keep_alive: list[Any] = []  # ensures the `id()`s used as memo keys stay valid
    open_nodes: dict[int, int] = {}  # id -> depth among currently open (in-progress) nodes
    # Smallest open-node depth targeted by a cyclic back reference: results of
    # nodes opened below it embed context-dependent back references and must
    # not be memoized.
    taint_depth: float = float("inf")

    work: list[tuple[_VisitAction, Any]] = [(_VisitAction.VISIT, obj)]
    results: list[_R] = []
    while work:
        action, payload = work.pop()
        match action:
            case _VisitAction.VISIT:
                current = payload
                current_id = id(current)

                if memoize and current_id in memo:
                    results.append(memo[current_id])
                    continue

                if (depth := open_nodes.get(current_id)) is not None:
                    # Cyclic reference: reduce to a back reference by relative
                    # depth, which is independent of memory addresses.
                    if allow_cycles:
                        relative_depth = len(open_nodes) - depth
                        result = aggregator(
                            EmptyDeconstruction(b"cycle\0" + str(relative_depth).encode())
                        )
                        results.append(result)
                        taint_depth = min(taint_depth, depth)
                        continue

                    raise ValueError(
                        f"Cycle detected on an object of type '{type(current).__name__}' "
                        "and cycles are not allowed."
                    )

                deconstruction = deconstructor(current)
                if isinstance(deconstruction, EmptyDeconstruction):
                    result = aggregator(deconstruction)
                    if memoize:
                        memo[current_id] = result
                        keep_alive.append(current)
                    results.append(result)
                else:
                    open_nodes[current_id] = len(open_nodes)
                    work.append((_VisitAction.COMBINE, (current, deconstruction)))
                    work.extend(
                        (_VisitAction.VISIT, child) for child in reversed(deconstruction.pieces)
                    )

            case _VisitAction.COMBINE:
                current, deconstruction = payload
                num_pieces = len(deconstruction.pieces)
                piece_results = results[len(results) - num_pieces :]
                del results[len(results) - num_pieces :]
                if isinstance(deconstruction, OrderInsensitiveDeconstruction):
                    # Sorting the piece *results* canonicalizes order-insensitive
                    # containers without requiring the pieces themselves to be
                    # orderable (the results must be, see the docstring).
                    piece_results = sorted(piece_results)  # type: ignore[type-var]  # conditional requirement on _R

                # Direct construction (a non-empty `deconstruction` reaching
                # COMBINE always has the `(state, pieces)` signature) avoids the
                # field-introspection overhead of `dataclasses.replace` per node.
                result = aggregator(
                    type(deconstruction)(deconstruction.state, tuple(piece_results))
                )
                depth = open_nodes.pop(id(current))

                if depth <= taint_depth:
                    if memoize:
                        memo[id(current)] = result
                        keep_alive.append(current)
                    if depth == taint_depth:
                        # The target of all pending back references is complete;
                        # its result is self-contained again.
                        taint_depth = float("inf")

                results.append(result)

            case _:
                raise RuntimeError(
                    f"Internal error: invalid action '{action}' in catabolism work list."
                )

    assert len(results) == 1
    return results[0]


# -- Generic fingerprinting of objects --
#: A fingerprinter is `catabolize` instantiated with digest
#: aggregators (xxhash64 with domain separation) over a deconstructor, which
#: must produce type tags as `state`.
Fingerprinter: TypeAlias = Callable[[Any], str]


#: Deconstructor used by `strict_fingerprinter`: the default dispatching
#: deconstructor (built-in per-type rules over the dataclass/datamodel field
#: fallback). Identifies types, functions and modules by their qualified name,
#: rejecting non-importable ones, and honors the
#: ``gt4py_metadata(fingerprint=False)`` opt-out.
strict_fingerprint_deconstructor: Final[Deconstructor] = make_strict_data_deconstructor(
    name="strict_fingerprint_deconstructor"
)


def fingerprint_aggregator(deconstruction: Deconstruction[str]) -> str:
    """Aggregate a deconstruction (with already-aggregated piece digests) into its digest."""
    hasher = xxhash.xxh64(b"node\0")
    hasher.update(deconstruction.state)
    hasher.update(b"pieces\0")
    for digest in deconstruction.pieces:  # fixed-length digests, unambiguous concatenation
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def make_fingerprinter(
    *,
    deconstructor: Deconstructor = strict_fingerprint_deconstructor,
    aggregator: Aggregator[str] = fingerprint_aggregator,
    allow_cycles: bool = True,
) -> Fingerprinter:
    """Helper to make a fingerprinter from a deconstructor and an aggregator."""
    return functools.partial(
        catabolize,
        deconstructor=deconstructor,
        aggregator=aggregator,
        allow_cycles=allow_cycles,
    )


#: Default fingerprinting function for GT4Py objects: deterministic across
#: processes (dict and set contributions are canonicalized by sorting the
#: fingerprints of their items) and identifying types, functions and modules
#: by their qualified name.
strict_fingerprinter: Fingerprinter = make_fingerprinter(
    deconstructor=strict_fingerprint_deconstructor
)


# -- Lenient fingerprinting for complex cases with dynamically-created objects --
#
# The default by-reference rules reject non-importable callables, types and
# modules (see `_reference_by_fully_qualified_name(check=True)`) to make sure persistent,
# on-disk key is always reproducible in a *different* process, where only an
# import path uniquely identifies an object. That requirement is impossible
# to satisfy when there are dynamically-created objects without an import path.
# In that case, the object's structure is a perfectly valid identity. This lenient
# fingerprinter keeps the strict by-reference behavior whenever it applies (so it
# agrees with `strict_fingerprinter` on graphs without non-importable objects), but
# falls back to structural ("by code") hashing for functions and to an
# unverified qualified name for dynamically-created types/modules, instead of
# raising.

lenient_fingerprint_deconstructor: Final[Deconstructor] = make_lenient_data_deconstructor(
    name="lenient_fingerprint_deconstructor"
)

#: Fingerprinter for in-memory (single-process) cache keys: like
#: `strict_fingerprinter`, but tolerant of non-importable callables, types
#: and modules in the object graph (functions are hashed by code + closure,
#: dynamic types/modules by unverified qualified name). It MUST NOT key a
#: persistent cache, whose keys must be reproducible across processes.
lenient_fingerprinter: Fingerprinter = make_fingerprinter(
    deconstructor=lenient_fingerprint_deconstructor
)


# -- Other deconstructors --
#: Deconstructs one level of an `eve.Node`; composed into a fingerprinter as a
#: `concepts.Node` override of a `make_*_data_deconstructor` map.
NodeDeconstructor: TypeAlias = Callable[[concepts.Node], Deconstruction]


@functools.cache
def skipping_fields_node_deconstructor(*skipped_fields: str) -> NodeDeconstructor:
    """
    Return a node deconstructor which skips some fields.

    The result deconstructs one level of an `eve.Node` (dropping the named child
    fields); it is *not* a full `Deconstructor`, so compose it as a
    `concepts.Node` override into `make_strict_data_deconstructor` /
    `make_lenient_data_deconstructor` before building a fingerprinter.
    """

    _skipped_fields_set = frozenset(skipped_fields)

    def node_deconstructor(node: concepts.Node) -> Deconstruction:
        return _fields_deconstruction(
            type(node),
            (
                (name, child)
                for name, child in node.iter_children_items()
                if name not in _skipped_fields_set
            ),
        )

    return node_deconstructor
