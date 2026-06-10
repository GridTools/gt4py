# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import inspect
import itertools
import struct
import types
from collections.abc import Callable, Iterable, Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
    ParamSpec,
    Sequence,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import xxhash

from gt4py.eve import concepts, datamodels, utils as eve_utils


GT4PY_CLASS_METADATA_NS: Final[str] = "GT4PY_META"


def gt4py_metadata(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """
    Helper function to store dataclass/datamodel field metadata within a GT4Py namespace.

    Individual fields can opt out of fingerprinting with
    `foo = field(..., metadata=gt4py_metadata(fingerprint=False))`.
    """
    return {GT4PY_CLASS_METADATA_NS: kwargs}


_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")


# -- Generic bottom-up tree reduction (catamorphism) --
@dataclasses.dataclass(frozen=True, slots=True)
class TreeLeaf:
    """One-level decomposition of a terminal object: an opaque value, no children."""

    value: Any


@dataclasses.dataclass(frozen=True, slots=True)
class TreeNode:
    """
    One-level decomposition of a non-terminal object.

    `metadata` carries whatever node-level information the reduction needs
    (e.g. a type tag), and `children` are the sub-objects to be reduced before
    this node. `ordered` declares whether the order of the children is
    semantically meaningful; order-insensitive reductions (e.g. for `dict` and
    `set`, whose equality ignores iteration order) may use it to canonicalize
    the combination of the child results.
    """

    metadata: Any
    children: tuple[Any, ...]
    ordered: bool = True


#: Decomposes an object into one level of tree structure.
TreeDecomposer: TypeAlias = Callable[[Any], TreeLeaf | TreeNode]

_VISIT: Final[int] = 0
_COMBINE: Final[int] = 1


def tree_cata(
    obj: Any,
    *,
    decompose: TreeDecomposer,
    leaf_alg: Callable[[TreeLeaf], _R],
    node_alg: Callable[[TreeNode, list[_R]], _R],
    cycle_alg: Optional[Callable[[int], _R]] = None,
    memoize: bool = True,
) -> _R:
    """
    Reduce an object bottom-up using a catamorphism (a generalized fold over trees).

    The traversal scheme is fixed: an iterative post-order walk over the
    one-level decompositions produced by `decompose`, so the structure depth
    is bounded neither by the Python recursion limit nor (on Python <= 3.10)
    by the C stack. The reduction logic is supplied as algebras: `leaf_alg`
    reduces a :class:`TreeLeaf` to a result, and `node_alg` combines a
    :class:`TreeNode` with the already-reduced results of its children.

    Keyword Args:
        decompose: Per-object one-level decomposition into a :class:`TreeLeaf`
            or a :class:`TreeNode`.
        leaf_alg: Reduction of a decomposed terminal object.
        node_alg: Combination of a decomposed non-terminal object with the
            results of its children (in child order).
        cycle_alg: Reduction of a cyclic reference, receiving the relative
            depth (in currently open nodes) from the reference back up to its
            target. If `None`, cycles raise :class:`ValueError`. Results
            computed below a cycle target embed context-dependent back
            references and are therefore never memoized.
        memoize: Reuse the result of already-reduced subobjects (matched by
            identity), so shared substructure is reduced only once. Requires
            pure algebras: results must depend only on the decomposition, not
            on where the object appears.

    Examples:
        Computing the depth of a nested structure (children are reduced
        before their parents, so the node algebra only sees child depths):

        >>> def decompose(obj):
        ...     if isinstance(obj, (list, tuple)):
        ...         return TreeNode(metadata=None, children=tuple(obj))
        ...     return TreeLeaf(obj)
        >>> tree_cata(
        ...     [[1, [2]], 3],
        ...     decompose=decompose,
        ...     leaf_alg=lambda leaf: 0,
        ...     node_alg=lambda node, child_depths: 1 + max(child_depths, default=0),
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

    work: list[tuple[int, Any]] = [(_VISIT, obj)]
    results: list[_R] = []
    while work:
        action, payload = work.pop()
        if action == _VISIT:
            current = payload
            current_id = id(current)
            if memoize and current_id in memo:
                results.append(memo[current_id])
                continue
            if (depth := open_nodes.get(current_id)) is not None:
                # Cyclic reference: reduce to a back reference by relative
                # depth, which is independent of memory addresses.
                if cycle_alg is None:
                    raise ValueError(
                        f"Cycle detected on an object of type '{type(current).__name__}' "
                        "and no 'cycle_alg' was provided."
                    )
                results.append(cycle_alg(len(open_nodes) - depth))
                taint_depth = min(taint_depth, depth)
                continue
            decomposed = decompose(current)
            if isinstance(decomposed, TreeLeaf):
                result = leaf_alg(decomposed)
                if memoize:
                    memo[current_id] = result
                    keep_alive.append(current)
                results.append(result)
            else:
                open_nodes[current_id] = len(open_nodes)
                work.append((_COMBINE, (current, decomposed)))
                work.extend((_VISIT, child) for child in reversed(decomposed.children))
        else:
            current, decomposed = payload
            num_children = len(decomposed.children)
            child_results = results[len(results) - num_children :]
            del results[len(results) - num_children :]
            result = node_alg(decomposed, child_results)
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

    assert len(results) == 1
    return results[0]


# -- Fingerprinting: digest algebras over a per-type decomposition registry --
Fingerprinter: TypeAlias = Callable[[Any], str]

#: Decomposer used for fingerprinting: `TreeLeaf.value` and `TreeNode.metadata`
#: must be `bytes` (a domain-separation tag, optionally followed by a payload).
FingerprintHandler: TypeAlias = TreeDecomposer

#: Pickle protocol used when decomposing unknown objects via `__reduce_ex__`.
#: Pinned so the decomposition (and therefore the fingerprint) does not depend
#: on the running Python's default protocol.
_FINGERPRINT_REDUCE_PROTOCOL: Final[int] = 2


def _reference_by_fully_qualified_name(obj: Any) -> str:
    """
    Return the qualified name of `obj`, rejecting non-importable (local) objects.

    Objects identified by their fully qualified name (import path) can only be
    properly fingerprinted if they are importable, module-level objects, whose
    qualified name uniquely identifies them. This function rejects locally defined
    objects: anything with ``<locals>`` in its qualified name (closures, nested
    functions, classes defined inside a function) and anonymous ``<lambda>``
    callables.
    """
    fqn = eve_utils.get_fully_qualified_name(obj)
    if "<locals>" in fqn or "<lambda>" in fqn:
        raise TypeError(
            "Non-importable objects (e.g. locally defined or anonymous callables) cannot be "
            "safely referenced since they share identical qualified names."
        )
    return fqn


_class_tag_cache: dict[type, bytes] = {}


def _class_tag(cls: type) -> bytes:
    if (tag := _class_tag_cache.get(cls)) is None:
        tag = _class_tag_cache[cls] = eve_utils.get_fully_qualified_name(cls).encode()
    return tag


def _leaf(tag: bytes, payload: bytes = b"") -> TreeLeaf:
    return TreeLeaf(tag + b"\0" + payload)


def _decompose_by_reference(obj: Any) -> TreeLeaf:
    return _leaf(b"ref", _reference_by_fully_qualified_name(obj).encode())


def _decompose_dict(obj: dict) -> TreeNode:
    # `collections.OrderedDict` equality is order-sensitive, so its fingerprint
    # must be too; plain `dict` (and other subclasses) compare order-insensitively.
    ordered = isinstance(obj, collections.OrderedDict)
    return TreeNode(_class_tag(type(obj)), tuple(obj.items()), ordered=ordered)


def _decompose_enum_class(obj: type[enum.Enum]) -> TreeNode:
    # Enum classes are decomposed by content (their members) instead of by
    # reference: they are commonly defined locally (non-importable), and their
    # semantic content is exactly the member set.
    return TreeNode(
        b"enum_class\0" + eve_utils.get_fully_qualified_name(obj).encode(),
        tuple((member.name, member.value) for member in obj),
    )


def _decompose_enum_member(obj: enum.Enum) -> TreeNode:
    # The member's class is included by content (see `_decompose_enum_class`),
    # so members of distinct but identically named (e.g. local) enum classes
    # cannot collide.
    return TreeNode(b"enum_member", (type(obj), obj.name, obj.value))


# `enum.Enum` alone is not enough: for mixin-based enums (`IntEnum`, `IntFlag`,
# `StrEnum`, ...) the value-type base precedes `Enum` in the MRO and would win
# the dispatch.
_ENUM_MEMBER_BASES: Final[tuple[type, ...]] = tuple(
    base
    for base in (enum.Enum, enum.IntEnum, enum.IntFlag, enum.Flag, getattr(enum, "StrEnum", None))
    if base is not None
)


_BASE_FINGERPRINT_HANDLERS: Final[dict[type, FingerprintHandler]] = {
    **{base: _decompose_enum_member for base in _ENUM_MEMBER_BASES},
    enum.EnumMeta: _decompose_enum_class,
    type(None): lambda obj: _leaf(b"builtins.NoneType"),
    bool: lambda obj: _leaf(b"builtins.bool", b"1" if obj else b"0"),
    int: lambda obj: _leaf(_class_tag(type(obj)), str(int(obj)).encode()),
    float: lambda obj: _leaf(_class_tag(type(obj)), struct.pack(">d", obj)),
    complex: lambda obj: _leaf(_class_tag(type(obj)), struct.pack(">dd", obj.real, obj.imag)),
    str: lambda obj: _leaf(_class_tag(type(obj)), obj.encode("utf-8", "surrogatepass")),
    bytes: lambda obj: _leaf(_class_tag(type(obj)), bytes(obj)),
    bytearray: lambda obj: _leaf(_class_tag(type(obj)), bytes(obj)),
    tuple: lambda obj: TreeNode(_class_tag(type(obj)), tuple(obj)),
    list: lambda obj: TreeNode(_class_tag(type(obj)), tuple(obj)),
    dict: _decompose_dict,
    set: lambda obj: TreeNode(_class_tag(type(obj)), tuple(obj), ordered=False),
    frozenset: lambda obj: TreeNode(_class_tag(type(obj)), tuple(obj), ordered=False),
    type: _decompose_by_reference,
    types.FunctionType: _decompose_by_reference,
    types.BuiltinFunctionType: _decompose_by_reference,
    types.ModuleType: _decompose_by_reference,
}


def _is_fingerprinted_field(metadata: Mapping[str, Any]) -> bool:
    gt4py_meta = metadata.get(GT4PY_CLASS_METADATA_NS, None)
    return not gt4py_meta or gt4py_meta.get("fingerprint", True)


def _decompose_fallback(obj: Any) -> TreeLeaf | TreeNode:
    cls = type(obj)
    if dataclasses.is_dataclass(cls) or datamodels.is_datamodel(cls):
        fields: Iterable[Any] = (
            dataclasses.fields(cls)
            if dataclasses.is_dataclass(cls)
            else datamodels.get_fields(cls).values()
        )
        return TreeNode(
            b"fields\0" + _class_tag(cls),
            tuple(getattr(obj, f.name) for f in fields if _is_fingerprinted_field(f.metadata)),
        )

    try:
        reduced = obj.__reduce_ex__(_FINGERPRINT_REDUCE_PROTOCOL)
    except Exception as error:
        raise TypeError(f"Cannot fingerprint object of type '{cls.__name__}'.") from error
    if isinstance(reduced, str):
        # `__reduce__` may return a bare name to be looked up in the object's module.
        return _leaf(b"global", f"{getattr(obj, '__module__', '')}:{reduced}".encode())
    constructor, constructor_args, *rest = reduced
    state = rest[0] if len(rest) > 0 else None
    list_items = tuple(rest[1]) if len(rest) > 1 and rest[1] is not None else ()
    dict_items = tuple(rest[2]) if len(rest) > 2 and rest[2] is not None else ()
    custom_setstate = rest[3] if len(rest) > 3 else None
    return TreeNode(
        b"reduce", (constructor, constructor_args, state, list_items, dict_items, custom_setstate)
    )


def _fingerprint_leaf_alg(leaf: TreeLeaf) -> str:
    """Reduce a decomposed terminal object to its digest."""
    return xxhash.xxh64(b"leaf\0" + leaf.value).hexdigest()


def _fingerprint_node_alg(node: TreeNode, child_digests: list[str]) -> str:
    """Combine the digests of a node's children into the node's digest."""
    if not node.ordered:
        # Sorting the child *digests* canonicalizes order-insensitive containers
        # without requiring the children themselves to be orderable.
        child_digests = sorted(child_digests)
    hasher = xxhash.xxh64(b"node\0")
    hasher.update(node.metadata)
    hasher.update(b"\0")
    for digest in child_digests:  # fixed-length digests, unambiguous concatenation
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def _fingerprint_cycle_alg(relative_depth: int) -> str:
    """Reduce a cyclic back reference to a digest of its relative depth."""
    return _fingerprint_leaf_alg(TreeLeaf(b"cycle\0" + str(relative_depth).encode()))


def make_fingerprinter(
    extra_handlers: Optional[Mapping[type, FingerprintHandler]] = None,
) -> Fingerprinter:
    """
    Create a fingerprinting function, optionally with customized per-type handling.

    A fingerprinter is :func:`tree_cata` instantiated with digest algebras
    (xxhash64 with domain separation) over a per-type decomposition registry.
    A handler decomposes an object into a :class:`TreeLeaf` or a
    :class:`TreeNode` whose `value` / `metadata` are byte tags. Handlers are
    dispatched on the object's MRO; `extra_handlers` take precedence over the
    default rules. Objects without a matching handler are decomposed by their
    fields (dataclasses and datamodels, honoring
    ``gt4py_metadata(fingerprint=False)`` field metadata) or via the standard
    ``__reduce_ex__`` protocol.
    """
    decompose = eve_utils.singledispatcher(
        _decompose_fallback,
        implementations={**_BASE_FINGERPRINT_HANDLERS, **(extra_handlers or {})},
    )

    def fingerprinter(obj: Any) -> str:
        return tree_cata(
            obj,
            decompose=decompose,
            leaf_alg=_fingerprint_leaf_alg,
            node_alg=_fingerprint_node_alg,
            cycle_alg=_fingerprint_cycle_alg,
        )

    return fingerprinter


#: Default fingerprinting function for GT4Py objects: deterministic across
#: processes (dict and set contributions are canonicalized by sorting the
#: fingerprints of their items) and identifying types, functions and modules
#: by their qualified name.
stable_fingerprinter: Fingerprinter = make_fingerprinter()


def _make_skipping_fields_node_handler(skipped_fields: frozenset[str]) -> FingerprintHandler:
    def handler(node: concepts.Node) -> TreeNode:
        return TreeNode(
            b"fields\0" + _class_tag(type(node)),
            tuple(
                (name, child)
                for name, child in node.iter_children_items()
                if name not in skipped_fields
            ),
        )

    return handler


@overload
def skipping_fields_node_fingerprinter(
    *skipped_fields: str, return_handlers: Literal[False] = False
) -> Fingerprinter: ...


@overload
def skipping_fields_node_fingerprinter(
    *skipped_fields: str, return_handlers: Literal[True]
) -> tuple[Fingerprinter, dict[type, FingerprintHandler]]: ...


@functools.cache
def skipping_fields_node_fingerprinter(
    *skipped_fields: str, return_handlers: bool = False
) -> Fingerprinter | tuple[Fingerprinter, dict[type, FingerprintHandler]]:
    """
    Return a fingerprinter that fingerprints a node while skipping fields.

    The provided field names are ignored recursively on all nodes. With
    `return_handlers=True`, additionally return the handler registry so it can
    be composed into another fingerprinter via :func:`make_fingerprinter`.
    """
    handlers: dict[type, FingerprintHandler] = {
        concepts.Node: _make_skipping_fields_node_handler(frozenset(skipped_fields))
    }
    fingerprinter = make_fingerprinter(handlers)

    return (fingerprinter, handlers) if return_handlers else fingerprinter


class RecursionGuard:
    """
    Context manager to guard against inifinite recursion.

    >>> def foo(i):
    ...     with RecursionGuard(i):
    ...         if i % 2 == 0:
    ...             foo(i)
    ...     return i
    >>> foo(3)
    3
    >>> foo(2)  # doctest:+ELLIPSIS
    Traceback (most recent call last):
        ...
    gt4py.next.utils.RecursionGuard.RecursionDetected
    """

    guarded_objects: ClassVar[set[int]] = set()

    obj: Any

    class RecursionDetected(Exception):
        pass

    def __init__(self, obj: Any):
        self.obj = obj

    def __enter__(self) -> None:
        if id(self.obj) in self.guarded_objects:
            raise self.RecursionDetected()
        self.guarded_objects.add(id(self.obj))

    def __exit__(self, *exc: Any) -> None:
        self.guarded_objects.remove(id(self.obj))


def is_tuple_of(v: Any, t: type[_T]) -> TypeGuard[tuple[_T, ...]]:
    return isinstance(v, tuple) and all(isinstance(e, t) for e in v)


# TODO(havogt): remove flatten duplications in the whole codebase
def flatten_nested_tuple(
    value: tuple[
        _T | tuple, ...
    ],  # `_T` omitted on purpose as type of `value`, to properly deduce `_T` on the user-side
) -> tuple[_T, ...]:
    if isinstance(value, tuple):
        return sum((flatten_nested_tuple(v) for v in value), start=())  # type: ignore[arg-type] # cannot properly express nesting
    else:
        return (value,)


@overload
def tree_map(
    fun: Callable[_P, _R],
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[..., _R | tuple[_R | tuple, ...]]: ...


@overload
def tree_map(
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[
    [Callable[_P, _R]], Callable[..., Any]
]: ...  # TODO(havogt): typing of `result_collection_constructor` is too weak here


def tree_map(
    fun: Optional[Callable[_P, _R]] = None,
    *,
    collection_type: type | tuple[type, ...] = tuple,
    result_collection_constructor: Optional[Callable] = None,
    unpack: bool = False,
    with_path_arg: bool = False,
) -> Callable[..., _R | tuple[_R | tuple, ...]] | Callable[[Callable[_P, _R]], Callable[..., Any]]:
    """
    Apply `fun` to each entry of (possibly nested) collections (by default `tuple`s).

    Args:
        fun: Function to apply to each entry of the collection.
        collection_type: Type of the collection to be traversed. Can be a single type or a tuple of types.
        result_collection_constructor: Type of the collection to be returned. If `None` the same type as `collection_type` is used.
        unpack: Replicate tuple structure returned from `fun` to the mapped result, i.e. return
          tuple of result collections instead of result collections of tuples.
        with_path_arg: Pass the path to access the current element to `fun`.
    Examples:
        >>> tree_map(lambda x: x + 1)(((1, 2), 3))
        ((2, 3), 4)

        >>> tree_map(lambda x, y: x + y)(((1, 2), 3), ((4, 5), 6))
        ((5, 7), 9)

        >>> tree_map(collection_type=list)(lambda x: x + 1)([[1, 2], 3])
        [[2, 3], 4]

        >>> tree_map(
        ...     collection_type=(list, tuple),
        ...     result_collection_constructor=lambda value, elts: (
        ...         tuple(elts) if isinstance(value, list) else list(elts)
        ...     ),
        ... )(lambda x: x + 1)([(1, 2), 3])
        ([2, 3], 4)

        >>> @tree_map
        ... def impl(x):
        ...     return x + 1
        >>> impl(((1, 2), 3))
        ((2, 3), 4)

        >>> @tree_map(with_path_arg=True)
        ... def impl(x, path: tuple[int, ...]):
        ...     path_str = "".join(f"[{i}]" for i in path)
        ...     return f"t{path_str} = {x}"
        >>> t = impl(((1, 2), 3))
        >>> t[0][0]
        't[0][0] = 1'
        >>> t[0][1]
        't[0][1] = 2'
        >>> t[1]
        't[1] = 3'

        >>> @tree_map(unpack=True)
        ... def impl(x):
        ...     return (x, x**2)
        >>> identity, squared = impl(((2, 3), 4))
        >>> identity
        ((2, 3), 4)
        >>> squared
        ((4, 9), 16)
    """

    if result_collection_constructor is None:
        if isinstance(collection_type, tuple):
            # Note: that doesn't mean `collection_type=tuple`, but e.g. `collection_type=(list, tuple)`
            raise TypeError(
                "tree_map() requires `result_collection_constructor` when `collection_type` is a tuple of types."
            )
        result_collection_constructor = lambda _, elts: collection_type(elts)  # noqa: E731 # because a lambda is clearer

    if fun:

        @functools.wraps(fun)
        def impl(*args: Any | tuple[Any | tuple, ...]) -> _R | tuple[_R | tuple, ...]:
            if isinstance(args[0], collection_type):
                non_path_args: Sequence[Any]
                if with_path_arg:
                    *non_path_args, path = args
                    args = (*non_path_args, tuple((*path, i) for i in range(len(args[0]))))
                else:
                    non_path_args = args

                assert all(
                    isinstance(arg, collection_type) and len(args[0]) == len(arg)
                    for arg in non_path_args
                )
                assert result_collection_constructor is not None
                ctor = functools.partial(result_collection_constructor, args[0])

                mapped = [impl(*arg) for arg in zip(*args)]
                if unpack:
                    return tuple(map(ctor, zip(*mapped)))
                else:
                    return ctor(mapped)

            return fun(  # type: ignore[call-arg]
                *cast(_P.args, args),  # type: ignore[valid-type]
            )  # mypy doesn't understand that `args` at this point is of type `_P.args`

        if with_path_arg:
            return lambda *args: impl(*args, ())
        else:
            return impl
    else:
        return functools.partial(
            tree_map,
            collection_type=collection_type,
            result_collection_constructor=result_collection_constructor,
            unpack=unpack,
            with_path_arg=with_path_arg,
        )


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def equalize_tuple_structure(
    d1: _T1, d2: _T2, *, fill_value: Any, bidirectional: bool = True
) -> tuple[_T1, _T2]:
    """
    Given two values or tuples thereof equalize their structure.

    If one of the arguments is a tuple the argument will be promoted to a tuple of the same
    structure.

    >>> equalize_tuple_structure((1,), (2, 3), fill_value=None) == (
    ...     (1, None),
    ...     (2, 3),
    ... )
    True

    >>> equalize_tuple_structure(1, (2, 3), fill_value=None) == (
    ...     (1, 1),
    ...     (2, 3),
    ... )
    True

    If `bidirectional` is `False` only the second argument is equalized and an error is raised if
    the first argument would require promotion.
    """
    d1_is_tuple = isinstance(d1, tuple)
    d2_is_tuple = isinstance(d2, tuple)
    if not d1_is_tuple and d2_is_tuple:
        if not bidirectional:
            raise ValueError(f"Expected `{d2!s}` to have same structure as `{d1!s}`.")
        return equalize_tuple_structure(
            (d1,) * len(d2),  # type: ignore[arg-type] # ensured by condition above
            d2,
            fill_value=fill_value,
            bidirectional=bidirectional,
        )
    if d1_is_tuple and not d2_is_tuple:
        return equalize_tuple_structure(
            d1,
            (d2,) * len(d1),  # type: ignore[arg-type] # ensured by condition above
            fill_value=fill_value,
            bidirectional=bidirectional,
        )
    if d1_is_tuple and d2_is_tuple:
        if not bidirectional and len(d1) < len(d2):  # type: ignore[arg-type] # ensured by condition above
            raise ValueError(f"Expected `{d2!s}` to be longer than or equal to `{d1!s}`.")
        return tuple(  # type: ignore[return-value] # mypy not smart enough
            zip(
                *(
                    equalize_tuple_structure(
                        el1, el2, fill_value=fill_value, bidirectional=bidirectional
                    )
                    for el1, el2 in itertools.zip_longest(d1, d2, fillvalue=fill_value)  # type: ignore[call-overload] # d1, d2 are tuples
                )
            )
        )
    return d1, d2


def make_args_canonicalizer(
    signature: inspect.Signature,
    *,
    name: str | None = None,
) -> Callable[..., tuple[tuple, dict[str, Any]]]:
    """
    Create a call arguments canonicalizer function from a given signature.

    The canonicalizer returns the call arguments in a tuple with the
    following two items:

        - a tuple of values containing all the positional parameters
        - a dictionary of parameter names to values for all keyword-only parameters

    Args:
        signature: The signature for which to create the canonicalizer.

    Keyword Args:
        name: Name of the function for which the canonicalizer is created.

    """

    params = []
    pos_args = []
    key_args = []
    var_pos_arg: str | None = None
    var_key_arg: str | None = None

    for key, param in signature.parameters.items():
        match param.kind:
            case inspect.Parameter.POSITIONAL_ONLY | inspect.Parameter.POSITIONAL_OR_KEYWORD:
                pos_args.append(f"{key}")
            case inspect.Parameter.VAR_POSITIONAL:
                var_pos_arg = key
                pos_args.append(f"*{key}")
            case inspect.Parameter.KEYWORD_ONLY:
                key_args.append(f"'{key}': {key}")
            case inspect.Parameter.VAR_KEYWORD:
                var_key_arg = key
                key_args.append(f"**{key}")

        # Remove annotations to avoid issues in case `str(annotation)`
        # produces incorrect python expressions
        params.append(param.replace(annotation=param.empty))

    canonicalizer_signature = inspect.Signature(parameters=params)

    if len(pos_args) == 1:
        if var_pos_arg:
            # If there is only an '*args' parameter, we can just return it directly
            pos_args_tuple_expr = var_pos_arg
        else:
            pos_args_tuple_expr = f"({pos_args[0]},)"  # Add trailing comma to make it a tuple
    else:
        # In the regular case, assemble the output tuple with all positional arguments
        pos_args_tuple_expr = f"({str.join(', ', pos_args)})"

    if len(key_args) == 1 and var_key_arg:
        # If there is only a '**kwargs' parameter, we can just return it directly
        key_args_dict_expr = var_key_arg
    else:
        # In the regular case, assemble the output dictionary with all keyword arguments
        key_args_dict_expr = f"{{ {str.join(', ', key_args)} }}"

    canonicalize_func_name = f"canonicalize_args_for_{name}" if name else "canonicalize_args"
    canonicalizer_src = f"""
def {canonicalize_func_name}{canonicalizer_signature!s}:
    return {pos_args_tuple_expr}, {key_args_dict_expr}
"""
    namespace: dict[str, Any] = {}
    exec(canonicalizer_src, namespace)
    canonicalizer = namespace[canonicalize_func_name]

    return cast(Callable[..., tuple[tuple, dict[str, Any]]], canonicalizer)


@functools.cache
def make_args_canonicalizer_for_function(
    func: types.FunctionType,
) -> Callable[..., tuple[tuple, dict[str, Any]]]:
    return make_args_canonicalizer(inspect.signature(func), name=func.__name__)


def canonicalize_call_args(
    func: Callable,
    /,
    args: tuple,
    kwargs: dict[str, Any],
) -> tuple[tuple, dict[str, Any]]:
    """
    Canonicalize call arguments for a given function.

    Args:
        func: The function for which to canonicalize the call arguments.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        A tuple of positional arguments and a dictionary with keyword arguments.

    Note:
        This is a convenience wrapper around `make_args_canonicalizer_for_function`.
    """

    return make_args_canonicalizer_for_function(func)(*args, **kwargs)


class IDGeneratorPool(eve_utils.CustomDefaultDictBase):
    """
    Utility for providing unique IDs for each prefix.

    The use-case for this implementation is to provide a single IDGeneratorPool for the whole
    transformation pipeline, which ensure unique names (prefix+counter) across all transformations.

    The reason for this implementation of a single counter per prefix (over a global increasing counter)
    is that parts of the IR might still be stable when comparing different versions of a transformation.

    >>> uids = IDGeneratorPool()
    >>> next(uids["foo"])
    'foo_0'
    >>> next(uids["foo"])
    'foo_1'
    >>> next(uids["bar"])
    'bar_0'
    """

    def value_factory(self, prefix: str) -> eve_utils.SequentialIDGenerator:
        return eve_utils.SequentialIDGenerator(prefix=prefix)
