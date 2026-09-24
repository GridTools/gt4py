# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import collections
import copyreg
import dataclasses
import enum
import functools
import importlib
import math
import numbers
import re
import sys
import types
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping, utils
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Literal,
    NamedTuple,
    Never,
    NoReturn,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    overload,
    runtime_checkable,
)
from gt4py.eve.type_definitions import StrEnum


DimT = TypeVar("DimT", bound="Dimension")  # , covariant=True)
DimT_co = TypeVar("DimT_co", bound="Dimension", covariant=True)
ShapeTs = TypeVarTuple("ShapeTs")


class Dims(tuple[Unpack[ShapeTs]]): ...


DimsT = TypeVar("DimsT", bound=Dims, covariant=True)

Tag: TypeAlias = str


_CODEGEN_UNESCAPE: Final = {"u": "_", "d": ".", "l": "[", "r": "]"}


def codegen_name(tag: Tag) -> str:
    """
    Mangle a dimension or offset tag into a valid generated identifier.

    A tag is a qualified Python name, so it contains dots, which are illegal in a C++
    identifier, in a DaCe symbol, and in `eve`'s `SymbolName` (`^[a-zA-Z_]\\w*$`). Since a
    generated identifier may only contain `[A-Za-z0-9_]`, the underscore is the only
    available separator, and escaping it is what makes the mapping reversible.

    The escape is a *prefix* escape. The obvious alternative -- double every underscore,
    then turn dots into single underscores -- is **not injective**: a dot becomes a single
    underscore, so `'..'` and `'_'` both map to `'__'`.

    Args:
        tag: A dimension or offset tag, i.e. a qualified Python name.

    Returns:
        A valid identifier, unique for each distinct `tag`.

    Examples:
        >>> codegen_name("mod.V2E.Local")
        'mod_dV2E_dLocal'
        >>> codegen_name("a_b.c")
        'a_ub_dc'
        >>> from_codegen_name(codegen_name("my__mod.X"))
        'my__mod.X'
    """
    # NOTE: `_` first, so the underscores introduced by the other escapes are not re-escaped.
    # A tag's alphabet is `[A-Za-z0-9_.[]]`: brackets come from a parametrized tag such as
    # `Staggered[pkg.K]`, and would otherwise survive into the identifier.
    return tag.replace("_", "_u").replace(".", "_d").replace("[", "_l").replace("]", "_r")


def from_codegen_name(name: str) -> Tag:
    """
    Recover a tag from the identifier `codegen_name` produced for it.

    Needed wherever a backend parses a generated name back into the dimension or offset it
    refers to.

    Args:
        name: An identifier produced by `codegen_name`.

    Returns:
        The original tag.

    Examples:
        >>> from_codegen_name("mod_dV2E_dLocal")
        'mod.V2E.Local'
    """
    return re.sub(r"_([udlr])", lambda m: _CODEGEN_UNESCAPE[m.group(1)], name)


@enum.unique
class DimensionKind(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LOCAL = "local"

    def __str__(self) -> str:
        return self.value


_DIM_KIND_ORDER = {DimensionKind.HORIZONTAL: 0, DimensionKind.LOCAL: 1, DimensionKind.VERTICAL: 2}


class DimensionMeta(type):
    """
    Metaclass of all dimension classes.

    Holds the behaviour that used to live on `Dimension` *instances*, but on the class
    object itself. Binary operators applied to a class object dispatch through its
    metaclass, so this is the only place they can live.
    """

    kind: DimensionKind

    # NOTE: mandatory, not redundant. Python sets `__hash__ = None` on any class body that
    # defines `__eq__` without it -- metaclasses included -- and `__eq__` below stays for the
    # `I == 5` overload. Without this every dimension class is unhashable, which breaks
    # `domain({I: 2})`, dimension-keyed dicts, and eve's validator memoisation on annotation
    # objects (so `ts.DimensionType` would fail at import).
    __hash__ = type.__hash__

    @property
    def tag(cls) -> Tag:
        """
        The dimension's identity: its qualified Python name, and its spelling in the IR.

        A property rather than a settable attribute, so it cannot drift from the type it
        names. Use `__qualname__` for display; see `__str__`.
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def value(cls) -> NoReturn:
        """
        Reject `SomeDim.value`, which used to be the dimension's name and is now `tag`.

        Without this the read silently returns the `value` slot descriptor of the *instance*
        attribute rather than raising, and the nonsense value only surfaces much later -- as
        a missing offset-provider key, or an `AxisLiteral` validation failure. Instance
        access (`SomeDim(0).value`) is unaffected: a metaclass attribute is not on an
        instance's lookup path.
        """
        raise AttributeError(
            f"'{cls.__qualname__}' is a dimension and has no 'value': its name is '.tag',"
            f" and an *index* into it -- '{cls.__qualname__}(0)' -- is what has '.value'."
        )

    def __repr__(cls) -> str:
        return f"{cls.tag}[{cls.kind}]"

    def __str__(cls) -> str:
        # NOTE: the unqualified name, so diagnostics stay readable. `tag` is identity, not a
        # display name; `repr` carries the module and disambiguates when it matters.
        return f"{cls.__qualname__}[{cls.kind}]"

    def __add__(cls: Dimension, offset: int | float) -> Connectivity:  # type: ignore[misc]
        return connectivity_for_cartesian_shift(cls, offset)

    def __sub__(cls: Dimension, offset: int | float) -> Connectivity:  # type: ignore[misc]
        return cls + (-offset)

    def __gt__(cls: Dimension, value: core_defs.IntegralScalar) -> Domain:  # type: ignore[misc]
        return Domain(dims=(cls,), ranges=(UnitRange(value + 1, Infinity.POSITIVE),))

    def __ge__(cls: Dimension, value: core_defs.IntegralScalar) -> Domain:  # type: ignore[misc]
        return Domain(dims=(cls,), ranges=(UnitRange(value, Infinity.POSITIVE),))

    def __lt__(cls: Dimension, value: core_defs.IntegralScalar) -> Domain:  # type: ignore[misc]
        return Domain(dims=(cls,), ranges=(UnitRange(Infinity.NEGATIVE, value),))

    def __le__(cls: Dimension, value: core_defs.IntegralScalar) -> Domain:  # type: ignore[misc]
        return Domain(dims=(cls,), ranges=(UnitRange(Infinity.NEGATIVE, value + 1),))

    @overload  # type: ignore[override]  # incompatible with `type.__eq__`, which returns `bool`.
    def __eq__(cls, value: DimensionMeta) -> bool: ...
    @overload
    def __eq__(cls, value: core_defs.IntegralScalar) -> Domain: ...
    def __eq__(  # type: ignore[misc]
        cls: Dimension, value: DimensionMeta | core_defs.IntegralScalar
    ) -> bool | Domain:
        # NOTE: dimension-vs-dimension comparison is deliberately *not* handled here. A
        # dimension's identity is its type, so `type.__eq__` (identity) is the correct
        # answer; overriding it with `(tag, kind)` equality is what ADR 0028 rejects.
        if isinstance(value, DimensionMeta):
            return NotImplemented  # both sides decline, so Python falls back to identity
        if isinstance(value, core_defs.INTEGRAL_TYPES):
            return Domain(dims=(cls,), ranges=(UnitRange(value, value + 1),))
        return NotImplemented

    @overload  # type: ignore[override]  # incompatible with `type.__ne__`, which returns `bool`.
    def __ne__(cls, value: DimensionMeta) -> bool: ...
    @overload
    def __ne__(cls, value: core_defs.IntegralScalar) -> Domain: ...
    def __ne__(  # type: ignore[misc]
        cls: Dimension, value: DimensionMeta | core_defs.IntegralScalar
    ) -> bool | Domain:
        if isinstance(value, core_defs.INTEGRAL_TYPES):
            raise NotImplementedError(
                "'Dimension.__ne__' with an integer value produces two disjoint domains, "
                "which is not supported. Use 'concat_where(dim < value, ...) "
                "concat_where(dim > value, ...)' to express the condition, see ADR 22."
            )
        return NotImplemented


class DimensionIndex(metaclass=DimensionMeta):
    """
    A dimension. A concrete dimension is a *subclass*; an index along it an *instance*.

    This is the shape `enum.Enum` uses: the class is the collection, the instances are its
    members. A dimension's identity is its type, and `tag` -- its qualified Python name --
    is how it is spelled in the IR. `value` is an index position along it.

    Examples:
        >>> class I(DimensionIndex): ...
        >>> class K(DimensionIndex, kind=DimensionKind.VERTICAL): ...
        >>> str(I), K.kind
        ('I[horizontal]', <DimensionKind.VERTICAL: 'vertical'>)

        >>> I(0)
        I=0
        >>> I(0).dim is I, I(0).value
        (True, 0)

        Two dimension classes are the same dimension only if they are the same class:

        >>> class I2(DimensionIndex): ...
        >>> I == I2
        False
    """

    kind: ClassVar[DimensionKind] = DimensionKind.HORIZONTAL

    __slots__ = ("value",)

    #: Index position along the dimension. The dimension's *name* is `tag`, on the class.
    value: int

    def __init_subclass__(cls, /, kind: Optional[DimensionKind] = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "tag" in cls.__dict__:
            raise TypeError(
                f"'{cls.__qualname__}' sets 'tag' in its class body, which has no effect:"
                " a dimension's tag is its qualified Python name. Rename the class instead."
            )
        if "<locals>" in cls.__qualname__:
            raise TypeError(
                f"'{cls.__qualname__}' must be declared at module level: a dimension is"
                " referenced from the IR by its qualified name, which has to be importable."
            )
        if kind is not None:
            cls.kind = kind
        if cls.kind is DimensionKind.LOCAL and not any(
            "_local_dimension_root" in base.__dict__ for base in cls.__mro__
        ):
            raise TypeError(
                f"'{cls.__qualname__}': a local dimension is declared by subclassing"
                " 'LocalDimensionIndex', or as the nested 'Local' class of a"
                " 'NeighborConnectivity', not with 'kind=DimensionKind.LOCAL'."
            )

    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}={self.value}"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DimensionIndex):
            # NOTE: `is`, not `==`: a dimension's identity is its type (ADR 0028).
            return type(self) is type(other) and self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    @property
    def dim(self) -> Dimension:
        """The dimension this index runs along, i.e. its own class."""
        return type(self)


#: A concrete dimension, i.e. the *class* itself rather than an index into it.
#:
#: NOTE: a PEP 695 `type` statement, not a plain `TypeAlias`, so that the removed
#: `Dimension("I")` spelling fails loudly. A plain alias for `type[X]` is a
#: `types.GenericAlias`, and calling one forwards to its `__origin__` while discarding the
#: arguments -- so `Dimension("I")` would evaluate to `type("I")`, i.e. `str`, with no error
#: at all. A `TypeAliasType` is simply not callable.
#:
#: The cost is that `get_origin()` of a PEP 695 alias is `None` rather than the aliased
#: origin, so a site dispatching on an annotation's shape must resolve it first (see
#: `xtyping.resolve_annotation`). `eve.datamodels` stores annotations *unresolved*, so this
#: applies to anything reading `__datamodel_fields__[...].type` too. See #2841 and ADR 0028.
type Dimension = type[DimensionIndex]


_STAGGERED_TAG_RE: Final = re.compile(r"^(?P<owner>[^\[\]]+)\[(?P<base>.+)\]$")


def staggered_base_tag(tag: Tag) -> Optional[Tag]:
    """
    Return the base dimension's tag if `tag` names a staggered dimension, else `None`.

    Reads the `<owner>[<base>]` grammar that `Staggered[D]` produces and `resolve` parses, so it
    works on a bare tag without importing anything. That matters where tags of dimensions and
    of offsets are mixed in one collection: an offset tag is not a dimension and cannot be
    resolved, but it simply does not match.

    Examples:
        >>> staggered_base_tag("gt4py.next.common.Staggered[pkg.KDim]")
        'pkg.KDim'
        >>> staggered_base_tag("pkg.KDim") is None
        True
    """
    return match["base"] if (match := _STAGGERED_TAG_RE.match(tag)) is not None else None


def resolve(tag: Tag) -> Dimension:
    """
    Return the dimension class a tag names, by importing it.

    The counterpart of `DimensionMeta.tag`, for the IR boundaries that rebuild a dimension
    from its name. A tag is a qualified Python name, so this is an import followed by an
    attribute walk -- the same way `pickle` references a class.

    A purely dotted tag does not record where the module path ends and the qualname begins,
    so the longest importable prefix wins and the remainder is walked as attributes. A
    collision would need a module path and an attribute chain to have the same spelling.

    Only where the module path ends is memoized; the attribute walk is repeated on every call, so
    a declaration redefined under the same name (e.g. by re-running a notebook cell) resolves to
    the new class.

    Parametrized dimensions such as `Staggered[K]` have no importable qualname; their tag
    has the form `<owner tag>[<base tag>]` and is resolved by subscripting the owner, which
    goes through its intern table and so returns the identical class.

    Args:
        tag: A dimension tag, as produced by `DimensionMeta.tag`.

    Returns:
        The dimension class.

    Raises:
        ValueError: If no prefix of `tag` is importable, or the attribute walk fails.

    Examples:
        >>> resolve("gt4py.next.common.DimensionIndex") is DimensionIndex
        True
    """
    if (match := _STAGGERED_TAG_RE.match(tag)) is not None:
        owner = resolve(match["owner"])
        if not isinstance(owner, StaggeredMeta):
            raise ValueError(
                f"Cannot resolve tag '{tag}': '{match['owner']}' is not a parametrized dimension."
            )
        return owner[resolve(match["base"])]  # type: ignore[index] # a StaggeredMeta, checked

    obj = _import_qualified_name(tag)
    if not isinstance(obj, DimensionMeta):
        raise ValueError(f"Tag '{tag}' resolves to '{obj}', which is not a dimension.")
    return cast(Dimension, obj)


def _import_qualified_name_or_none(tag: Tag) -> Any:
    """`_import_qualified_name`, returning `None` for a name that does not resolve."""
    if (split := _split_qualified_name_or_none(tag)) is None:
        return None
    module_name, attrs = split
    obj: Any = sys.modules.get(module_name) or importlib.import_module(module_name)
    for attr in attrs:
        if (obj := getattr(obj, attr, None)) is None:
            return None
    return obj


def resolve_loaded(tag: Tag) -> Optional[Dimension]:
    """
    Return the dimension a tag names if its module is already loaded, else `None`.

    Like `resolve`, but never imports: for code that must not have import side effects, such as
    printing IR.
    """
    if (match := _STAGGERED_TAG_RE.match(tag)) is not None:
        owner, base = resolve_loaded(match["owner"]), resolve_loaded(match["base"])
        return owner[base] if owner is not None and base is not None else None  # type: ignore[index] # parametrized dimension
    parts = tag.split(".")
    for split in range(len(parts) - 1, 0, -1):
        if (obj := sys.modules.get(".".join(parts[:split]))) is None:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
        return obj if isinstance(obj, DimensionMeta) else None
    return None


def _import_qualified_name(tag: Tag) -> Any:
    """Import the object a dotted qualified name refers to; see `resolve`."""
    module_name, attrs = _split_qualified_name(tag)
    obj: Any = sys.modules.get(module_name) or importlib.import_module(module_name)
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError as ex:
            raise ValueError(
                f"Cannot resolve tag '{tag}': '{module_name}' has no attribute '{'.'.join(attrs)}'."
            ) from ex
    return obj


@functools.cache
def _split_qualified_name_or_none(tag: Tag) -> Optional[tuple[str, tuple[str, ...]]]:
    """
    `_split_qualified_name`, returning `None` instead of raising.

    Separate and memoized so that a string that is not a qualified name -- an offset-provider key
    of hand-written IR, say -- costs one import attempt in total, not one per call. A module whose
    import *fails* other than by not being found is reported, not cached away.
    """
    try:
        return _split_qualified_name(tag)
    except ValueError:
        return None


@functools.cache
def _split_qualified_name(tag: Tag) -> tuple[str, tuple[str, ...]]:
    """Split a dotted name at its longest importable module prefix: `(module, attributes)`."""
    parts = tag.split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return module_name, tuple(parts[split:])
    raise ValueError(
        f"Cannot resolve tag '{tag}': no importable module prefix. A dimension or connectivity"
        " referenced from the IR must be declared at module level in an importable module."
    )


class Infinity(enum.Enum):
    """Describes an unbounded `UnitRange`."""

    NEGATIVE = enum.auto()
    POSITIVE = enum.auto()

    def __add__(self, _: int) -> Self:
        return self

    __radd__ = __add__

    def __sub__(self, _: int) -> Self:
        return self

    __rsub__ = __sub__

    def __le__(self, other: int | Infinity) -> bool:
        return self is self.NEGATIVE or other is self.POSITIVE

    def __lt__(self, other: int | Infinity) -> bool:
        return self is self.NEGATIVE and other is not self

    def __ge__(self, other: int | Infinity) -> bool:
        return self is self.POSITIVE or other is self.NEGATIVE

    def __gt__(self, other: int | Infinity) -> bool:
        return self is self.POSITIVE and other is not self


def _as_int(v: core_defs.IntegralScalar | Infinity) -> int | Infinity:
    return v if isinstance(v, Infinity) else int(v)


_Left = TypeVar("_Left", int, Infinity)
_Right = TypeVar("_Right", int, Infinity)


@dataclasses.dataclass(frozen=True, init=False)
class UnitRange(Sequence[int], Generic[_Left, _Right]):
    """Range from `start` to `stop` with step size one."""

    start: _Left
    stop: _Right

    def __init__(
        self, start: core_defs.IntegralScalar | Infinity, stop: core_defs.IntegralScalar | Infinity
    ) -> None:
        if start < stop:
            object.__setattr__(self, "start", _as_int(start))
            object.__setattr__(self, "stop", _as_int(stop))
        else:
            # make UnitRange(0,0) the single empty UnitRange
            object.__setattr__(self, "start", 0)
            object.__setattr__(self, "stop", 0)

    @classmethod
    def infinite(cls) -> UnitRange:
        return cls(Infinity.NEGATIVE, Infinity.POSITIVE)

    def __len__(self) -> int:
        if UnitRange.is_finite(self):
            return max(0, self.stop - self.start)
        raise ValueError("Cannot compute length of open 'UnitRange'.")

    @classmethod
    def is_finite(cls, obj: UnitRange) -> TypeGuard[FiniteUnitRange]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.start is not Infinity.NEGATIVE and obj.stop is not Infinity.POSITIVE

    @classmethod
    def is_right_finite(cls, obj: UnitRange) -> TypeGuard[UnitRange[_Left, int]]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.stop is not Infinity.POSITIVE

    @classmethod
    def is_left_finite(cls, obj: UnitRange) -> TypeGuard[UnitRange[int, _Right]]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return obj.start is not Infinity.NEGATIVE

    def is_empty(self) -> bool:
        return (
            self.start == 0 and self.stop == 0
        )  # post_init ensures that empty is represented as UnitRange(0, 0)

    def __repr__(self) -> str:
        return f"UnitRange({self.start}, {self.stop})"

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> UnitRange: ...

    def __getitem__(self, index: int | slice) -> int | UnitRange:
        assert UnitRange.is_finite(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("'UnitRange': step required to be '1'.")
            new_start = self.start + (start or 0)
            new_stop = (self.start if stop > 0 else self.stop) + stop
            return UnitRange(new_start, new_stop)
        else:
            if index < 0:
                index += len(self)

            if 0 <= index < len(self):
                return self.start + index
            else:
                raise IndexError("'UnitRange' index out of range")

    def __and__(self, other: UnitRange) -> UnitRange:
        return UnitRange(max(self.start, other.start), min(self.stop, other.stop))

    def __contains__(self, value: Any) -> bool:
        # TODO(egparedes): use core_defs.IntegralScalar for `isinstance()` checks (see PEP 604)
        #   and remove int cast, once the related mypy bug (#16358) gets fixed
        if isinstance(value, core_defs.INTEGRAL_TYPES):
            return self.start <= cast(int, value) < self.stop
        else:
            return False

    def __le__(self, other: UnitRange) -> bool:
        return self.start >= other.start and self.stop <= other.stop

    def __lt__(self, other: UnitRange) -> bool:
        return (self.start > other.start and self.stop <= other.stop) or (
            self.start >= other.start and self.stop < other.stop
        )

    def __ge__(self, other: UnitRange) -> bool:
        return self.start <= other.start and self.stop >= other.stop

    def __gt__(self, other: UnitRange) -> bool:
        return (self.start < other.start and self.stop >= other.stop) or (
            self.start <= other.start and self.stop > other.stop
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UnitRange):
            return self.start == other.start and self.stop == other.stop
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __add__(self, other: int) -> UnitRange:
        return UnitRange(self.start + other, self.stop + other)

    def __sub__(self, other: int) -> UnitRange:
        return UnitRange(self.start - other, self.stop - other)

    def __str__(self) -> str:
        return f"({self.start}:{self.stop})"


FiniteUnitRange: TypeAlias = UnitRange[int, int]

_Rng = TypeVar(
    "_Rng",
    FiniteUnitRange,
    UnitRange[Infinity, int],
    UnitRange[int, Infinity],
    UnitRange[Infinity, Infinity],
)

RangeLike: TypeAlias = (
    _Rng
    | range
    | tuple[core_defs.IntegralScalar, core_defs.IntegralScalar]
    | core_defs.IntegralScalar
    | None
)


def unit_range(r: RangeLike) -> UnitRange:
    if isinstance(r, UnitRange):
        return r
    if isinstance(r, range):
        if r.step != 1:
            raise ValueError(f"'UnitRange' requires step size 1, got '{r.step}'.")
        return UnitRange(r.start, r.stop)
    # TODO(egparedes): use core_defs.IntegralScalar for `isinstance()` checks (see PEP 604)
    #   once the related mypy bug (#16358) gets fixed
    if (
        isinstance(r, tuple)
        and (isinstance(r[0], core_defs.INTEGRAL_TYPES) or r[0] in (None, Infinity.NEGATIVE))
        and (isinstance(r[1], core_defs.INTEGRAL_TYPES) or r[1] in (None, Infinity.POSITIVE))
    ):
        start = r[0] if r[0] is not None else Infinity.NEGATIVE
        stop = r[1] if r[1] is not None else Infinity.POSITIVE
        return UnitRange(start, stop)
    if isinstance(r, core_defs.INTEGRAL_TYPES):
        return UnitRange(0, cast(core_defs.IntegralScalar, r))
    if r is None:
        return UnitRange.infinite()
    raise ValueError(f"'{r!r}' cannot be interpreted as 'UnitRange'.")


class NamedRange(NamedTuple, Generic[_Rng]):
    dim: Dimension
    unit_range: _Rng

    def __str__(self) -> str:
        return f"{self.dim}={self.unit_range}"


IntIndex: TypeAlias = int | core_defs.IntegralScalar


FiniteNamedRange: TypeAlias = NamedRange[FiniteUnitRange]
RelativeIndexElement: TypeAlias = IntIndex | slice | types.EllipsisType
NamedSlice: TypeAlias = slice  # once slice is generic we should do: slice[DimensionIndex, DimensionIndex, Literal[1]], see https://peps.python.org/pep-0696/
AbsoluteIndexElement: TypeAlias = DimensionIndex | NamedRange | NamedSlice
AnyIndexElement: TypeAlias = RelativeIndexElement | AbsoluteIndexElement
AbsoluteIndexSequence: TypeAlias = Sequence[NamedRange | DimensionIndex]
RelativeIndexSequence: TypeAlias = tuple[
    slice | IntIndex | types.EllipsisType, ...
]  # is a tuple but called Sequence for symmetry
AnyIndexSequence: TypeAlias = RelativeIndexSequence | AbsoluteIndexSequence
AnyIndexSpec: TypeAlias = AnyIndexElement | AnyIndexSequence


def is_int_index(p: Any) -> TypeGuard[IntIndex]:
    # should be replaced by isinstance(p, IntIndex), but mypy complains with
    # `Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]`
    return isinstance(p, (int, core_defs.INTEGRAL_TYPES))


def is_finite_named_range(v: NamedRange) -> TypeGuard[FiniteNamedRange]:
    return UnitRange.is_finite(v.unit_range)


def is_named_slice(obj: AnyIndexSpec) -> TypeGuard[slice]:
    return isinstance(obj, slice) and (
        isinstance(obj.start, DimensionIndex) and isinstance(obj.stop, DimensionIndex)
    )


def is_any_index_element(v: AnyIndexSpec) -> TypeGuard[AnyIndexElement]:
    return is_int_index(v) or isinstance(v, (NamedRange, DimensionIndex, slice)) or v is Ellipsis


def is_absolute_index_sequence(v: AnyIndexSequence) -> TypeGuard[AbsoluteIndexSequence]:
    return isinstance(v, Sequence) and all(isinstance(e, (NamedRange, DimensionIndex)) for e in v)


def is_relative_index_sequence(v: AnyIndexSequence) -> TypeGuard[RelativeIndexSequence]:
    return isinstance(v, tuple) and all(
        isinstance(e, slice) or is_int_index(e) or e is Ellipsis for e in v
    )


def as_any_index_sequence(index: AnyIndexSpec) -> AnyIndexSequence:
    # `cast` because mypy/typing doesn't special case 1-element tuples, i.e. `tuple[A|B] != tuple[A]|tuple[B]`
    return cast(AnyIndexSequence, (index,) if is_any_index_element(index) else index)


def named_range(v: tuple[Dimension, RangeLike]) -> NamedRange:
    if isinstance(v, NamedRange):
        return v
    return NamedRange(v[0], unit_range(v[1]))


@dataclasses.dataclass(frozen=True, init=False)
class Domain(Sequence[NamedRange[_Rng]], Generic[_Rng]):
    """Describes the `Domain` of a `Field` as a `Sequence` of `NamedRange` s."""

    dims: tuple[Dimension, ...]
    ranges: tuple[_Rng, ...]

    def __init__(
        self,
        *args: NamedRange[_Rng],
        dims: Optional[Sequence[Dimension]] = None,
        ranges: Optional[Sequence[_Rng]] = None,
    ) -> None:
        if dims is not None or ranges is not None:
            if dims is None and ranges is None:
                raise ValueError("Either specify both 'dims' and 'ranges' or neither.")
            if len(args) > 0:
                raise ValueError(
                    "No extra 'args' allowed when constructing from 'dims' and 'ranges'."
                )

            assert dims is not None and ranges is not None  # for mypy
            if not all(isinstance(dim, DimensionMeta) for dim in dims):
                raise ValueError(
                    f"'dims' argument needs to be a 'tuple[Dimension, ...]', got '{dims}'."
                )
            if not all(isinstance(rng, UnitRange) for rng in ranges):
                raise ValueError(
                    f"'ranges' argument needs to be a 'tuple[UnitRange, ...]', got '{ranges}'."
                )
            if len(dims) != len(ranges):
                raise ValueError(
                    f"Number of provided dimensions ({len(dims)}) does not match number of provided ranges ({len(ranges)})."
                )

            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))
        else:
            if not all(isinstance(arg, NamedRange) for arg in args):
                raise ValueError(
                    f"Elements of 'Domain' need to be instances of 'NamedRange', got '{args}'."
                )
            dims, ranges = zip(*args) if args else ((), ())
            object.__setattr__(self, "dims", tuple(dims))
            object.__setattr__(self, "ranges", tuple(ranges))

        if len(set(self.dims)) != len(self.dims):
            raise NotImplementedError(f"Domain dimensions must be unique, not '{self.dims}'.")

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(r) for r in self.ranges)

    @property
    def size(self) -> Optional[int]:
        return math.prod(self.shape) if all(UnitRange.is_finite(r) for r in self.ranges) else None

    def __len__(self) -> int:
        return len(self.ranges)

    def __str__(self) -> str:
        return f"Domain({', '.join(f'{e}' for e in self)})"

    @overload
    def __getitem__(self, index: int) -> NamedRange: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Dimension) -> NamedRange: ...

    def __getitem__(self, index: int | slice | Dimension) -> NamedRange | Domain:
        if isinstance(index, DimensionMeta):
            try:
                index = self.dims.index(index)
            except ValueError as ex:
                raise KeyError(f"No Dimension of type '{index}' is present in the Domain.") from ex
        if isinstance(index, int):
            return NamedRange(dim=self.dims[index], unit_range=self.ranges[index])
        if isinstance(index, slice):
            dims_slice = self.dims[index]
            ranges_slice = self.ranges[index]
            return Domain(dims=dims_slice, ranges=ranges_slice)

        raise KeyError("Invalid index type, must be either int, slice, or Dimension.")

    def __and__(self, other: Domain) -> Domain:
        """
        Intersect `Domain`s, missing `Dimension`s are considered infinite.

        Examples:
            >>> class I(DimensionIndex): ...
            >>> class J(DimensionIndex): ...

            >>> Domain(NamedRange(I, UnitRange(-1, 3))) & Domain(NamedRange(I, UnitRange(1, 6)))
            Domain(dims=(gt4py.next.common.I[horizontal],), ranges=(UnitRange(1, 3),))

            >>> Domain(NamedRange(I, UnitRange(-1, 3)), NamedRange(J, UnitRange(2, 4))) & Domain(
            ...     NamedRange(I, UnitRange(1, 6))
            ... )
            Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(1, 3), UnitRange(2, 4)))
        """
        broadcast_dims = tuple(promote_dims(self.dims, other.dims))
        intersected_ranges = tuple(
            rng1 & rng2
            for rng1, rng2 in zip(
                _broadcast_ranges(broadcast_dims, self.dims, self.ranges),
                _broadcast_ranges(broadcast_dims, other.dims, other.ranges),
            )
        )
        return Domain(dims=broadcast_dims, ranges=intersected_ranges)

    def __or__(self, other: Domain) -> Domain:
        """
        Union of `Domain`s, currently limited to 1D overlapping or adjacent domains.

        Raises `NotImplementedError` for multidimensional domains or disjoint 1D domains.
        See ADR 22.
        """
        if self.ndim > 1 or other.ndim > 1:
            raise NotImplementedError(
                "Union of multidimensional domains is not supported, see ADR 22."
            )
        if self.ndim == 0:
            return other
        if other.ndim == 0:
            return self
        if self.dims[0] != other.dims[0]:
            raise NotImplementedError(
                f"Union of 1D domains with different dimensions '{self.dims[0]}' and '{other.dims[0]}' is not supported."
            )
        first, second = sorted((self, other), key=lambda x: x.ranges[0].start)
        if first.ranges[0].stop >= second.ranges[0].start:
            return Domain(
                dims=(self.dims[0],),
                ranges=(UnitRange(first.ranges[0].start, second.ranges[0].stop),),
            )
        raise NotImplementedError(
            f"Union of disjoint domains '{first}' and '{second}' is not supported. "
            f"Use nested 'concat_where' to express non-contiguous conditions, see ADR 22."
        )

    @functools.cached_property
    def slice_at(self) -> utils.IndexerCallable[slice, Domain]:
        """
        Create a new domain by slicing the domain ranges at the provided relative slices.

        Examples:
            >>> class I(DimensionIndex): ...
            >>> class J(DimensionIndex): ...
            >>> domain = Domain(NamedRange(I, UnitRange(0, 10)), NamedRange(J, UnitRange(5, 15)))
            >>> domain.slice_at[2:3, 2:5]
            Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(2, 3), UnitRange(7, 10)))
        """

        def _domain_slicer(*args: slice) -> Domain:
            if not all(isinstance(a, slice) for a in args):
                raise TypeError(f"Indices must be 'slice's but got '{args}'")
            if len(args) != len(self):
                raise ValueError(
                    f"Number of provided slices ({len(args)}) does not match the number of dimensions ({len(self)})."
                )
            return Domain(dims=self.dims, ranges=[r[s] for r, s in zip(self.ranges, args)])

        return utils.IndexerCallable(_domain_slicer)

    @classmethod
    def is_finite(cls, obj: Domain) -> TypeGuard[FiniteDomain]:
        # classmethod since TypeGuards requires the guarded obj as separate argument
        return all(UnitRange.is_finite(rng) for rng in obj.ranges)

    def is_empty(self) -> bool:
        return any(rng.is_empty() for rng in self.ranges)

    @overload
    def dim_index(self, dim: Dimension, *, allow_missing: Literal[False]) -> int: ...

    @overload
    def dim_index(
        self, dim: Dimension, *, allow_missing: Literal[True] = True
    ) -> Optional[int]: ...

    def dim_index(self, dim: Dimension, *, allow_missing: bool = True) -> Optional[int]:
        if dim in self.dims:
            return self.dims.index(dim)
        elif allow_missing:
            return None
        else:
            raise ValueError(f"Dimension '{dim}' not found in Domain.")

    def pop(self, index: int | Dimension = -1) -> Domain:
        return self.replace(index)

    def insert(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        if isinstance(index, int) and index == len(self.dims):
            new_dims, new_ranges = zip(*named_ranges)
            return Domain(dims=self.dims + new_dims, ranges=self.ranges + new_ranges)
        else:
            return self.replace(index, *named_ranges)

    def replace(self, index: int | Dimension, *named_ranges: NamedRange) -> Domain:
        assert all(isinstance(nr, NamedRange) for nr in named_ranges)
        if isinstance(index, DimensionMeta):
            dim_index = self.dim_index(index)
            if dim_index is None:
                raise ValueError(f"Dimension '{index}' not found in Domain.")
            index = dim_index
        if not (-len(self.dims) <= index < len(self.dims)):
            raise IndexError(
                f"Index '{index}' out of bounds for Domain of length {len(self.dims)}."
            )
        if index < 0:
            index += len(self.dims)
        new_dims = (arg.dim for arg in named_ranges) if len(named_ranges) > 0 else ()
        new_ranges = (arg.unit_range for arg in named_ranges) if len(named_ranges) > 0 else ()
        dims = self.dims[:index] + tuple(new_dims) + self.dims[index + 1 :]
        ranges = self.ranges[:index] + tuple(new_ranges) + self.ranges[index + 1 :]

        return Domain(dims=dims, ranges=ranges)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # remove cached property
        state.pop("slice_at", None)
        return state


FiniteDomain: TypeAlias = Domain[FiniteUnitRange]


DomainLike: TypeAlias = (
    Sequence[tuple[Dimension, RangeLike]] | Mapping[Dimension, RangeLike]
)  # `Domain` is `Sequence[NamedRange]` and therefore a subset


def domain(domain_like: DomainLike) -> Domain:
    """
    Construct `Domain` from `DomainLike` object.

    Examples:
        >>> class I(DimensionIndex): ...
        >>> class J(DimensionIndex): ...

        >>> domain(((I, (2, 4)), (J, (3, 5))))
        Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(2, 4), UnitRange(3, 5)))

        >>> domain({I: (2, 4), J: (3, 5)})
        Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(2, 4), UnitRange(3, 5)))

        >>> domain(((I, 2), (J, 4)))
        Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(0, 2), UnitRange(0, 4)))

        >>> domain({I: 2, J: 4})
        Domain(dims=(gt4py.next.common.I[horizontal], gt4py.next.common.J[horizontal]), ranges=(UnitRange(0, 2), UnitRange(0, 4)))
    """
    if isinstance(domain_like, Domain):
        return domain_like
    if isinstance(domain_like, Sequence):
        return Domain(*tuple(named_range(d) for d in domain_like))
    if isinstance(domain_like, Mapping):
        if all(isinstance(elem, core_defs.INTEGRAL_TYPES) for elem in domain_like.values()):
            return Domain(
                dims=tuple(domain_like.keys()),
                ranges=tuple(UnitRange(0, s) for s in domain_like.values()),
            )
        return Domain(
            dims=tuple(domain_like.keys()),
            ranges=tuple(unit_range(r) for r in domain_like.values()),
        )
    raise ValueError(f"'{domain_like}' is not 'DomainLike'.")


def _broadcast_ranges(
    broadcast_dims: Sequence[Dimension], dims: Sequence[Dimension], ranges: Sequence[UnitRange]
) -> tuple[UnitRange, ...]:
    return tuple(
        ranges[dims.index(d)] if d in dims else UnitRange.infinite() for d in broadcast_dims
    )


if TYPE_CHECKING:
    import gt4py.next.ffront.fbuiltins as fbuiltins

    _Value: TypeAlias = "Field" | core_defs.ScalarT
    _P = ParamSpec("_P")
    _R = TypeVar("_R", _Value, tuple[_Value, ...])

    class GTBuiltInFuncDispatcher(Protocol):
        def __call__(self, /, func: fbuiltins.BuiltInFunction[_R, _P]) -> Callable[_P, _R]: ...


# TODO(havogt): we need to describe when this interface should be used instead of the `Field` protocol.
class GTFieldInterface(core_defs.GTDimsInterface, core_defs.GTOriginInterface, Protocol):
    """
    Protocol for object providing the `__gt_domain__` property, specifying the :class:`Domain` of a :class:`Field`.

    Note:
    - A default implementation of the `__gt_dims__` interface from `gt4py.cartesian` is provided.
    - No implementation of `__gt_origin__` is provided because of infinite fields.
    """

    @property
    def __gt_domain__(self) -> Domain:
        # TODO probably should be changed to `DomainLike` (with a new concept `DimensionLike`)
        # to allow implementations without having to import gtx.Domain.
        ...

    @property
    def __gt_dims__(self) -> tuple[str, ...]:
        # NOTE: the unqualified name, not the `tag`. This is the interop protocol with
        # `gt4py.cartesian`, which identifies axes by their bare names (`"I"`, `"J"`, `"K"`); a
        # qualified tag would not match and the axes would be transposed wrongly (ADR 0028).
        return tuple(d.__qualname__ for d in self.__gt_domain__.dims)


@runtime_checkable
class Field(GTFieldInterface, Protocol[DimsT, core_defs.ScalarT]):
    __gt_builtin_func__: ClassVar[GTBuiltInFuncDispatcher]

    @property
    def domain(self) -> Domain: ...

    @property
    def __gt_domain__(self) -> Domain:
        return self.domain

    @property
    def codomain(self) -> type[core_defs.ScalarT] | Dimension: ...

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]: ...

    # TODO(havogt)
    # This property is wrong, because for a function field we would not know to which NDArrayObject we want to convert
    # at the very least, we need to take an allocator and rename this to `as_ndarray`.
    @property
    def ndarray(self) -> core_defs.NDArrayObject: ...

    def __str__(self) -> str:
        return f"⟨{self.domain!s} → {self.dtype}⟩"

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "The truth value of a Field is ambiguous. For one element Fields use '.as_scalar()'."
        )

    @abc.abstractmethod
    def asnumpy(self) -> np.ndarray: ...

    @abc.abstractmethod
    def as_scalar(self) -> core_defs.ScalarT: ...

    @abc.abstractmethod
    def premap(self, index_field: Connectivity | type[NeighborConnectivity]) -> Field: ...

    @abc.abstractmethod
    def restrict(self, item: AnyIndexSpec) -> Self: ...
    # Operators
    @abc.abstractmethod
    def __call__(
        self,
        index_field: Connectivity | type[NeighborConnectivity],
        *args: Connectivity | type[NeighborConnectivity],
    ) -> Field: ...

    @abc.abstractmethod
    def __getitem__(self, item: AnyIndexSpec) -> Self: ...

    @abc.abstractmethod
    def __abs__(self) -> Field: ...

    @abc.abstractmethod
    def __neg__(self) -> Field: ...

    @abc.abstractmethod
    def __invert__(self) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __eq__(self, other: Any) -> Field:  # type: ignore[override] # mypy wants return `bool`
        ...

    @abc.abstractmethod
    def __ne__(self, other: Any) -> Field:  # type: ignore[override] # mypy wants return `bool`
        ...

    @abc.abstractmethod
    def __add__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __radd__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __sub__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rsub__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __mul__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rmul__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __floordiv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rfloordiv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __truediv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __rtruediv__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __pow__(self, other: Field | core_defs.ScalarT) -> Field: ...

    @abc.abstractmethod
    def __lt__(self, other: Field | core_defs.ScalarT) -> Field[Any, bool]: ...

    @abc.abstractmethod
    def __le__(self, other: Field | core_defs.ScalarT) -> Field[Any, bool]: ...

    @abc.abstractmethod
    def __gt__(self, other: Field | core_defs.ScalarT) -> Field[Any, bool]: ...

    @abc.abstractmethod
    def __ge__(self, other: Field | core_defs.ScalarT) -> Field[Any, bool]: ...

    @abc.abstractmethod
    def __and__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __or__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""

    @abc.abstractmethod
    def __xor__(self, other: Field | core_defs.ScalarT) -> Field:
        """Only defined for `Field` of value type `bool`."""


@runtime_checkable
class MutableField(Field[DimsT, core_defs.ScalarT], Protocol[DimsT, core_defs.ScalarT]):
    @abc.abstractmethod
    def __setitem__(self, index: AnyIndexSpec, value: Field | core_defs.ScalarT) -> None: ...


#: Type alias for primitive numeric values (i.e. scalars or fields).
NumericValue: TypeAlias = core_defs.Scalar | Field
NumericValueT = TypeVar("NumericValueT", bound=NumericValue)
NUMERIC_VALUE_TYPES: Final[tuple[type[NumericValue], ...]] = xtyping.get_represented_types(
    NumericValue
)

#: Type alias for any kind primitive value understood by GT4Py DSL.
PrimitiveValue: TypeAlias = NumericValue  # For now, only numeric values, in the future it could include functions, enums, ...
PRIMITIVE_VALUE_TYPES: Final[tuple[type[PrimitiveValue], ...]] = xtyping.get_represented_types(
    PrimitiveValue
)


@dataclasses.dataclass(frozen=True)
class BufferInfo:
    """Holds information about a buffer in memory."""

    data_ptr: int
    ndim: int
    shape: tuple[int, ...]
    elem_strides: tuple[int, ...]
    byte_strides: tuple[int, ...]
    device: core_defs.Device

    @classmethod
    def from_ndarray(cls, ndarray: core_defs.NDArrayObject) -> BufferInfo:
        # TODO(egparedes): Implement this function using __dlpack__ and ctypes.
        #   The current implementation is messy and only works for numpy and cupy.
        try:
            array_ns = ndarray.__array_namespace__()  # type: ignore[attr-defined]
        except AttributeError:
            array_ns = sys.modules[ndarray.__class__.__module__]

        array_byte_bounds_func = (
            getattr(array_ns, "byte_bounds", None) or array_ns.lib.array_utils.byte_bounds
        )

        data_ptr = array_byte_bounds_func(ndarray)[0]
        ndim = ndarray.ndim
        shape = ndarray.shape
        byte_strides = ndarray.strides
        elem_strides = tuple(s // ndarray.dtype.itemsize for s in byte_strides)

        try:
            device = core_defs.from_dlpack_device(ndarray.__dlpack_device__())  # type: ignore[attr-defined]
        except AttributeError as err:
            ns = ndarray.__class__.__module__
            if ns.startswith("numpy"):
                device = core_defs.Device(core_defs.DeviceType.CPU, 0)
            elif ns.startswith("cupy"):
                device = core_defs.Device(core_defs.CUPY_DEVICE_TYPE, ndarray.device.id)  # type: ignore[attr-defined]
            else:
                raise RuntimeError(f"Unsupported ndarray type '{type(ndarray)}'") from err

        return BufferInfo(
            data_ptr=data_ptr,
            ndim=ndim,
            shape=shape,
            elem_strides=elem_strides,
            byte_strides=byte_strides,
            device=device,
        )

    @functools.cached_property
    def hash_key(self) -> int:
        return hash(
            (
                self.data_ptr,
                self.ndim,
                self.shape,
                self.elem_strides,
                self.byte_strides,
                self.device,
            )
        )

    def __hash__(self) -> int:
        return self.hash_key


@dataclasses.dataclass(frozen=True)
class ConnectivityType:  # TODO(havogt): would better live in type_specifications but would have to solve a circular import
    domain: tuple[Dimension, ...]
    codomain: Dimension
    skip_value: Optional[core_defs.IntegralScalar]
    dtype: core_defs.DType

    @property
    def has_skip_values(self) -> bool:
        return self.skip_value is not None


@dataclasses.dataclass(frozen=True)
class NeighborTableType:
    """
    The type of a neighbor table bound to a connectivity: what transformations and code generation
    see instead of the table (ADR 0019).

    `connectivity` is the `NeighborConnectivity` declaration the table is bound to. It determines
    the table's `domain` -- the declaration's domain extended by its local dimension -- and its
    `codomain`. A table alone cannot name its declaration: one sharing another connectivity's
    local dimension has a table over the same domain as the owner's, with another codomain. So the
    record is built where a table is bound, from its offset-provider key (`offset_provider_to_type`,
    `check_neighbor_table`), or given directly for ahead-of-time compilation.

    A table bound under a name that no declaration answers to, as hand-written IR binds them, has
    no declaration: `connectivity` is then the table's own structural `ConnectivityType`, which is
    also what `NeighborTable.__gt_type__()` returns.
    """

    connectivity: type[NeighborConnectivity] | ConnectivityType
    dtype: core_defs.DType
    skip_value: Optional[core_defs.IntegralScalar]
    #: The table's number of entries per element. A declaration may leave it to the table; where
    #: it states one, `check_neighbor_table` checks the table against it.
    max_neighbors: int

    @property
    def domain(self) -> tuple[Dimension, Dimension]:
        if isinstance(self.connectivity, ConnectivityType):
            first, second = self.connectivity.domain
            return (first, second)
        return (self.connectivity.domain, local_dimension_of(self.connectivity))

    @property
    def codomain(self) -> Dimension:
        return self.connectivity.codomain

    @property
    def has_skip_values(self) -> bool:
        return self.skip_value is not None


@runtime_checkable
class Connectivity(Field[DimsT, core_defs.IntegralScalar], Protocol[DimsT, DimT_co]):
    @property
    @abc.abstractmethod
    def codomain(self) -> DimT_co:
        """
        The `codomain` is the set of all indices in a certain `Dimension`.

        We use the `Dimension` itself to describe the (infinite) set of all indices.

        Note:
        We could restrict the infinite codomain to only the indices that are actually contained in the mapping.
        Currently, this would just complicate implementation as we do not use this information.
        """

    def __gt_type__(self) -> ConnectivityType:
        # NOTE: structural, also for a neighbor table: the table cannot tell which declaration it
        # is bound to, so its `NeighborTableType` is built from its offset-provider key.
        return ConnectivityType(
            domain=self.domain.dims,
            codomain=self.codomain,
            dtype=self.dtype,
            skip_value=self.skip_value,
        )

    @abc.abstractmethod
    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]: ...

    @property
    @abc.abstractmethod
    def skip_value(self) -> Optional[core_defs.IntegralScalar]: ...

    # Operators
    def __abs__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __neg__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __invert__(self) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __eq__(self, other: Any) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __ne__(self, other: Any) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __add__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __radd__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __sub__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rsub__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __mul__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rmul__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __truediv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rtruediv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __floordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __rfloordiv__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __pow__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __lt__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __le__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __gt__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __ge__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __and__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __or__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")

    def __xor__(self, other: Field | core_defs.IntegralScalar) -> Never:
        raise TypeError("'Connectivity' does not support this operation.")


class GatherConnectivity(Connectivity[DimsT, DimT_co]):
    """A `Connectivity` whose `premap` rearranges data via advanced indexing (a gather).

    The defining contract is that the index map is materializable as an integer index table via
    `ndarray` (table-backed today; in principle a function evaluated over the domain). The gather
    algorithm (`embedded.nd_array_field._gather_premap`) is responsible for laying that table out
    over the output domain. Affine connectivities (cartesian shifts / relocations) are *not*
    `GatherConnectivity`: their `premap` is a compact domain relabel that moves no data and has no
    `ndarray`. The affine-vs-gather distinction has no structural witness (an affine connectivity
    still has an `ndarray` attribute, it just raises), so it is a nominal type, not a `Protocol`.
    """

    # TODO(havogt): This is a bare annotation, not an `@abc.abstractmethod`, on purpose. Making it
    #  abstract would force `NdArrayConnectivityField` abstract too: with `GatherConnectivity` ahead
    #  of `NdArrayField` in the MRO, the abstract `ndarray` shadows `NdArrayField`'s concrete one.
    #  Reordering the bases would fix that but then `NdArrayField`'s arithmetic operators would
    #  shadow `Connectivity`'s raising stubs (connectivities would start accepting `+` etc.). A bare
    #  annotation documents the contract for type checkers with no runtime effect.
    #  See also the TODO on `Field.ndarray`: ideally `ndarray` would not live on the abstract base.
    ndarray: core_defs.NDArrayObject


# Utility function to construct a `Field` from different buffer representations.
# Consider removing this function and using `Field` constructor directly. See also `_connectivity`.
@functools.singledispatch
def _field(
    definition: Any,
    /,
    *,
    domain: Optional[DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
) -> Field:
    raise NotImplementedError


# See comment for `_field`.
@functools.singledispatch
def _connectivity(
    definition: Any,
    /,
    codomain: Dimension,
    *,
    domain: Optional[DomainLike] = None,
    dtype: Optional[core_defs.DType] = None,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> Connectivity:
    raise NotImplementedError


@runtime_checkable
class NeighborTable(Connectivity, Protocol):
    def __gt_type__(self) -> ConnectivityType: ...

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        # Note that this property is currently already there from inheriting from `Field`,
        # however this seems wrong, therefore we explicitly introduce it here (or it should come
        # implicitly from the `NdArrayConnectivityField` protocol).
        ...


def is_neighbor_table(obj: Any) -> TypeGuard[NeighborTable]:
    if not isinstance(obj, Connectivity):
        return False
    domain_dims = obj.domain.dims
    return (
        len(domain_dims) == 2
        and domain_dims[0].kind is DimensionKind.HORIZONTAL
        and domain_dims[1].kind is DimensionKind.LOCAL
    )


OffsetProviderElem: TypeAlias = NeighborTable
# Note: `OffsetProvider` and `TableTypes` should not be accessed directly,
# use the `get_offset` and `get_offset_type` functions instead.
#: Neighbor tables keyed by the connectivity's `offset_tag`, which is how the IR names it.
OffsetProvider: TypeAlias = Mapping[Tag, OffsetProviderElem]
#: The types of an offset provider's tables, under the same keys: what transformations and code
#: generation see instead of the tables (ADR 0019).
TableTypes: TypeAlias = Mapping[Tag, NeighborTableType]
#: An offset provider as users write it: keyed by `NeighborConnectivity` declarations (or, at the
#: IR level, by tags). The entry points of a program normalize it to an `OffsetProvider` with
#: `as_tag_keyed_offset_provider`, so everything below them sees tags only.
#: NOTE: `Any` keys, since `Mapping` is invariant in its key type: a tag-keyed provider would not
#: be a `Mapping[type[NeighborConnectivity] | Tag, ...]`. Keys are checked at runtime instead.
OffsetProviderLike: TypeAlias = Mapping[Any, OffsetProviderElem]
TableTypesLike: TypeAlias = Mapping[Any, NeighborTableType]


def is_offset_provider(obj: Any) -> TypeGuard[OffsetProvider]:
    if not isinstance(obj, Mapping):
        return False
    return all(isinstance(el, OffsetProviderElem) for el in obj.values())


def is_table_types(obj: Any) -> TypeGuard[TableTypes]:
    if not isinstance(obj, Mapping):
        return False
    return all(isinstance(el, NeighborTableType) for el in obj.values())


def offset_provider_to_type(offset_provider: OffsetProvider | TableTypes) -> TableTypes:
    """The types of an offset provider's tables, each typed by the declaration its key names."""
    return {
        key: value if isinstance(value, NeighborTableType) else _neighbor_table_type(key, value)
        for key, value in offset_provider.items()
    }


def _unbound_table_type(table: NeighborTable) -> NeighborTableType:
    structure = table.__gt_type__()
    return NeighborTableType(
        connectivity=structure,
        dtype=structure.dtype,
        skip_value=structure.skip_value,
        max_neighbors=len(table.domain[1].unit_range),
    )


def _neighbor_table_type(key: Tag, table: NeighborTable) -> NeighborTableType:
    """
    The type of `table` bound under `key`: typed by the declaration `key` is the `offset_tag` of.

    The declaration is found through the table's local dimension, which knows its owner and the
    connectivities sharing it; a key none of them answers to leaves the table undeclared.
    """
    table_type = _unbound_table_type(table)
    local = table_type.domain[1]
    if not (isinstance(local, DimensionMeta) and issubclass(local, LocalDimensionIndex)):
        return table_type
    # NOTE: the most recent sharer first: a redefined declaration (a re-run notebook cell) is
    # appended again under the same tag.
    for connectivity in (local.owner, *reversed(local.sharers)):
        if connectivity is not None and connectivity.offset_tag == key:
            return check_neighbor_table(connectivity, table_type)
    return table_type


def get_offset(offset_provider: OffsetProvider, offset_tag: str) -> OffsetProviderElem:
    """
    Get the `OffsetProviderElem` or `NeighborTableType` for the given `offset` string.

    Note: All accesses of `OffsetProvider` or `TableTypes` should go through this function.
    """
    # TODO(havogt): Once we have a custom class for `OffsetProvider`, we can absorb this functionality into it.
    if offset_tag not in offset_provider:
        raise KeyError(
            f"Connectivity '{offset_tag}' not found in the offset provider, which has"
            f" {sorted(map(str, offset_provider))}. Offset providers are keyed by"
            " 'NeighborConnectivity' declarations, e.g. '{V2E: v2e_table}'."
        )
    return offset_provider[offset_tag]


get_offset_type: Callable[[TableTypes, str], NeighborTableType] = get_offset  # type: ignore[assignment] # overload not possible since OffsetProvider and TableTypes overlap


def connectivity_key_over(
    offset_provider: OffsetProvider | TableTypes, local_dim: Dimension | Tag
) -> str:
    """
    The key of a bound connectivity whose local dimension is `local_dim` (a dimension or its tag).

    Neighbor reductions and sparse fields know only their local dimension, and use its table
    for the neighbor count and the skip values. That is the table keyed by the local dimension's
    tag, i.e. its owner's, if bound. Otherwise it is one of the connectivities *sharing* the local
    dimension (see `NeighborConnectivity`), each keyed by its own tag; the smallest key is taken,
    so the choice does not depend on the order of the provider. Connectivities sharing a local
    dimension have the same neighbor structure (see `NeighborConnectivity`), so which one does
    not matter.

    Raises:
        KeyError: If no bound connectivity has `local_dim` as its local dimension.
    """
    local_tag = local_dim if isinstance(local_dim, str) else local_dim.tag
    if local_tag in offset_provider:
        return local_tag
    candidates = [
        key
        for key, connectivity in offset_provider.items()
        if (neighbor_dim := _neighbor_dim_of(connectivity)) is not None
        and neighbor_dim.tag == local_tag
    ]
    if not candidates:
        raise KeyError(
            f"No connectivity over the local dimension '{local_tag}' is bound in the offset"
            f" provider, which has {sorted(map(str, offset_provider))}."
        )
    return min(candidates)


def _neighbor_dim_of(connectivity: Any) -> Optional[Dimension]:
    if isinstance(connectivity, NeighborTableType):
        return connectivity.domain[1]
    if is_neighbor_table(connectivity):
        return connectivity.domain.dims[1]
    return None


def has_offset(offset_provider: OffsetProvider | TableTypes, offset_tag: str) -> bool:
    """Determine if offset provider has an element for the given offset tag."""
    try:
        get_offset(offset_provider, offset_tag)  # type: ignore[arg-type]  # implementation is shared with `get_offset_type`, no need to duplicate the function
    except KeyError:
        return False
    return True


def hash_offset_provider_items_by_id(
    offset_provider: OffsetProviderLike | TableTypesLike,
) -> int:
    """
    Compute hash of an offset provider on the tuples of key and value id.

    This function is unsafe since it uses the `id` of the values in the
    offset provider, which could generate different hashes for two
    offset providers that are semantically equal. It additionally relies
    on the ordering of the items in the mapping, which could also lead to
    different hashes for semantically equal offset providers.
    """
    return hash(tuple((k, id(v)) for k, v in offset_provider.items()))


DomainDimT = TypeVar("DomainDimT", bound="Dimension")


@dataclasses.dataclass(frozen=True, eq=False)
class CartesianConnectivity(Connectivity[Dims[DomainDimT], DimT]):
    domain_dim: DomainDimT
    codomain: DimT
    offset: int = 0

    def __init__(
        self, domain_dim: DomainDimT, offset: int = 0, *, codomain: Optional[DimT] = None
    ) -> None:
        object.__setattr__(self, "domain_dim", domain_dim)
        object.__setattr__(self, "codomain", codomain if codomain is not None else domain_dim)
        object.__setattr__(self, "offset", offset)

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    def ndarray(self) -> Never:
        raise NotImplementedError()

    def asnumpy(self) -> Never:
        raise NotImplementedError()

    def as_scalar(self) -> Never:
        raise NotImplementedError()

    @functools.cached_property
    def domain(self) -> Domain:
        return Domain(dims=(self.domain_dim,), ranges=(UnitRange.infinite(),))

    @property
    def __gt_origin__(self) -> Never:
        raise TypeError("'CartesianConnectivity' does not support this operation.")

    @property
    def dtype(self) -> core_defs.DType[core_defs.IntegralScalar]:
        return core_defs.Int32DType()

    # This is a workaround to make this class concrete, since `codomain` is an
    # abstract property of the `Connectivity` Protocol.
    if not TYPE_CHECKING:

        @functools.cached_property
        def codomain(self) -> DimT:
            raise RuntimeError("This property should be always set in the constructor.")

    @property
    def skip_value(self) -> None:
        return None

    @classmethod
    def for_translation(
        cls, dimension: DomainDimT, offset: int
    ) -> CartesianConnectivity[DomainDimT, DomainDimT]:
        return cast(CartesianConnectivity[DomainDimT, DomainDimT], cls(dimension, offset))

    @classmethod
    def for_relocation(cls, old: DimT, new: DomainDimT) -> CartesianConnectivity[DomainDimT, DimT]:
        return cls(new, codomain=old)

    def inverse_image(self, image_range: UnitRange | NamedRange) -> Sequence[NamedRange]:
        if not isinstance(image_range, UnitRange):
            if image_range.dim != self.codomain:
                raise ValueError(
                    f"Dimension '{image_range.dim}' does not match the codomain dimension '{self.codomain}'."
                )

            image_range = image_range.unit_range

        assert isinstance(image_range, UnitRange)
        return (named_range((self.domain_dim, image_range - self.offset)),)

    def premap(
        self,
        index_field: Connectivity | type[NeighborConnectivity],
        *args: Connectivity | type[NeighborConnectivity],
    ) -> Connectivity:
        raise NotImplementedError()

    __call__ = premap

    def restrict(self, index: AnyIndexSpec) -> Never:
        raise NotImplementedError()  # we could possibly implement with a FunctionField, but we don't have a use-case

    __getitem__ = restrict


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


def order_dimensions(dims: Iterable[Dimension]) -> list[Dimension]:
    """Find the canonical ordering of the dimensions in `dims`."""
    if sum(1 for dim in dims if dim.kind == DimensionKind.LOCAL) > 1:
        raise ValueError("There are more than one dimension with DimensionKind 'LOCAL'.")
    # NOTE: `__qualname__`, not `tag`. The tag is qualified, so ordering by it would make a
    # field's canonical dimension order depend on *which module* each dimension is declared in --
    # moving a declaration would silently reorder a field's dimensions. The unqualified name keeps
    # the ordering a property of the dimensions themselves; `tag` only breaks ties between
    # same-named dimensions from different modules, so the order stays total.
    return sorted(
        dims,
        key=lambda dim: (
            _DIM_KIND_ORDER[dim.kind],
            as_non_staggered(dim).__qualname__,
            as_non_staggered(dim).tag,
        ),
    )


def check_dims(dims: Sequence[Dimension]) -> None:
    # A dimension and its staggered counterpart (i.e. sharing the same non-staggered base dimension)
    # denote different grid locations and must not appear together in the same field/domain: mixing
    # them is ambiguous (it makes `order_dimensions` non-total and produces duplicate backend tags).
    seen: dict[Dimension, Dimension] = {}
    for dim in dims:
        base = as_non_staggered(dim)
        if base in seen:
            raise ValueError(
                f"Dimensions '{seen[base]}' and '{dim}' cannot be combined: a dimension and its "
                f"staggered counterpart must not appear together in the same field or domain."
            )
        seen[base] = dim
    if list(dims) != order_dimensions(dims):
        raise ValueError(
            f"Dimensions '{', '.join(map(str, dims))}' are not ordered correctly, expected '{', '.join(map(str, order_dimensions(dims)))}'."
        )


def promote_dims(*dims_list: Sequence[Dimension]) -> list[Dimension]:
    """
    Find an ordering of multiple lists of dimensions.

    The resulting list contains all unique dimensions from the input lists,
    sorted first by dims_kind_order, i.e., `Dimension.kind` (`HORIZONTAL` < `LOCAL` < `VERTICAL`) and then
    lexicographically by `Dimension.tag`.

    Examples:
        >>> from gt4py.next.common import Dimension
        >>> class I(DimensionIndex, kind=DimensionKind.HORIZONTAL): ...
        >>> class J(DimensionIndex, kind=DimensionKind.HORIZONTAL): ...
        >>> class K(DimensionIndex, kind=DimensionKind.VERTICAL): ...
        >>> class E2V(LocalDimensionIndex): ...
        >>> class E2C(LocalDimensionIndex): ...
        >>> promote_dims([J, K], [I, K]) == [I, J, K]
        True
        >>> promote_dims([K, J], [I, K])
        Traceback (most recent call last):
        ...
        ValueError: Dimensions 'K[vertical], J[horizontal]' are not ordered correctly, expected 'J[horizontal], K[vertical]'.
        >>> promote_dims([I, K], [J, E2V]) == [I, J, E2V, K]
        True
        >>> promote_dims([I, E2C], [E2V, K])
        Traceback (most recent call last):
        ...
        ValueError: There are more than one dimension with DimensionKind 'LOCAL'.
    """

    for dims in dims_list:
        check_dims(list(dims))
    unique_dims = {dim for dims in dims_list for dim in dims}

    promoted_dims = order_dimensions(unique_dims)
    check_dims(promoted_dims)
    return promoted_dims


class FieldBuiltinFuncRegistry:
    """
    Mixin for adding `fbuiltins` registry to a `Field`.

    Subclasses of a `Field` with `FieldBuiltinFuncRegistry` get their own registry,
    dispatching (via ChainMap) to its parent's registries.
    """

    _builtin_func_map: collections.ChainMap[fbuiltins.BuiltInFunction, Callable] = (
        collections.ChainMap()
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls._builtin_func_map = collections.ChainMap(
            {},  # New empty `dict` for new registrations on this class
            *[
                c.__dict__["_builtin_func_map"].maps[0]  # adding parent `dict`s in mro order
                for c in cls.__mro__
                if "_builtin_func_map" in c.__dict__
            ],
        )

    @classmethod
    def register_builtin_func(
        cls, /, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Optional[Callable[_P, _R]] = None
    ) -> Any:
        assert op not in cls._builtin_func_map
        if op_func is None:  # when used as a decorator
            return functools.partial(cls.register_builtin_func, op)
        return cls._builtin_func_map.setdefault(op, op_func)

    @classmethod
    def __gt_builtin_func__(cls, /, func: fbuiltins.BuiltInFunction[_R, _P]) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)


#: Numeric value used to represent missing values in connectivities.
#: Equivalent to the `_FillValue` attribute in the UGRID Conventions
#: (see: http://ugrid-conventions.github.io/ugrid-conventions/).
_DEFAULT_SKIP_VALUE: Final[int] = -1
#: Interned staggered dimensions, keyed by their *base dimension class*.
#:
#: NOTE: this is not the name-keyed dimension registry ADR 0028 rejects. It is memoization of
#: a type constructor -- keyed by identity, populated only by `StaggeredMeta.__getitem__`, and
#: never consulted to turn a user-authored name into a class. `typing`'s own subscription cache
#: plays the same role for generic aliases.
_STAGGERED_CACHE: dict[Dimension, Dimension] = {}


class StaggeredMeta(DimensionMeta):
    """
    Metaclass of `Staggered`, whose subscription builds and interns a *real* class.

    A PEP 695 generic cannot be used here: `Staggered[K]` would be a `typing._GenericAlias`,
    not a class, so it would fail `issubclass` and eve's `type[DimensionIndex]` validation,
    and its `tag` could not name the base dimension. See ADR 0028.
    """

    #: Set by `__getitem__` on each parametrization. Its presence is what distinguishes a
    #: staggered dimension from the bare `Staggered` base, which is also a `StaggeredMeta`.
    base: Dimension

    @property
    def tag(cls) -> Tag:
        # NOTE: overridden so that `__qualname__` can stay the short, readable form used in
        # diagnostics while the tag carries the base's *full* tag, which `resolve` needs to find
        # a base declared in another module. Display is `__qualname__` and identity is `tag`,
        # for staggered dimensions exactly as for any other.
        if "base" in cls.__dict__:
            return f"{cls.__module__}.Staggered[{cls.base.tag}]"
        return super().tag

    def __getitem__(cls, base: Dimension) -> Dimension:
        if "base" in cls.__dict__:
            raise TypeError(
                f"'{cls.__qualname__}' is already staggered; a dimension cannot be staggered twice."
            )
        if not isinstance(base, DimensionMeta):
            raise TypeError(f"'Staggered' expects a dimension, got '{base!r}'.")
        if base.kind is DimensionKind.LOCAL:
            raise TypeError(f"'{base.__qualname__}' is a local dimension and cannot be staggered.")
        if is_staggered(base):
            raise TypeError(
                f"'{base.__qualname__}' is already staggered; a dimension cannot be staggered twice."
            )
        if (staggered := _STAGGERED_CACHE.get(base)) is None:
            staggered = cast(
                Dimension,
                StaggeredMeta(
                    f"Staggered[{base.__qualname__}]",
                    # NOTE: deliberately not `(cls, base)`. A staggered dimension is a
                    # *different* dimension, so `issubclass(Staggered[K], K)` must be false,
                    # or a staggered field would be accepted wherever a base one is required.
                    (cls,),
                    {
                        "_staggered_base": base,
                        "base": base,
                        "kind": base.kind,
                        "__slots__": (),
                        "__module__": cls.__module__,
                        "__qualname__": f"{cls.__qualname__}[{base.__qualname__}]",
                    },
                ),
            )
            # NOTE: `setdefault`, not an assignment: compilation runs in threads, and two of them
            # building `Staggered[K]` at once must still see one class (identity is the dimension).
            staggered = _STAGGERED_CACHE.setdefault(base, staggered)
        return staggered


if TYPE_CHECKING:
    # Checkers see an ordinary generic dimension, so `Staggered[K]` works in an annotation and
    # inside `Field[Dims[Staggered[K]], ...]`. The runtime form below builds a real, interned
    # class so that `issubclass` and eve's `type[...]` validation work. Verified clean under
    # `mypy --strict` and pyright.
    class Staggered[D: DimensionIndex](DimensionIndex):
        base: ClassVar[Dimension]

else:

    class Staggered(DimensionIndex, metaclass=StaggeredMeta):
        """
        A dimension sitting at the half-integer positions of a base dimension (ADR 0026).

        `Staggered[K]` is a real, interned dimension class: subscripting the same base twice
        returns the identical object, so it round-trips through the IR by identity.
        """

        __slots__ = ()

        def __init_subclass__(cls, /, **kwargs: Any) -> None:
            # NOTE: gate on the namespace marker the metaclass sets, not on a module-level
            # "currently building" flag -- compilation runs in worker processes and threads.
            if "_staggered_base" not in cls.__dict__:
                raise TypeError(
                    f"'{cls.__qualname__}' cannot subclass a staggered dimension directly;"
                    " write 'Staggered[BaseDim]'."
                )
            super().__init_subclass__(**kwargs)


def _reduce_staggered(cls: StaggeredMeta) -> Any:
    """
    Pickle a staggered dimension through its base, falling back to by-reference.

    `Staggered[K].__qualname__` contains brackets, which `pickle.save_global` cannot look up,
    and `copyreg` is the only hook consulted before `save_global` for a class. Reconstruction
    goes through `StaggeredMeta.__getitem__`, so identity is preserved.

    The bare `Staggered` base is also a `StaggeredMeta` instance but has no `base`, so it must
    fall through to ordinary by-reference pickling.
    """
    if "base" not in cls.__dict__:
        return cls.__qualname__
    return (_make_staggered, (cls.base,))


def _make_staggered(base: Dimension) -> Dimension:
    return Staggered[base]  # type: ignore[valid-type] # runtime subscription, see StaggeredMeta


copyreg.pickle(StaggeredMeta, _reduce_staggered)


def is_staggered(dim: Dimension) -> bool:
    """
    Return whether `dim` is a staggered dimension.

    Checks for the marker the metaclass sets, not `issubclass(dim, Staggered)`: the latter is
    also true of the bare `Staggered` base, which has no base dimension to recover.
    """
    return "base" in dim.__dict__


def flip_staggered(dim: Dimension) -> Dimension:
    """Return the staggered counterpart of `dim`."""
    if is_staggered(dim):
        return cast(Dimension, dim.base)  # type: ignore[attr-defined] # guarded by is_staggered
    return Staggered[dim]  # type: ignore[valid-type] # runtime subscription


def as_non_staggered(dim: Dimension) -> Dimension:
    """Return the non-staggered base dimension of `dim` (`dim` itself if already non-staggered)."""
    if is_staggered(dim):
        return flip_staggered(dim)
    return dim


def connectivity_for_cartesian_shift(dim: Dimension, offset: int | float) -> CartesianConnectivity:
    """
    Build the connectivity that shifts `dim` by `offset`.

    An integer `offset` shifts within `dim` (the codomain stays `dim`). A half-integer `offset`
    (fractional part `0.5`) shifts to the staggered counterpart of `dim` (the codomain becomes
    `flip_staggered(dim)`).

    The half-integer case encodes the convention that a staggered index sits half a cell *below*
    its base index (see ADR 0026): `IHalf(0)` is the edge below `I(0)`. Because of this asymmetry,
    shifting out of a non-staggered dimension needs a `+1` index correction that shifting out of a
    staggered dimension does not, e.g. `I + 0.5` maps `I(i)` to `IHalf(i+1)` (position `i+½`) while
    `IHalf + 0.5` maps `IHalf(i)` to `I(i)`.
    """
    integral_offset, staggered_offset = divmod(offset, 1)
    if staggered_offset == 0.5:
        if not is_staggered(dim):
            integral_offset += 1
        return CartesianConnectivity(dim, int(integral_offset), codomain=flip_staggered(dim))
    else:
        assert staggered_offset == 0
        return CartesianConnectivity(dim, int(integral_offset), codomain=dim)


class LocalDimensionIndex(DimensionIndex):
    """
    A local dimension: the axis that runs over the neighbors of one element.

    A local dimension is declared either inside a `NeighborConnectivity`, as its nested `Local`
    class, or on its own for a local axis that indexes no table (`owner is None`), such as the
    coefficients of a fixed-size stencil:

        >>> class LsqCoeff(LocalDimensionIndex, size=3): ...
        >>> LsqCoeff.kind, LsqCoeff.owner, LsqCoeff.max_neighbors
        (<DimensionKind.LOCAL: 'local'>, None, 3)

    Neighbor counts are optional. A declared count is a constraint the bound table has to
    satisfy (see `check_neighbor_table`); an undeclared one is taken from the table.
    """

    __slots__ = ()

    kind: ClassVar[DimensionKind] = DimensionKind.LOCAL
    _local_dimension_root: ClassVar[bool] = True

    #: The connectivity this dimension is the local axis of, or `None` if it indexes no table.
    #: Set by `NeighborConnectivity` when the connectivity is declared.
    owner: ClassVar[Optional[type[NeighborConnectivity]]] = None
    #: The connectivities sharing this dimension with its owner, in declaration order. Kept so that
    #: a table bound under a sharer's `offset_tag` can be typed by its declaration: the table alone
    #: looks like the owner's (see `NeighborTableType`).
    sharers: ClassVar[tuple[type[NeighborConnectivity], ...]] = ()
    #: Number of entries per element, i.e. the table's second extent, if declared.
    max_neighbors: ClassVar[Optional[int]] = None
    #: Least number of *valid* neighbors of any element, if declared. Fewer than
    #: `max_neighbors` means the table pads with skip values.
    min_neighbors: ClassVar[Optional[int]] = None
    #: The `size=` of this declaration, kept apart from the counts an owner writes below.
    declared_size: ClassVar[Optional[int]] = None

    def __init_subclass__(
        cls,
        /,
        *,
        size: Optional[int] = None,
        kind: Optional[DimensionKind] = None,
        **kwargs: Any,
    ) -> None:
        if kind is not None and kind is not DimensionKind.LOCAL:
            raise TypeError(
                f"'{cls.__qualname__}' is a local dimension and cannot have kind '{kind}'."
            )
        super().__init_subclass__(**kwargs)
        # NOTE: reset rather than inherited: a subclass of an owned local dimension is a
        # different dimension, and does not index its parent's table.
        cls.owner = None
        cls.sharers = ()
        cls.declared_size = _check_neighbor_count(cls, "size", size)
        cls.max_neighbors = cls.min_neighbors = cls.declared_size


def _check_neighbor_count(cls: type, name: str, count: Optional[int]) -> Optional[int]:
    if count is None:
        return None
    if not isinstance(count, numbers.Integral) or isinstance(count, bool):
        raise TypeError(f"'{cls.__qualname__}': '{name}' must be an integer, got '{count!r}'.")
    if count < 0:
        raise ValueError(f"'{cls.__qualname__}': '{name}' must be non-negative, got {count}.")
    return int(count)


class ConstList(LocalDimensionIndex, size=1):
    """
    The local dimension of a list of one repeated value (`make_const_list`).

    An owner-less local dimension of size 1: the value is broadcast against the neighbor lists it
    is combined with, and a materialized constant list has extent 1 along it. It indexes no table,
    so it is never in an offset provider.

    Declared here, once: it used to be built independently in `iterator/embedded.py` and in the
    DaCe lowering, which only worked while dimensions compared by `(name, kind)`.
    """

    __slots__ = ()


class ConnectivityMeta(type):
    """
    Metaclass of `NeighborConnectivity` declarations.

    A connectivity declaration is a class that is never instantiated. It is written in DSL code
    (`a(V2E)`, `a(V2E[0])`), and it is what the neighbor table bound at call time must match.
    """

    # NOTE: `Local` is deliberately *not* annotated here, nor on `NeighborConnectivity`: an
    # annotated `Local` makes every declaration's nested class a *variable* for the checkers, so
    # `Field[Dims[V, V2E.Local]]` is rejected (pyright) or "not valid as a type" (mypy, for the
    # assigned form). Library code reads it through `local_dimension_of`.
    domain: Dimension
    codomain: Dimension

    @property
    def tag(cls) -> Tag:
        """The connectivity's identity: its qualified Python name."""
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def offset_tag(cls) -> Tag:
        """
        The name of the connectivity in the IR, and its key in a normalized offset provider.

        The tag of its local dimension, if it declares it: shifts, neighbor reductions and sparse
        arguments then all find the table under one string. A connectivity that *shares* another
        one's local dimension (a flattened sparse pattern, e.g. cell-to-cell-edge indexing the
        same neighbor axis as cell-to-edge) is named by its own tag, since the local dimension's
        tag already names its owner's table.
        """
        local = cls._local()
        return local.tag if local.owner is cls else cls.tag

    def _local(cls) -> type[LocalDimensionIndex]:
        if (local := cls.__dict__.get("Local")) is None:
            raise TypeError(
                f"'{cls.__qualname__}' is not a connectivity declaration; declare one by"
                " subclassing 'NeighborConnectivity[Domain, Codomain]'."
            )
        return cast(type[LocalDimensionIndex], local)

    def __call__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(
            f"'{cls.__qualname__}' is a connectivity declaration and cannot be instantiated;"
            " bind a neighbor table to it through the offset provider."
        )

    @overload
    def __getitem__(cls, item: int) -> Connectivity: ...
    @overload
    def __getitem__(cls, item: Any) -> Any: ...
    def __getitem__(cls, item: Any) -> Any:
        # NOTE: `numbers.Integral`, not `int`, so `V2E[np.int32(1)]` does not fall through to
        # type-parameter subscription; `bool` is excluded so `V2E[True]` is an error.
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            return cls.bound_table()[cls._local()(int(item))]
        if "Local" in cls.__dict__:
            raise TypeError(
                f"'{cls.__qualname__}[{item!r}]': a connectivity is indexed by an integer"
                " neighbor position."
            )
        # A metaclass `__getitem__` shadows `__class_getitem__`, so type-parameter
        # subscription (`NeighborConnectivity[V, E]`) has to be forwarded explicitly.
        return cast(Any, cls).__class_getitem__(item)

    def __repr__(cls) -> str:
        return cls.tag

    def __str__(cls) -> str:
        return cls.__qualname__

    def __gt_type__(cls) -> Any:
        """
        The type of the connectivity in DSL code: a shift from `Codomain` to `(Domain, Local)`.

        Its tag is `offset_tag`, which is how the IR names the connectivity and how the offset
        provider is keyed once normalized (see `as_tag_keyed_offset_provider`).
        """
        from gt4py.next.type_system import type_specifications as ts

        local = cls._local()
        return ts.ShiftType(codomain=cls.codomain, domain=(cls.domain, local), tag=cls.offset_tag)

    def bound_table(cls) -> NeighborTable:
        """The neighbor table bound to this connectivity in the current embedded execution."""
        from gt4py.next import embedded

        offset_provider = embedded.context.get_offset_provider(None)
        if offset_provider is None:
            raise RuntimeError(
                f"'{cls.__qualname__}' can only be resolved to a table during embedded execution."
            )
        table = get_offset(offset_provider, cls.offset_tag)
        if not is_neighbor_table(table):
            raise TypeError(
                f"'{cls.__qualname__}' is bound to '{table}', which is not a neighbor table."
            )
        return table


class NeighborConnectivity[Domain: DimensionIndex, Codomain: DimensionIndex](
    metaclass=ConnectivityMeta
):
    """
    Declare a neighbor connectivity: for each `Domain` element, a list of `Codomain` neighbors.

    The declaration names the connectivity's local dimension -- its nested `Local` class --
    and optionally its neighbor counts. It holds no data: the neighbor table is bound at call
    time through the offset provider. `check_neighbor_table` checks a table against the
    declaration.

    Examples:
        >>> class Vertex(DimensionIndex): ...
        >>> class Edge(DimensionIndex): ...
        >>> class V2E(NeighborConnectivity[Vertex, Edge], max_neighbors=6, min_neighbors=5):
        ...     class Local(LocalDimensionIndex): ...
        >>> V2E.domain is Vertex, V2E.codomain is Edge
        (True, True)
        >>> V2E.Local.owner is V2E, V2E.Local.max_neighbors, V2E.Local.min_neighbors
        (True, 6, 5)
    """

    # NOTE: `Local` is not annotated (see `ConnectivityMeta`); every subclass declares it, as a
    # nested class or as `Local: TypeAlias = <a local dimension>`.
    domain: ClassVar[Dimension]
    codomain: ClassVar[Dimension]

    def __init_subclass__(
        cls,
        /,
        *,
        max_neighbors: Optional[int] = None,
        min_neighbors: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__qualname__
        if "<locals>" in name:
            raise TypeError(
                f"'{name}' must be declared at module level: a connectivity is referenced from"
                " the IR by its qualified name, which has to be importable."
            )
        params = [
            xtyping.get_args(base)
            for base in cls.__dict__.get("__orig_bases__", ())
            if xtyping.get_origin(base) is NeighborConnectivity
        ]
        if len(params) != 1 or len(params[0]) != 2:
            raise TypeError(
                f"'{name}' must derive from 'NeighborConnectivity[Domain, Codomain]' directly,"
                " with both dimensions given."
            )
        domain, codomain = params[0]
        for role, dim in (("Domain", domain), ("Codomain", codomain)):
            if not isinstance(dim, DimensionMeta) or dim.kind is DimensionKind.LOCAL:
                raise TypeError(f"'{name}': '{role}' must be a non-local dimension, got '{dim}'.")

        local = cls.__dict__.get("Local")
        if not (isinstance(local, DimensionMeta) and issubclass(local, LocalDimensionIndex)):
            raise TypeError(
                f"'{name}' must declare its local dimension, either as a nested class"
                " ('class Local(LocalDimensionIndex): ...') or by adopting one"
                " ('Local: TypeAlias = SomeLocalDim')."
            )
        if local is ConstList:
            raise TypeError(
                f"'{name}' cannot adopt '{ConstList.__qualname__}': it is the local dimension"
                " of 'make_const_list' results and belongs to no connectivity."
            )
        max_neighbors = _check_neighbor_count(cls, "max_neighbors", max_neighbors)
        min_neighbors = _check_neighbor_count(cls, "min_neighbors", min_neighbors)
        # NOTE: a declaration whose tag is the owner's is a *redefinition* of it (a re-run
        # notebook cell), not a second connectivity sharing the local dimension, so it takes
        # ownership over again. Ownership of a local dimension that is adopted, rather than
        # nested, otherwise goes to whoever declares first.
        if local.owner is not None and local.owner.tag != cls.tag:
            # Sharing another connectivity's local dimension: the neighbor structure is the
            # owner's, including its counts, and the sharing connectivity is named by its own tag.
            owner_name = local.owner.__qualname__
            if domain is not local.owner.domain:
                raise TypeError(
                    f"'{name}' cannot share the local dimension of '{owner_name}': it has domain"
                    f" '{domain}', but the neighbors of '{owner_name}' are those of"
                    f" '{local.owner.domain}'."
                )
            for count_name, count in (
                ("max_neighbors", max_neighbors),
                ("min_neighbors", min_neighbors),
            ):
                if count is not None and count != getattr(local, count_name):
                    raise TypeError(
                        f"'{name}': '{count_name}={count}' contradicts the local dimension it"
                        f" shares with '{owner_name}', which declares"
                        f" {count_name}={getattr(local, count_name)}."
                    )
            cls.domain, cls.codomain = domain, codomain
            local.sharers = (*local.sharers, cls)
            return
        for count_name, count in (
            ("max_neighbors", max_neighbors),
            ("min_neighbors", min_neighbors),
        ):
            # NOTE: against the local dimension's own `size=`, not against counts a previous
            # owner wrote: a redefinition must be checked against what its `Local` declares.
            if (
                count is not None
                and local.declared_size is not None
                and count != local.declared_size
            ):
                raise TypeError(
                    f"'{name}': '{count_name}={count}' contradicts the size declared by"
                    f" '{local.__qualname__}' ({local.declared_size})."
                )
        max_neighbors = max_neighbors if max_neighbors is not None else local.declared_size
        min_neighbors = min_neighbors if min_neighbors is not None else local.declared_size
        if (
            max_neighbors is not None
            and min_neighbors is not None
            and min_neighbors > max_neighbors
        ):
            raise TypeError(
                f"'{name}': 'min_neighbors' ({min_neighbors}) exceeds 'max_neighbors'"
                f" ({max_neighbors})."
            )

        cls.domain, cls.codomain = domain, codomain
        local.owner = cls
        local.max_neighbors, local.min_neighbors = max_neighbors, min_neighbors


def local_dimension_of(connectivity: type[NeighborConnectivity]) -> type[LocalDimensionIndex]:
    """
    The local dimension a connectivity declares, adopts or shares.

    Library code reads `V2E.Local` through this accessor: the attribute is intentionally not
    annotated, so that a declaration's `Local` stays a *type* for the type checkers (see
    `ConnectivityMeta`).

    Raises:
        TypeError: If `connectivity` declares no local dimension.
    """
    return cast(ConnectivityMeta, connectivity)._local()


def check_neighbor_table(
    connectivity: type[NeighborConnectivity],
    table: NeighborTable | NeighborTableType,
) -> NeighborTableType:
    """
    Check that a neighbor table matches the connectivity declaration it is bound to.

    Skip values are checked on the table's *type*: a table with a `skip_value` counts as
    having skip values whether or not any entry uses it.

    Args:
        connectivity: The declaration.
        table: The bound table, or its type (which is all an ahead-of-time compilation has).

    Returns:
        The type of the table bound to `connectivity`.

    Raises:
        ValueError: On the first mismatch, naming the connectivity and the mismatch.
    """
    name = connectivity.__qualname__
    local = local_dimension_of(connectivity)

    def fail(reason: str) -> NoReturn:
        raise ValueError(f"The table bound to '{name}' does not match its declaration: {reason}.")

    if isinstance(table, NeighborTableType):
        table_type = table
    elif is_neighbor_table(table):
        table_type = _unbound_table_type(table)
    else:
        fail(f"expected a neighbor table, got '{table}'")

    def redefined(found: Sequence[Any], expected: Sequence[Any], what: str = "dimension") -> str:
        if any(f is not e and f.tag == e.tag for f, e in zip(found, expected)):
            return (
                f" (a {what} of the same name but a different class: was the declaration"
                " redefined, e.g. by re-running a notebook cell?)"
            )
        return ""

    if isinstance(bound_to := table_type.connectivity, ConnectivityMeta) and (
        bound_to is not connectivity
    ):
        fail(
            f"its type is bound to '{bound_to.__qualname__}'"
            + redefined((bound_to,), (connectivity,), what="connectivity")
        )

    expected_domain = (connectivity.domain, local)
    if tuple(table_type.domain) != expected_domain:
        fail(
            f"its domain is '({', '.join(map(str, table_type.domain))})',"
            f" expected '({', '.join(map(str, expected_domain))})'"
            + redefined(table_type.domain, expected_domain)
        )
    if table_type.codomain is not connectivity.codomain:
        fail(
            f"its codomain is '{table_type.codomain}', expected '{connectivity.codomain}'"
            + redefined((table_type.codomain,), (connectivity.codomain,))
        )
    if not np.issubdtype(table_type.dtype.scalar_type, np.integer):
        fail(f"its dtype '{table_type.dtype}' is not integral")
    if local.max_neighbors is not None and table_type.max_neighbors != local.max_neighbors:
        fail(
            f"it has {table_type.max_neighbors} neighbors per element,"
            f" expected max_neighbors={local.max_neighbors}"
        )
    if local.min_neighbors is not None:
        max_neighbors = table_type.max_neighbors
        if local.min_neighbors > max_neighbors:
            fail(
                f"min_neighbors={local.min_neighbors} exceeds its {max_neighbors} neighbors"
                " per element"
            )
        if local.min_neighbors < max_neighbors and not table_type.has_skip_values:
            fail(
                f"min_neighbors={local.min_neighbors} < {max_neighbors} requires a skip value,"
                " but the table has none"
            )
        if local.min_neighbors == max_neighbors and table_type.has_skip_values:
            fail(
                f"min_neighbors == max_neighbors == {max_neighbors} means every element has all"
                f" its neighbors, but the table has skip value {table_type.skip_value}"
            )
    return dataclasses.replace(table_type, connectivity=connectivity)


@overload
def as_tag_keyed_offset_provider(
    offset_provider: OffsetProviderLike, *, strict: bool = True
) -> OffsetProvider: ...
@overload
def as_tag_keyed_offset_provider(
    offset_provider: TableTypesLike, *, strict: bool = True
) -> TableTypes: ...
def as_tag_keyed_offset_provider(
    offset_provider: OffsetProviderLike | TableTypesLike, *, strict: bool = True
) -> OffsetProvider | TableTypes:
    """
    Key an offset provider by tags, the form the IR and the backends use.

    A `NeighborConnectivity` key becomes its `offset_tag`: its local dimension's tag, or its own
    for a connectivity sharing another one's local dimension. A string key is taken to be
    such a tag already, and is rejected if it cannot be one: a tag is a qualified name, so a bare
    name such as `"V2E"` is the removed `FieldOffset` spelling.

    Called on every program call, so it does not check tables against their declarations; see
    `check_offset_provider`.

    Args:
        offset_provider: The provider to normalize.
        strict: Whether to reject string keys that cannot be tags. Internal hooks that are handed
            hand-written providers, such as `embedded.context.update`, pass `False`.
    """
    if not any(isinstance(key, ConnectivityMeta) for key in offset_provider):
        if strict:
            _check_tag_keys(offset_provider)
        return offset_provider
    result: dict[Tag, Any] = {}
    for key, value in offset_provider.items():
        tag = key.offset_tag if isinstance(key, ConnectivityMeta) else key
        if tag in result:
            raise ValueError(f"The offset provider binds '{tag}' twice.")
        result[tag] = value
    if strict:
        _check_tag_keys(result)
    return result


def _check_tag_keys(offset_provider: Mapping[Any, Any]) -> None:
    for key in offset_provider:
        if not isinstance(key, str) or "." not in key:
            raise TypeError(
                f"Invalid offset-provider key {key!r}: offset providers are keyed by"
                " 'NeighborConnectivity' declarations, e.g. '{V2E: v2e_table}'. A bare name is the"
                " spelling of the removed 'FieldOffset' (see ADR 0029)."
            )


#: Offset providers already checked, by the hash of their `(key, id(table))` items. Bounded, and
#: not authoritative: like the compiled-program cache (which keys on the same hash), it can in
#: principle skip a check when a freed table is replaced at the same address. See
#: `check_offset_provider`.
_CHECKED_OFFSET_PROVIDERS: Final[collections.OrderedDict[int, None]] = collections.OrderedDict()
_CHECKED_OFFSET_PROVIDERS_MAX: Final = 256


def check_offset_provider(
    offset_provider: OffsetProviderLike | TableTypesLike, *, deep: bool = False
) -> None:
    """
    Check every table of an offset provider against its connectivity declaration.

    A key that does not name a declared connectivity -- e.g. a tag used only by hand-written IR --
    has no declaration to be checked against and is skipped. Providers are remembered by the
    identity of their tables, so repeated calls with the same tables cost one hash.

    Args:
        offset_provider: The provider, keyed by declarations or by tags.
        deep: Also compare the skip-value positions of tables over one shared local dimension,
            which reads the tables. The compile path does; the call path does not.

    Raises:
        ValueError: If a table does not match its declaration, see `check_neighbor_table`.
    """
    if (seen := hash_offset_provider_items_by_id(offset_provider)) in _CHECKED_OFFSET_PROVIDERS:
        return
    for key, table in offset_provider.items():
        declaration: Any = key
        if isinstance(key, str):
            if (declaration := _import_qualified_name_or_none(key)) is None:
                continue
        if isinstance(declaration, DimensionMeta):
            # the local dimension's tag names its owner's table
            declaration = getattr(declaration, "owner", None)
        if isinstance(declaration, ConnectivityMeta):
            if declaration.offset_tag not in (key, getattr(key, "offset_tag", None)):
                # e.g. `{V2E.tag: table}`: the connectivity's own tag, which is the IR name only
                # of a connectivity *sharing* a local dimension
                raise ValueError(
                    f"Invalid offset-provider key '{key}': it names the connectivity"
                    f" '{declaration.__qualname__}', whose key is the declaration itself"
                    f" ('{{{declaration.__qualname__}: table}}')."
                )
            check_neighbor_table(cast(type[NeighborConnectivity], declaration), table)
    _check_shared_local_dimensions(offset_provider, deep=deep)
    _CHECKED_OFFSET_PROVIDERS[seen] = None
    while len(_CHECKED_OFFSET_PROVIDERS) > _CHECKED_OFFSET_PROVIDERS_MAX:
        _CHECKED_OFFSET_PROVIDERS.popitem(last=False)


def _check_shared_local_dimensions(
    offset_provider: OffsetProviderLike | TableTypesLike, *, deep: bool = False
) -> None:
    """
    Check that the tables over one local dimension have the same neighbor structure.

    Reductions and sparse fields take the neighbor count and the skip values of a local dimension
    from any one table over it (see `connectivity_key_over`), which is only sound if all of them
    agree: the same number of neighbors, and a skip value at the same positions.
    """
    by_local_dim: dict[Tag, list[tuple[Any, Any]]] = collections.defaultdict(list)
    for key, table in offset_provider.items():
        if (neighbor_dim := _neighbor_dim_of(table)) is not None:
            by_local_dim[neighbor_dim.tag].append((key, table))
    for local_tag, tables in by_local_dim.items():
        (first_key, first), *others = tables
        first_type = first if isinstance(first, NeighborTableType) else _unbound_table_type(first)
        for key, table in others:
            table_type = (
                table if isinstance(table, NeighborTableType) else _unbound_table_type(table)
            )
            same_structure = (table_type.max_neighbors, table_type.has_skip_values) == (
                first_type.max_neighbors,
                first_type.has_skip_values,
            )
            if (
                deep
                and same_structure
                and first_type.has_skip_values
                and is_neighbor_table(first)
                and is_neighbor_table(table)
            ):
                # NOTE: compared where the tables live, without copying device arrays to the host.
                xp = first.array_ns  # type: ignore[attr-defined] # all tables are NdArrayFields
                same_structure = first.ndarray.shape == table.ndarray.shape and bool(
                    xp.all(
                        (first.ndarray == first_type.skip_value)
                        == (xp.asarray(table.ndarray) == table_type.skip_value)
                    )
                )
            if not same_structure:
                raise ValueError(
                    f"'{key}' and '{first_key}' are bound to tables over the same local dimension"
                    f" '{local_tag}' with a different neighbor structure: connectivities sharing a"
                    " local dimension must have the same number of neighbors, and skip values at"
                    " the same positions."
                )
