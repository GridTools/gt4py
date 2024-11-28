# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import dataclasses
import enum
import itertools
import numbers
from typing import Any, Callable, Optional, Sequence, Tuple

import hypothesis.strategies as hyp_st
import numpy as np
from hypothesis.extra import numpy as hyp_np


# ---- Test Suites utilities ----
class SymbolKind(enum.Enum):
    NONE = 0
    GLOBAL_STRATEGY = 1
    GLOBAL_SET = 2
    SINGLETON = 3
    PARAMETER = 4
    FIELD = 5


@dataclasses.dataclass(frozen=True)
class _SymbolStrategy:
    kind: SymbolKind
    boundary: Optional[Sequence[Tuple[int, int]]]
    axes: Optional[str]
    data_dims: Optional[Tuple[int, ...]]
    value_st_factory: Callable[..., hyp_st.SearchStrategy]


@dataclasses.dataclass(frozen=True)
class _SymbolValueTuple:
    kind: str
    boundary: Sequence[Tuple[int, int]]
    values: Tuple[Any]


def global_name(*, singleton=None, symbol=None, one_of=None, in_range=None):
    """Define a *global* symbol."""
    if singleton is not None:
        assert symbol is None and one_of is None and in_range is None
        return _SymbolValueTuple(kind=SymbolKind.SINGLETON, boundary=None, values=(singleton,))

    if symbol is not None:
        assert singleton is None and one_of is None and in_range is None
        return _SymbolValueTuple(kind=SymbolKind.GLOBAL_SET, boundary=None, values=(symbol,))

    if one_of is not None:
        assert (
            singleton is None
            and symbol is None
            and in_range is None
            and isinstance(one_of, collections.abc.Sequence)
        )
        return _SymbolValueTuple(kind=SymbolKind.GLOBAL_SET, boundary=None, values=one_of)

    if in_range is not None:
        assert singleton is None and symbol is None and one_of is None and len(in_range) == 2
        return _SymbolStrategy(
            kind=SymbolKind.GLOBAL_STRATEGY,
            boundary=None,
            axes=None,
            data_dims=None,
            value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
        )

    else:
        raise AssertionError("Missing value descriptor")


def field(*, in_range, boundary=None, axes=None, data_dims=None, extent=None):
    """Define a *field* symbol."""
    assert (boundary is not None or extent is not None) and len(in_range) == 2
    boundary = boundary or [(abs(e[0]), abs(e[1])) for e in extent]
    extent = extent or [(-b[0], b[1]) for b in boundary]
    assert all((-b[0], b[1]) == (e[0], e[1]) for b, e in zip(boundary, extent))
    assert all((b[0] >= 0 and b[1]) >= 0 for b in boundary)

    return _SymbolStrategy(
        kind=SymbolKind.FIELD,
        boundary=boundary,
        axes=axes,
        data_dims=data_dims or tuple(),
        value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
    )


def parameter(*, one_of=None, in_range=None):
    """Define a *parameter* symbol."""
    if one_of is not None:
        assert in_range is None and isinstance(one_of, collections.abc.Sequence)
        return _SymbolStrategy(
            kind=SymbolKind.PARAMETER,
            boundary=None,
            axes=None,
            data_dims=None,
            value_st_factory=lambda dt: one_of_values_st(one_of).map(dt),
        )

    elif in_range is not None:
        assert one_of is None and len(in_range) == 2
        return _SymbolStrategy(
            kind=SymbolKind.PARAMETER,
            boundary=None,
            axes=None,
            data_dims=None,
            value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
        )

    else:
        raise AssertionError("Missing value descriptor")


def none():
    """Define the symbol ``None``."""
    return _SymbolStrategy(
        kind=SymbolKind.NONE,
        boundary=None,
        axes=None,
        data_dims=None,
        value_st_factory=lambda dt: hyp_st.none(),
    )


# ---- Custom Hypothesis strategies ----
def scalar_value_st(dtype, min_value, max_value, allow_nan=False):
    """Hypothesis strategy for `dtype` scalar values in range [min_value, max_value]."""
    allow_infinity = not (np.isfinite(min_value) and np.isfinite(max_value))

    if issubclass(dtype.type, numbers.Integral):
        value_st = hyp_st.integers(min_value, max_value)
    elif issubclass(
        dtype.type, numbers.Real
    ):  # after numbers.Integral because np.int32 is a subclass of numbers.Real
        value_st = hyp_st.floats(
            min_value,
            max_value,
            allow_infinity=allow_infinity,
            allow_nan=allow_nan,
            width=dtype.itemsize * 8,
        )

    return value_st.map(dtype.type)


def one_of_values_st(args):
    """Hypothesis strategy returning one of the values passed in the arguments."""
    if len(args) == 0:
        return hyp_st.just(None)
    else:
        return hyp_st.sampled_from(args)


def ndarray_shape_st(sizes):
    """Hypothesis strategy for shapes of `ndims` dimensions and size within `size_range`."""
    return hyp_st.tuples(*[hyp_st.integers(min_size, max_size) for (min_size, max_size) in sizes])


def derived_shape_st(shape_st, extra: Sequence[Optional[int]]):
    """Hypothesis strategy for extending shapes generated from a provided strategy with some extra padding.

    If an element of extra contains None, the item will be dropped from the final shape, otherwise,
    both shape and extra elements are summed together.
    """
    return hyp_st.builds(
        lambda shape: tuple(
            [d + e for d, e in itertools.zip_longest(shape, extra, fillvalue=0) if e is not None]
        ),
        shape_st,
    )


def ndarray_in_range_st(dtype, shape_st, value_range):
    """Hypothesis strategy for ndarrays generated using the provided `dtype` and `shape` strategies and the `value_range`."""
    return hyp_np.arrays(
        dtype=dtype,
        shape=shape_st,
        elements=scalar_value_st(dtype, min_value=value_range[0], max_value=value_range[1]),
        fill=scalar_value_st(dtype, min_value=value_range[0], max_value=value_range[1]),
    )


def ndarray_st(dtype, shape_strategy, value_st_factory):
    """Hypothesis strategy for ndarrays generated using the provided `dtype` and `shape` and `value_st_factory` strategies/factories."""
    return hyp_np.arrays(
        dtype=dtype,
        shape=shape_strategy,
        elements=value_st_factory(dtype),
        fill=value_st_factory(dtype),
    )


# ---- Utility functions ----
@hyp_st.composite
def draw_from_strategies(draw, strategies):
    """Generate a Hypothesis strategy example by composing named strategies from a dictionary.

    Parameters
    ----------
    draw : `function`
        Hypothesis :func:`draw` implementation.
    strategies : `dict`
        Named strategies.
        - ``name``: Hypothesis strategy (`function`).

    Returns
    ----------
    args: `dict`
        A dictionary of Hypothesis examples.
        - ``name``: Hypothesis strategy example (``value``).
    """
    args = dict()
    for name, strategy in strategies.items():
        args[name] = draw(strategy)

    return args


def composite_strategy_factory(dtypes, strategy_factories):
    strategy_dict = dict()
    for name, factory in strategy_factories.items():
        strategy_dict[name] = factory(dtypes[name])

    @hyp_st.composite
    def strategy(draw):
        args = dict()
        for name, strategy in strategy_dict.items():
            args[name] = draw(strategy)
        return args

    return strategy


def composite_implementation_strategy_factory(
    dtypes, validation_strategy_factories, global_boundaries
):
    """Generate strategy for run-time :class:`StencilTestSuite` parameters."""
    strategy_dict = dict()
    for name, factory in validation_strategy_factories.items():
        strategy_dict[name] = factory(dtypes[name])

    @hyp_st.composite
    def strategy(draw):
        inputs = dict()
        for name, strat in strategy_dict.items():
            inputs[name] = draw(strat)

        return inputs, dict()

    return strategy
