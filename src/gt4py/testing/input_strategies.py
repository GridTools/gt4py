# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import numbers
import types

import hypothesis.strategies as hyp_st
import numpy as np
from hypothesis.extra import numpy as hyp_np


# ---- Test Suites utilities ----
class _SymbolStrategy(types.SimpleNamespace):
    def __init__(self, kind, boundary, value_st_factory):
        super().__init__(kind=kind, boundary=boundary, value_st_factory=value_st_factory)


class _SymbolValueTuple(types.SimpleNamespace):
    def __init__(self, kind, boundary, values):
        super().__init__(kind=kind, boundary=boundary, values=values)


def global_name(*, singleton=None, symbol=None, one_of=None, in_range=None):
    """Define a *global* symbol."""
    if singleton is not None:
        assert symbol is None and one_of is None and in_range is None
        return _SymbolValueTuple(kind="singleton", boundary=None, values=(singleton,))

    elif symbol is not None:
        assert singleton is None and one_of is None and in_range is None
        return _SymbolValueTuple(kind="global_set", boundary=None, values=(symbol,))

    elif one_of is not None:
        assert (
            singleton is None
            and symbol is None
            and in_range is None
            and isinstance(one_of, collections.abc.Sequence)
        )
        return _SymbolValueTuple(kind="global_set", boundary=None, values=one_of)

    elif in_range is not None:
        assert singleton is None and symbol is None and one_of is None and len(in_range) == 2
        return _SymbolStrategy(
            kind="global_strategy",
            boundary=None,
            value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
        )

    else:
        raise AssertionError("Missing value descriptor")


def field(*, in_range, boundary=None, extent=None):
    """Define a *field* symbol."""
    assert (boundary is not None or extent is not None) and len(in_range) == 2
    boundary = boundary or [(abs(e[0]), abs(e[1])) for e in extent]
    extent = extent or [(-b[0], b[1]) for b in boundary]
    assert all((-b[0], b[1]) == (e[0], e[1]) for b, e in zip(boundary, extent))
    return _SymbolStrategy(
        kind="field",
        boundary=boundary,
        value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
    )


def parameter(*, one_of=None, in_range=None):
    """Define a *parameter* symbol."""
    if one_of is not None:
        assert in_range is None and isinstance(one_of, collections.abc.Sequence)
        return _SymbolStrategy(
            kind="parameter",
            boundary=None,
            value_st_factory=lambda dt: one_of_values_st(one_of).map(dt),
        )

    elif in_range is not None:
        assert one_of is None and len(in_range) == 2
        return _SymbolStrategy(
            kind="parameter",
            boundary=None,
            value_st_factory=lambda dt: scalar_value_st(dt, in_range[0], in_range[1]),
        )

    else:
        raise AssertionError("Missing value descriptor")


def none():
    """Define the symbol ``None``."""
    return _SymbolStrategy(kind="none", boundary=None, value_st_factory=lambda dt: hyp_st.none())


# ---- Custom Hypothesis strategies ----
def scalar_value_st(dtype, min_value, max_value, allow_nan=False):
    """Hypothesis strategy for `dtype` scalar values in range [min_value, max_value]."""
    allow_infinity = not (np.isfinite(min_value) and np.isfinite(max_value))

    if issubclass(dtype.type, numbers.Real):
        value_st = hyp_st.floats(
            min_value,
            max_value,
            allow_infinity=allow_infinity,
            allow_nan=allow_nan,
            width=dtype.itemsize * 8,
        )
    elif issubclass(dtype.type, numbers.Integral):
        value_st = hyp_st.integers(min_value, max_value)

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


def padded_shape_st(shape_st, extra):
    """Hypothesis strategy for extending shapes generated from a provided strategy with some extra padding."""
    return hyp_st.builds(lambda shape: tuple([d + e for d, e in zip(shape, extra)]), shape_st)


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
    tmp = hyp_np.arrays(
        dtype=dtype,
        shape=shape_strategy,
        elements=value_st_factory(dtype),
        fill=value_st_factory(dtype),
    )
    return tmp


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
