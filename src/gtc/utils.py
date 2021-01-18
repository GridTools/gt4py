# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import collections.abc


def flatten_list(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    return list(flatten_list_iter(nested_iterables, filter_none, skip_types=skip_types))


def flatten_list_iter(nested_iterables, filter_none=False, *, skip_types=(str, bytes)):
    for item in nested_iterables:
        if isinstance(item, list) and not isinstance(item, skip_types):
            yield from flatten_list(item, filter_none)
        else:
            if item is not None or not filter_none:
                yield item


class ListTuple(collections.abc.Sequence):
    """
    A tuple of lists.

    Helper class to deal with multiple list return values.
    Supports adding (concatenating) ListTuples and (nested) list of ListTuples.

    Examples
    --------
    ListTuple([1,2],[3,4,5]) + ListTuple([6,7],[8]) == ListTuple([1,2,6,7], [3,4,5,8])

    ListTuple([1,2], [3,4,5]) + [ListTuple([6], [7]), [ListTuple([8],[9])]]
        == ListTuple([1,2,6,8], [3,4,5,7,9])
    """

    def __init__(self, *args):
        assert len(args) > 0
        assert all([isinstance(arg, list) for arg in args])
        self.tpl = args

    def __len__(self):
        return len(self.tpl)

    def __getitem__(self, key):
        return self.tpl[key]

    def _add_list_tuple(self, other):
        if len(self.tpl) != len(other.tpl):
            raise ValueError("Can only concatenate ListTuples of same arity.")
        return ListTuple(*(_self + _other for _self, _other in zip(self.tpl, other.tpl)))

    def _add_list_of_list_tuple(self, other):
        flattened = flatten_list(other)
        if all([isinstance(elem, ListTuple) for elem in flattened]):
            for elem in flattened:
                self += elem
            return self
        else:
            raise ValueError(
                "Can only concat ListTuples and lists of ListTuples, got {}".format(other)
            )

    def __add__(self, other):
        if isinstance(other, ListTuple):
            return self._add_list_tuple(other)
        elif isinstance(other, list):
            return self._add_list_of_list_tuple(other)
        else:
            raise ValueError(
                "Can only concat ListTuples and lists of ListTuples, got {}.".format(
                    type(other).__name__
                )
            )
