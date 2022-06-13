# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
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


from __future__ import annotations

from typing import List, Union

import pytest

import eve


class SampleTree(eve.concepts.Node):
    children: List[Union[SampleTree, int]]


def _make_tree(values_list):
    children = [_make_tree(item) if isinstance(item, list) else item for item in values_list]
    SampleTree.update_forward_refs()
    return SampleTree(children=children)


def _collect_shallow_tree_nodes(nodes_list):
    # Collect lower levels of the tree
    collected = []
    for item in nodes_list:
        if isinstance(item, eve.Node) and any(isinstance(c, eve.Node) for c in item.children):
            continue
        if isinstance(item, list) and any(isinstance(c, eve.Node) for c in item):
            continue
        collected.append(item)
    return collected


@pytest.fixture
def dfs_ordered_tree():
    dfs_ordered_values = [
        [[1, 2, 3], [4, 5, 6]],
        [7, [8, 9], 10],
        [11, 12, [13, [14, [15, [16, 17]]], 18]],
    ]
    yield _make_tree(dfs_ordered_values)


@pytest.fixture
def bfs_ordered_tree():
    bfs_ordered_values = [[[1], [2], [3], [4, 5]], [[6, 7], [8]], [[9], [10, 11], [12]]]
    yield _make_tree(bfs_ordered_values)


def test_iter_tree_pre(dfs_ordered_tree):
    values = [
        value for value in eve.trees.pre_walk_values(dfs_ordered_tree) if isinstance(value, int)
    ]
    assert values == list(sorted(values))

    # Test if collected values and nodes are sorted in the right order
    simple_levels = _collect_shallow_tree_nodes(values)
    for i, item in enumerate(simple_levels):
        if isinstance(item, eve.Node):
            assert item.children == simple_levels[i + 1]
        elif isinstance(item, list):
            assert item == simple_levels[i + 1 : i + 1 + len(item)]


def test_iter_tree_post(dfs_ordered_tree):
    values = [
        value for value in eve.trees.post_walk_values(dfs_ordered_tree) if isinstance(value, int)
    ]
    assert values == list(sorted(values))

    # Test if collected values and nodes are sorted in the right order
    simple_levels = _collect_shallow_tree_nodes(values)
    for i, item in enumerate(simple_levels):
        if isinstance(item, eve.Node):
            assert item.children == simple_levels[i - 1]
        elif isinstance(item, list):
            assert item == simple_levels[i - len(item) : i]


def test_iter_tree_levels(bfs_ordered_tree):
    values = [
        value for value in eve.trees.bfs_walk_values(bfs_ordered_tree) if isinstance(value, int)
    ]
    assert values == list(sorted(values))


@pytest.mark.parametrize("tree", [bfs_ordered_tree, dfs_ordered_tree])
def test_iter_tree(tree):
    traversals = []
    for order in eve.trees.TraversalOrder:
        values = [value for value in eve.trees.walk_items(tree, order)]
        assert all(isinstance(v, tuple) for v in values)
        traversals.append(values)
        traversals.append([value for value in eve.trees.walk_values(tree, order)])

    assert all(len(traversals[0]) == len(t) for t in traversals)
