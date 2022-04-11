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


import pydantic
import pytest


class TestNode:
    def test_validation(self, invalid_sample_node_maker):
        with pytest.raises(pydantic.ValidationError):
            invalid_sample_node_maker()

    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert id(node_a) != id(node_b) != id(node_c)

    def test_impl_fields(self, sample_node):
        impl_names = set(name for name, _ in sample_node.iter_impl_fields())

        assert all(name.endswith("_") and not name.endswith("__") for name in impl_names)
        assert (
            set(
                name
                for name in sample_node.__fields__.keys()
                if name.endswith("_") and not name.endswith("__")
            )
            == impl_names
        )

    def test_children(self, sample_node):
        impl_field_names = set(name for name, _ in sample_node.iter_impl_fields())
        children_names = set(name for name, _ in sample_node.iter_children())
        public_names = impl_field_names | children_names
        field_names = set(sample_node.__fields__.keys())

        assert not any(name.endswith("__") for name in children_names)
        assert not any(name.endswith("_") for name in children_names)

        assert public_names <= field_names
        assert all(name.endswith("_") for name in field_names - public_names)

        assert all(
            node1 is node2
            for (name, node1), node2 in zip(
                sample_node.iter_children(), sample_node.iter_children_values()
            )
        )

    def test_node_metadata(self, sample_node):
        assert all(
            name in sample_node.__node_impl_fields__ for name, _ in sample_node.iter_impl_fields()
        )
        assert all(
            isinstance(metadata, dict)
            and isinstance(metadata["definition"], pydantic.fields.ModelField)
            for metadata in sample_node.__node_impl_fields__.values()
        )

        assert all(name in sample_node.__node_children__ for name, _ in sample_node.iter_children())
        assert all(
            isinstance(metadata, dict)
            and isinstance(metadata["definition"], pydantic.fields.ModelField)
            for metadata in sample_node.__node_children__.values()
        )

    def test_serialization_roundtrip(self, sample_node):
        assert type(sample_node).parse_raw(sample_node.json()) == sample_node
