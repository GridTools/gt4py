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

from .. import definitions


class TestNode:
    def test_validation(self, invalid_sample_node_maker):
        with pytest.raises(pydantic.ValidationError):
            invalid_sample_node_maker()

    def test_mutability(self, sample_node):
        sample_node.id_ = None

    def test_inmutability(self, frozen_sample_node):
        with pytest.raises(TypeError):
            frozen_sample_node.id_ = None

    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert node_a.id_ != node_b.id_ != node_c.id_

    def test_custom_id(self, source_location, sample_node_maker):
        custom_id = "my_custom_id"
        my_node = definitions.LocationNode(id_=custom_id, loc=source_location)
        other_node = sample_node_maker()

        assert my_node.id_ == custom_id
        assert my_node.id_ != other_node.id_

        with pytest.raises(pydantic.ValidationError, match="id_"):
            definitions.LocationNode(id_=32, loc=source_location)

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
