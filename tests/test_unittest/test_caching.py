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

import pytest

import gt4py
from gt4py.gtscript import PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder


# type ignores in stencils are because mypy does not yet
# deal with gtscript types well


def simple_stencil(field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        field += 1  # type: ignore


def simple_stencil_same(field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        field += 1  # type: ignore


def simple_stencil_with_doc(field: Field[float]):  # type: ignore
    """Increment the in/out field by one."""
    with computation(PARALLEL), interval(...):  # type: ignore
        field += 1  # type: ignore


@pytest.fixture
def builder():
    """Preconfigure builder so everything but definition has defaults."""

    def make_builder(definition, backend_name="debug", module=None):
        """Make a builder instance a definition and default params."""
        return StencilBuilder(
            definition,
            backend=gt4py.backend.from_name(backend_name),
            options=gt4py.definitions.BuildOptions(name="foo", module=module or f"{__name__}"),
        )

    return make_builder


def test_jit_properties(builder):
    builder = builder(simple_stencil).with_caching("jit")
    jit_caching = builder.caching

    # check state of properties directly after init, cache_info file has not been written
    assert not jit_caching.cache_info
    assert jit_caching.options_id

    # check properties of the calculated stencil_id
    stencil_id = jit_caching.stencil_id
    assert stencil_id.qualified_name == f"{__name__}.foo"
    assert stencil_id.version
    assert jit_caching.cache_info_path.parent == builder.module_path.parent
    assert stencil_id.version in jit_caching.module_postfix
    assert stencil_id.version in jit_caching.class_name


def stencil_fingerprints_are_equal(builder_a, builder_b):
    return builder_a.caching.stencil_id.version == builder_b.caching.stencil_id.version


def could_load_stencil_from_cache(builder, catch_exceptions=False):
    return builder.caching.is_cache_info_available_and_consistent(
        validate_hash=True, catch_exceptions=catch_exceptions
    )


def test_jit_version(builder):
    simple_stencil_same.__name__ = "simple_stencil"

    # create builders with jit caching strategy
    original = builder(simple_stencil, "gtx86").with_caching("jit")
    duplicate = builder(simple_stencil, "gtx86").with_caching("jit")
    samebody = builder(simple_stencil_same, "gtx86").with_caching("jit")
    withdoc = builder(simple_stencil_with_doc, "gtx86").with_caching("jit")

    assert stencil_fingerprints_are_equal(original, duplicate)
    assert not stencil_fingerprints_are_equal(original, samebody)
    assert not stencil_fingerprints_are_equal(original, withdoc)

    original.backend.generate()
    # cache_info file was written and can now be read
    assert original.caching.cache_info
    assert could_load_stencil_from_cache(original)
    assert could_load_stencil_from_cache(duplicate)

    duplicate.backend.generate()
    assert could_load_stencil_from_cache(original)

    withdoc.backend.generate()
    assert could_load_stencil_from_cache(withdoc)

    original.definition.__doc__ = "Added docstring." ""
    assert not could_load_stencil_from_cache(original, catch_exceptions=True)
    assert not could_load_stencil_from_cache(duplicate, catch_exceptions=True)
    # fingerprint has changed and with it the file paths, new cache_info file does not exist.
    assert not original.caching.cache_info
    assert not duplicate.caching.cache_info


def test_jit_extrainfo(builder):
    builder = builder(simple_stencil, "gtx86").with_caching("jit")
    builder.backend.generate()

    assert "pyext_file_path" in builder.caching.cache_info
    assert "pyext_md5" in builder.caching.cache_info


def test_nocaching_paths(builder, tmp_path):
    builder = builder(simple_stencil).with_caching("nocaching", output_path=tmp_path)
    no_caching = builder.caching

    # the paths are all within the the given path or None
    assert no_caching.root_path == tmp_path
    assert no_caching.backend_root_path == tmp_path
    assert no_caching.cache_info_path is None


def assert_nocaching_gtcpp_source_file_tree_conforms_to_expectations(root_path, stencil_name):
    stencil_path = root_path / stencil_name
    build_path = stencil_path / f"{stencil_name}_pyext_BUILD"

    module_path = stencil_path / f"{stencil_name}.py"
    comp_cpp_path = build_path / "computation.cpp"
    comp_hpp_path = build_path / "computation.hpp"
    comp_bindings_path = build_path / "bindings.cpp"

    assert module_path.exists()
    assert comp_cpp_path.exists()
    assert comp_hpp_path.exists()
    assert comp_bindings_path.exists()


def test_nocaching_generate(builder, tmp_path):
    # generate pure python stencil and ensure the module is in the right place
    builder_d = builder(simple_stencil, backend_name="debug", module="foo_d").with_caching(
        "nocaching", output_path=tmp_path
    )
    builder_d.backend.generate()
    assert tmp_path.joinpath("foo_d", "foo", "foo.py").exists()

    # generate a GT C++ extension stencil and check the locations of the source files
    builder_g = builder(simple_stencil, backend_name="gtx86", module="foo_g").with_caching(
        "nocaching", output_path=tmp_path
    )
    builder_g.backend.generate()

    assert_nocaching_gtcpp_source_file_tree_conforms_to_expectations(tmp_path / "foo_g", "foo")
