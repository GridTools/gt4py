# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from gt4py import cartesian as gt4pyc
from gt4py.cartesian.gtscript import FORWARD, IJ, IJK, PARALLEL, Field, computation, interval
from gt4py.cartesian.stencil_builder import StencilBuilder


# type ignores in stencils are because mypy does not yet
# deal with gtscript types well


def simple_stencil(field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):
        field += 1


def simple_stencil_same(field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):
        field += 1


def simple_stencil_with_doc(field: Field[float]):  # type: ignore
    """Increment the in/out field by one."""
    with computation(PARALLEL), interval(...):
        field += 1


@pytest.fixture
def builder() -> StencilBuilder:
    """Preconfigure builder so everything but definition has defaults."""

    def make_builder(definition, backend_name="numpy", module=None):
        """Make a builder instance a definition and default params."""
        return StencilBuilder(
            definition,
            backend=gt4pyc.backend.from_name(backend_name),
            options=gt4pyc.definitions.BuildOptions(name="foo", module=module or f"{__name__}"),
        )

    return make_builder


def test_jit_properties(builder: StencilBuilder) -> None:
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


def stencil_fingerprints_are_equal(builder_a: StencilBuilder, builder_b: StencilBuilder) -> None:
    return builder_a.caching.stencil_id.version == builder_b.caching.stencil_id.version


def could_load_stencil_from_cache(builder: StencilBuilder, catch_exceptions: bool = False) -> None:
    return builder.caching.is_cache_info_available_and_consistent(
        validate_hash=True, catch_exceptions=catch_exceptions
    )


def test_jit_version(builder: StencilBuilder) -> None:
    simple_stencil_same.__name__ = "simple_stencil"

    # create builders with jit caching strategy
    original = builder(simple_stencil, "gt:cpu_kfirst").with_caching("jit")
    duplicate = builder(simple_stencil, "gt:cpu_kfirst").with_caching("jit")
    samebody = builder(simple_stencil_same, "gt:cpu_kfirst").with_caching("jit")
    withdoc = builder(simple_stencil_with_doc, "gt:cpu_kfirst").with_caching("jit")

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

    original.definition.__doc__ = "Added docstring."
    assert not could_load_stencil_from_cache(original, catch_exceptions=True)
    assert not could_load_stencil_from_cache(duplicate, catch_exceptions=True)
    # fingerprint has changed and with it the file paths, new cache_info file does not exist.
    assert not original.caching.cache_info
    assert not duplicate.caching.cache_info


def test_jit_extrainfo(builder: StencilBuilder) -> None:
    builder = builder(simple_stencil, "gt:cpu_kfirst").with_caching("jit")
    builder.backend.generate()

    assert "pyext_file_path" in builder.caching.cache_info
    assert "pyext_md5" in builder.caching.cache_info


def test_nocaching_paths(builder: StencilBuilder, tmp_path: Path) -> None:
    builder = builder(simple_stencil).with_caching("nocaching", output_path=tmp_path)
    no_caching = builder.caching

    # the paths are all within the the given path or None
    assert no_caching.root_path == tmp_path
    assert no_caching.backend_root_path == tmp_path
    assert no_caching.cache_info_path is None


def assert_nocaching_gtcpp_source_file_tree_conforms_to_expectations(
    root_path: Path, stencil_name: str
) -> None:
    stencil_path = root_path / stencil_name
    build_path = stencil_path / f"{stencil_name}_pyext_BUILD"

    module_path = stencil_path / f"{stencil_name}.py"
    comp_hpp_path = build_path / "computation.hpp"
    comp_bindings_path = build_path / "bindings.cpp"

    assert module_path.exists()
    assert comp_hpp_path.exists()
    assert comp_bindings_path.exists()


def test_nocaching_generate(builder: StencilBuilder, tmp_path: Path) -> None:
    # generate pure python stencil and ensure the module is in the right place
    builder_d = builder(simple_stencil, backend_name="numpy", module="foo_d").with_caching(
        "nocaching", output_path=tmp_path
    )
    builder_d.backend.generate()
    assert tmp_path.joinpath("foo_d", "foo", "foo.py").exists()

    # generate a GT C++ extension stencil and check the locations of the source files
    builder_g = builder(simple_stencil, backend_name="gt:cpu_kfirst", module="foo_g").with_caching(
        "nocaching", output_path=tmp_path
    )
    builder_g.backend.generate()

    assert_nocaching_gtcpp_source_file_tree_conforms_to_expectations(tmp_path / "foo_g", "foo")


def test_compiler_optimizations(builder: StencilBuilder) -> None:
    builder_1 = builder(simple_stencil, backend_name="gt:cpu_kfirst")
    builder_2 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={
            "opt_level": gt4pyc.config.GT4PY_COMPILE_OPT_LEVEL,
            "extra_opt_flags": gt4pyc.config.GT4PY_EXTRA_COMPILE_OPT_FLAGS,
        }
    )

    builder_1.backend.generate()
    builder_2.backend.generate()

    assert not stencil_fingerprints_are_equal(builder_1, builder_2)


def test_different_opt_levels(builder: StencilBuilder) -> None:
    builder_1 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"opt_level": "0"}
    )
    builder_2 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"opt_level": "1"}
    )

    builder_1.backend.generate()
    builder_2.backend.generate()

    assert not stencil_fingerprints_are_equal(builder_1, builder_2)


def test_different_extra_opt_flags(builder: StencilBuilder) -> None:
    builder_1 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"opt_level": "0"}
    )
    builder_2 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"opt_level": "0", "extra_opt_flags": "-ftree-vectorize"}
    )

    builder_1.backend.generate()
    builder_2.backend.generate()

    assert not stencil_fingerprints_are_equal(builder_1, builder_2)


def test_debug_mode(builder: StencilBuilder) -> None:
    builder_1 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"debug_mode": True, "opt_level": "0"}
    )
    builder_2 = builder(simple_stencil, backend_name="gt:cpu_kfirst").with_changed_options(
        backend_opts={"debug_mode": False, "opt_level": "0"}
    )

    builder_1.backend.generate()
    builder_2.backend.generate()

    assert not stencil_fingerprints_are_equal(builder_1, builder_2)


def stencil_tmp_IJ(in_field: Field[float], out_field: Field[float]):  # type: ignore
    with computation(FORWARD), interval(0, 1):
        tmp: Field[IJ, float] = 0  # type: ignore

    with computation(FORWARD), interval(...):
        tmp = in_field + tmp

    with computation(PARALLEL), interval(...):
        out_field = tmp * 2


def stencil_tmp_modified(in_field: Field[float], out_field: Field[float]):  # type: ignore
    with computation(FORWARD), interval(0, 1):
        tmp: Field[IJK, float] = 0  # type: ignore

    with computation(FORWARD), interval(...):
        tmp = in_field + tmp

    with computation(PARALLEL), interval(...):
        out_field = tmp * 2


def test_temporaries_annotation(builder: StencilBuilder) -> None:
    """
    The scenario in this test is that users place an annotation for temporaries
    inside the stencil (e.g. 2d temporaries). The user is then assumed to
    change/remove that annotation, which should invalidate the cache.
    """
    # setup "initial stencil"
    initial_stencil = builder(stencil_tmp_IJ)
    # trigger writing cache info to disk
    initial_stencil.backend.generate()

    assert initial_stencil.caching.cache_info
    assert could_load_stencil_from_cache(initial_stencil)

    # setup "modified stencil"
    modified_stencil = builder(stencil_tmp_modified)

    # access .definition to trigger stencil annotation
    modified_stencil.definition
    assert hasattr(stencil_tmp_modified, "_gtscript_")

    # patch stencil s.t. the only change are inside the stencil
    stencil_tmp_modified._gtscript_["canonical_ast"] = stencil_tmp_modified._gtscript_[
        "canonical_ast"
    ].replace("stencil_tmp_modified", "stencil_tmp_IJ")

    assert not could_load_stencil_from_cache(modified_stencil, catch_exceptions=True)
