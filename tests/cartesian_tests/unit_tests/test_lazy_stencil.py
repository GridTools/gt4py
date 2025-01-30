# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the backend-agnostic build system."""

import pytest

import gt4py
from gt4py import cartesian as gt4pyc
from gt4py.cartesian.frontend.gtscript_frontend import GTScriptDefinitionError
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.cartesian.lazy_stencil import LazyStencil
from gt4py.cartesian.stencil_builder import StencilBuilder

from ..definitions import ALL_BACKENDS


def copy_stencil_definition(out_f: Field[float], in_f: Field[float]):  # type: ignore
    """Copy input into output."""
    with computation(PARALLEL), interval(...):  # type: ignore
        out_f = in_f  # type: ignore # noqa


def wrong_syntax_stencil_definition(out_f: Field[float], in_f: Field[float]):  # type: ignore
    """Contains a GTScript specific syntax error."""
    from __externals__ import undefined  # type: ignore

    with computation(PARALLEL), interval(...):  # type: ignore
        out_f = undefined(in_f)  # type: ignore # noqa


@pytest.fixture(params=["gtscript"])
def frontend(request):
    yield gt4pyc.frontend.from_name(request.param)


@pytest.fixture(params=ALL_BACKENDS)
def backend_name(request):
    yield request.param


@pytest.fixture
def backend(backend_name):
    yield gt4pyc.backend.from_name(backend_name)


def test_lazy_stencil():
    """Test lazy stencil construction."""
    builder = StencilBuilder(copy_stencil_definition).with_options(
        name="copy_stencil", module=copy_stencil_definition.__module__
    )
    lazy_s = LazyStencil(builder)

    assert lazy_s.backend.name == "numpy"


def test_lazy_syntax_check(frontend, backend):
    """Test syntax checking."""
    lazy_pass = LazyStencil(
        StencilBuilder(copy_stencil_definition, frontend=frontend, backend=backend)
    )
    lazy_pass.check_syntax()
    with pytest.raises(GTScriptDefinitionError):
        LazyStencil(
            StencilBuilder(wrong_syntax_stencil_definition, frontend=frontend, backend=backend)
        )


def test_lazy_call(frontend, backend):
    """Test that the lazy stencil is callable like the compiled stencil object."""
    import numpy

    a = gt4py.storage.from_array(
        numpy.array([[[1.0]]]), aligned_index=(0, 0, 0), backend=backend.name
    )
    b = gt4py.storage.from_array(
        numpy.array([[[0.0]]]), aligned_index=(0, 0, 0), backend=backend.name
    )
    lazy_s = LazyStencil(
        StencilBuilder(copy_stencil_definition, frontend=frontend, backend=backend).with_options(
            name="copy", module=copy_stencil_definition.__module__, rebuild=True
        )
    )
    lazy_s(b, a)
    assert b[0, 0, 0] == 1.0
