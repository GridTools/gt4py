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

"""Test the backend-agnostic build system."""

import pytest

import gt4py
from gt4py.frontend.gtscript_frontend import GTScriptDefinitionError
from gt4py.gtscript import PARALLEL, Field, computation, interval
from gt4py.lazy_stencil import LazyStencil
from gt4py.stencil_builder import StencilBuilder


def has_cupy():
    try:
        import cupy  # noqa

        return True
    except ImportError:
        return False


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
    yield gt4py.frontend.from_name(request.param)


@pytest.fixture(
    params=[
        "debug",
        "numpy",
        "gtx86",
        "gtmc",
        pytest.param(
            "gtcuda", marks=pytest.mark.skipif(not has_cupy(), reason="cupy not installed")
        ),
    ]
)
def backend_name(request):
    yield request.param


@pytest.fixture
def backend(backend_name):
    yield gt4py.backend.from_name(backend_name)


def test_lazy_stencil():
    """Test lazy stencil construction."""
    builder = StencilBuilder(copy_stencil_definition).with_options(
        name="copy_stencil", module=copy_stencil_definition.__module__
    )
    lazy_s = LazyStencil(builder)

    assert lazy_s.backend.name == "debug"


def test_lazy_syntax_check(frontend, backend):
    """Test syntax checking."""
    lazy_pass = LazyStencil(
        StencilBuilder(copy_stencil_definition, frontend=frontend, backend=backend)
    )
    lazy_fail = LazyStencil(
        StencilBuilder(wrong_syntax_stencil_definition, frontend=frontend, backend=backend)
    )
    lazy_pass.check_syntax()
    with pytest.raises(GTScriptDefinitionError):
        lazy_fail.check_syntax()


def test_lazy_call(frontend, backend):
    """Test that the lazy stencil is callable like the compiled stencil object."""
    import numpy

    a = gt4py.storage.storage(numpy.array([[[1.0]]]), halo=(0, 0, 0), defaults=backend.name)
    b = gt4py.storage.storage(numpy.array([[[0.0]]]), halo=(0, 0, 0), defaults=backend.name)
    lazy_s = LazyStencil(
        StencilBuilder(copy_stencil_definition, frontend=frontend, backend=backend).with_options(
            name="copy", module=copy_stencil_definition.__module__, rebuild=True
        )
    )
    lazy_s(b, a)
    assert b[0, 0, 0] == 1.0
