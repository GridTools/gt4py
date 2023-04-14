# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

"""Test the backend-agnostic build system."""


import time

import pytest

import gt4py
from gt4py import cartesian as gt4pyc
from gt4py.cartesian import gtscript
from gt4py.cartesian.frontend.gtscript_frontend import GTScriptDefinitionError
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.cartesian.lazy_stencil import LazyStencil
from gt4py.cartesian.stencil_builder import FUTURES_REGISTRY, StencilBuilder, wait_all

from ..definitions import ALL_BACKENDS


@pytest.fixture(scope="function")
def reset_async_executor():
    for f in FUTURES_REGISTRY.values():
        if f is not None:
            f.cancel()
            if not f.cancelled():
                f.wait()
    FUTURES_REGISTRY.clear()


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


@pytest.mark.parametrize("build_async", [False, True])
def test_lazy_call(frontend, backend, build_async):
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
            name="copy",
            module=copy_stencil_definition.__module__,
            rebuild=True,
            build_async=build_async,
        )
    )
    lazy_s(b, a)
    assert b[0, 0, 0] == 1.0


class TestAsyncConsistency:
    def test_same_key(self, backend, reset_async_executor):
        assert len(FUTURES_REGISTRY) == 0
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="numpy", rebuild=True, eager="async"
        )
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="numpy", rebuild=True, eager="async"
        )
        assert len(FUTURES_REGISTRY) == 1

    def test_separate_keys(self, backend, reset_async_executor):
        assert len(FUTURES_REGISTRY) == 0
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="numpy", rebuild=True, eager="async"
        )
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="dace:cpu", rebuild=True, eager="async"
        )
        assert len(FUTURES_REGISTRY) == 2

    def test_barrier(self, reset_async_executor):
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="gt:cpu_kfirst", rebuild=True, eager="async"
        )
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend="dace:cpu", rebuild=True, eager="async"
        )

        wait_all()

        assert all(f.future.done() for f in FUTURES_REGISTRY.values())


class TestAsyncTiming:
    def test_async_submit_fast(self, backend, reset_async_executor):
        """Test that the async lazy_stencil call actually only schedules the build by comparing time to a full build."""
        start = time.perf_counter()
        gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend=backend.name, rebuild=True, eager="async"
        )
        mid = time.perf_counter()
        gtscript.stencil(
            definition=copy_stencil_definition, backend=backend.name, rebuild=True, eager=True
        )
        end = time.perf_counter()

        submit_time = mid - start
        full_build_time = end - mid
        assert full_build_time > 5 * submit_time

    def test_async_finished_fast(self, backend, reset_async_executor):
        """Test caching after waiting for async lazy_stencil call."""
        start = time.perf_counter()
        ls = gtscript.lazy_stencil(
            definition=copy_stencil_definition, backend=backend.name, rebuild=True, eager=False
        )
        mid1 = time.perf_counter()
        future = ls.builder.build_async()
        future.wait()
        mid2 = time.perf_counter()
        gtscript.stencil(definition=copy_stencil_definition, backend=backend.name, eager=True)
        end = time.perf_counter()

        submit_time = mid1 - start
        wait_time = mid2 - mid1
        cached_build_time = end - mid2
        assert wait_time > 5 * submit_time
        assert wait_time > 5 * cached_build_time
