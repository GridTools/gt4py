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

"""Integration tests for StencilObjects."""

import typing
from typing import Any, Dict

import pytest

from gt4py import storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval

from cartesian_tests.definitions import ALL_BACKENDS
from cartesian_tests.utils import OriginWrapper


@pytest.mark.parametrize("backend", ["numpy"])
def test_stencil_object_cache(backend: str):
    @gtscript.stencil(backend=backend)
    @typing.no_type_check
    def stencil(in_field: Field[float], out_field: Field[float], *, offset: float):
        with computation(PARALLEL), interval(...):
            out_field = (  # noqa: F841 [unused-variable]
                in_field + offset
            )

    shape = (4, 4, 4)
    in_storage = OriginWrapper(
        array=gt_storage.ones(backend=backend, aligned_index=(0, 0, 0), shape=shape, dtype=float),
        origin=(0, 0, 0),
    )
    out_storage = OriginWrapper(
        array=gt_storage.ones(backend=backend, aligned_index=(0, 0, 0), shape=shape, dtype=float),
        origin=(0, 0, 0),
    )

    def runit(*args, **kwargs) -> float:
        exec_info: Dict[str, Any] = {}
        stencil(*args, **kwargs, exec_info=exec_info)
        run_time: float = exec_info["run_end_time"] - exec_info["run_start_time"]
        call_time: float = exec_info["call_run_end_time"] - exec_info["call_run_start_time"]
        return call_time - run_time

    base_time = runit(in_storage, out_storage, offset=1.0)
    fast_time = runit(in_storage, out_storage, offset=1.0)
    assert fast_time < base_time

    # When an origin changes, it needs to recompute more, so the time should increase
    other_out_storage = OriginWrapper(
        array=gt_storage.ones(backend=backend, aligned_index=(1, 0, 0), shape=shape, dtype=float),
        origin=(1, 0, 0),
    )
    other_origin_time = runit(in_storage, other_out_storage, offset=1.0)
    assert other_origin_time > fast_time

    # When the cache is cleared, everything is recomputed and the time will increase
    assert len(stencil._domain_origin_cache) > 0
    stencil.clean_call_args_cache()
    assert len(stencil._domain_origin_cache) == 0
    cleaned_cache_time = runit(in_storage, out_storage, offset=1.0)
    assert cleaned_cache_time > fast_time


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_warning_for_unsupported_backend_option(backend):
    with pytest.warns(RuntimeWarning, match="Unknown option"):

        @gtscript.stencil(backend=backend, **{"this_option_is_not_supported": "foo"})
        def foo(f: Field[float]):
            with computation(PARALLEL), interval(...):  # type: ignore
                f = 42.0  # noqa F841
