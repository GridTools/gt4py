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

from typing import Any, Dict

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays

from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage

from ..definitions import INTERNAL_BACKENDS


backend_list = [backend for backend in INTERNAL_BACKENDS if backend.values[0] != "debug"]


class TestExecInfo:
    @staticmethod
    def advection_def(
        in_phi: gtscript.Field[float],  # type: ignore
        in_u: gtscript.Field[float],  # type: ignore
        in_v: gtscript.Field[float],  # type: ignore
        out_phi: gtscript.Field[float],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):  # type: ignore  # noqa
            u = 0.5 * (in_u[-1, 0, 0] + in_u[0, 0, 0])  # type: ignore
            flux_x = u[0, 0, 0] * (
                in_phi[-1, 0, 0] if u[0, 0, 0] > 0 else in_phi[0, 0, 0]  # type: ignore
            )
            v = 0.5 * (in_v[0, -1, 0] + in_v[0, 0, 0])  # type: ignore
            flux_y = v[0, 0, 0] * (
                in_phi[0, -1, 0] if v[0, 0, 0] > 0 else in_phi[0, 0, 0]  # type: ignore
            )
            out_phi = (  # noqa
                in_phi - (flux_x[1, 0, 0] - flux_x[0, 0, 0]) - (flux_y[0, 1, 0] - flux_y[0, 0, 0])
            )

    @staticmethod
    def diffusion_def(
        in_phi: gtscript.Field[float], out_phi: gtscript.Field[float], *, alpha: float  # type: ignore
    ):
        with computation(PARALLEL), interval(...):  # type: ignore  # noqa
            lap1 = (
                -4 * in_phi[0, 0, 0]  # type: ignore
                + in_phi[-1, 0, 0]  # type: ignore
                + in_phi[1, 0, 0]  # type: ignore
                + in_phi[0, -1, 0]  # type: ignore
                + in_phi[0, 1, 0]  # type: ignore
            )
            lap2 = (
                -4 * lap1[0, 0, 0] + lap1[-1, 0, 0] + lap1[1, 0, 0] + lap1[0, -1, 0] + lap1[0, 1, 0]
            )
            flux_x = lap2[1, 0, 0] - lap2[0, 0, 0]
            flux_y = lap2[0, 1, 0] - lap2[0, 0, 0]
            out_phi = in_phi + alpha * (  # noqa
                flux_x[0, 0, 0] - flux_x[-1, 0, 0] + flux_y[0, 0, 0] - flux_y[0, -1, 0]
            )

    def compile_stencils(self, backend):
        self.advection = gtscript.stencil(backend=backend, definition=self.advection_def)
        self.diffusion = gtscript.stencil(backend=backend, definition=self.diffusion_def)

    def init_fields(self, data, backend):
        self.nx = data.draw(hyp_st.integers(min_value=7, max_value=32), label="nx")
        self.ny = data.draw(hyp_st.integers(min_value=7, max_value=32), label="ny")
        self.nz = data.draw(hyp_st.integers(min_value=1, max_value=32), label="nz")
        shape = (self.nx, self.ny, self.nz)

        self.in_phi = gt_storage.from_array(
            data.draw(st_arrays(dtype=float, shape=shape)),
            backend=backend,
            default_origin=(0, 0, 0),
            dtype=float,
        )
        self.in_u = gt_storage.from_array(
            data.draw(st_arrays(dtype=float, shape=shape)),
            backend=backend,
            default_origin=(0, 0, 0),
            dtype=float,
        )
        self.in_v = gt_storage.from_array(
            data.draw(st_arrays(dtype=float, shape=shape)),
            backend=backend,
            default_origin=(0, 0, 0),
            dtype=float,
        )
        self.tmp_phi = gt_storage.from_array(
            data.draw(st_arrays(dtype=float, shape=shape)),
            backend=backend,
            default_origin=(1, 1, 0),
            dtype=float,
        )
        self.out_phi = gt_storage.from_array(
            data.draw(st_arrays(dtype=float, shape=shape)),
            backend=backend,
            default_origin=(3, 3, 0),
            dtype=float,
        )
        self.alpha = 1 / 32

    def subtest_exec_info(self, exec_info):
        assert "call_start_time" in exec_info
        assert "call_end_time" in exec_info
        assert exec_info["call_end_time"] > exec_info["call_start_time"]

        assert "run_start_time" in exec_info
        assert exec_info["run_start_time"] > exec_info["call_start_time"]
        assert "run_end_time" in exec_info
        assert exec_info["run_end_time"] > exec_info["run_start_time"]
        assert exec_info["call_end_time"] > exec_info["run_end_time"]

        if gt_backend.from_name(self.backend).languages["computation"] == "c++":
            assert "run_cpp_start_time" in exec_info
            assert "run_cpp_end_time" in exec_info
            # note: do not compare the outputs of python and c++ stopwatches
            # consider only deltas

        assert "origin" in exec_info
        assert exec_info["origin"] == {
            "_all_": (3, 3, 0),
            "in_phi": (3, 3, 0),
            "out_phi": (3, 3, 0),
        }

        assert "domain" in exec_info
        assert exec_info["domain"] == (self.nx - 6, self.ny - 6, self.nz)

    def subtest_stencil_info(self, exec_info, stencil_info, last_called_stencil=False):
        assert "ncalls" in stencil_info
        assert stencil_info["ncalls"] == self.nt

        assert "call_start_time" in stencil_info
        assert "call_end_time" in stencil_info
        assert stencil_info["call_end_time"] > stencil_info["call_start_time"]
        assert "call_time" in stencil_info
        assert "total_call_time" in stencil_info
        assert np.isclose(
            stencil_info["call_time"],
            stencil_info["call_end_time"] - stencil_info["call_start_time"],
        )
        if self.nt == 1:
            assert stencil_info["total_call_time"] == stencil_info["call_time"]
        else:
            assert stencil_info["total_call_time"] > stencil_info["call_time"]
        if last_called_stencil:
            assert stencil_info["call_start_time"] == exec_info["call_start_time"]
            assert stencil_info["call_end_time"] == exec_info["call_end_time"]

        assert "run_time" in stencil_info
        if last_called_stencil:
            assert np.isclose(
                stencil_info["run_time"],
                exec_info["run_end_time"] - exec_info["run_start_time"],
            )
        assert stencil_info["call_time"] > stencil_info["run_time"]
        assert "total_run_time" in stencil_info
        if self.nt == 1:
            assert stencil_info["total_run_time"] == stencil_info["run_time"]
        else:
            assert stencil_info["total_run_time"] > stencil_info["run_time"]

        if gt_backend.from_name(self.backend).languages["computation"] == "c++":
            assert "run_cpp_time" in stencil_info
            if last_called_stencil:
                assert np.isclose(
                    stencil_info["run_cpp_time"],
                    exec_info["run_cpp_end_time"] - exec_info["run_cpp_start_time"],
                )
            assert stencil_info["run_time"] > stencil_info["run_cpp_time"]
            assert "total_run_cpp_time" in stencil_info
            if self.nt == 1:
                assert stencil_info["total_run_cpp_time"] == stencil_info["run_cpp_time"]
            else:
                assert stencil_info["total_run_cpp_time"] > stencil_info["run_cpp_time"]

    @given(data=hyp_st.data())
    @pytest.mark.parametrize("backend", backend_list)
    def test_backcompatibility(self, data, backend):
        # set backend as instance attribute
        self.backend = backend

        # compile stencils
        self.compile_stencils(backend)

        # initialize storages and parameters
        self.init_fields(data, backend)

        # initialize exec_info
        exec_info: Dict[str, Any] = {}

        # run (sequential-splitting mode)
        self.nt = data.draw(hyp_st.integers(1, 64), label="nt")
        for _ in range(self.nt):
            self.advection(
                self.in_phi,
                self.in_u,
                self.in_v,
                self.tmp_phi,
                origin=(1, 1, 0),
                domain=(self.nx - 2, self.ny - 2, self.nz),
                exec_info=exec_info,
            )
            self.diffusion(
                self.in_phi,
                self.out_phi,
                alpha=self.alpha,
                origin=(3, 3, 0),
                domain=(self.nx - 6, self.ny - 6, self.nz),
                exec_info=exec_info,
            )

        # check
        self.subtest_exec_info(exec_info)
        assert "__aggregate_data" in exec_info
        assert exec_info["__aggregate_data"] is False
        assert type(self.advection).__name__ not in exec_info
        assert type(self.diffusion).__name__ not in exec_info

    @given(data=hyp_st.data())
    @pytest.mark.parametrize("backend", backend_list)
    def test_aggregate(self, data, backend):
        # set backend as instance attribute
        self.backend = backend

        # compile stencils
        self.compile_stencils(backend)

        # initialize storages and parameters
        self.init_fields(data, backend)

        # initialize exec_info
        exec_info: Dict[str, Any] = {"__aggregate_data": True}

        # run (sequential-splitting mode)
        self.nt = data.draw(hyp_st.integers(1, 64), label="nt")
        for _ in range(self.nt):
            self.advection(
                self.in_phi,
                self.in_u,
                self.in_v,
                self.tmp_phi,
                origin=(1, 1, 0),
                domain=(self.nx - 2, self.ny - 2, self.nz),
                exec_info=exec_info,
            )
            self.diffusion(
                self.in_phi,
                self.out_phi,
                alpha=self.alpha,
                origin=(3, 3, 0),
                domain=(self.nx - 6, self.ny - 6, self.nz),
                exec_info=exec_info,
            )

        # check
        self.subtest_exec_info(exec_info)
        assert type(self.advection).__name__ in exec_info
        self.subtest_stencil_info(exec_info, exec_info[type(self.advection).__name__])
        assert type(self.diffusion).__name__ in exec_info
        self.subtest_stencil_info(
            exec_info, exec_info[type(self.diffusion).__name__], last_called_stencil=True
        )


if __name__ == "__main__":
    pytest.main([__file__])
