# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the names exported by the public `gt4py.next` namespace."""

import pytest

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners import dace as dace_runners, gtfn as gtfn_runners


def test_backends_are_exported():
    for name in ["gtfn_cpu", "gtfn_gpu", "dace_cpu", "dace_gpu", "itir_python"]:
        assert name in gtx.__all__
        assert name in dir(gtx)
        assert isinstance(getattr(gtx, name), gtx_backend.Backend)


def test_gtfn_backends():
    assert gtx.gtfn_cpu is gtfn_runners.run_gtfn
    assert gtx.gtfn_gpu is gtfn_runners.run_gtfn_gpu


def test_dace_backends():
    assert gtx.dace_cpu is dace_runners.run_dace_cpu
    assert gtx.dace_gpu is dace_runners.run_dace_gpu


def test_dace_backends_target_the_expected_device():
    # Like `gtfn_gpu`, `dace_gpu` can be accessed on machines without a GPU, it only
    # fails when it is used to allocate fields (see `InvalidGPUFieldBufferAllocator`).
    assert gtx.dace_cpu.name == "run_dace_cpu_opt"
    assert gtx.dace_gpu.name == "run_dace_gpu_opt"


def test_dace_noopt_backends_are_not_exported():
    for name in ["dace_cpu_noopt", "dace_gpu_noopt", "run_dace_cpu", "run_dace_gpu"]:
        assert name not in gtx.__all__
        assert not hasattr(gtx, name)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_backend"):
        gtx.no_such_backend  # noqa: B018
