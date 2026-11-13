# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import threading

import array_api_compat.numpy as compat_numpy
import array_api_strict as xs
import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.embedded import array_libraries, array_ops, nd_array_field
from gt4py.next.ffront import fbuiltins


D = common.Dimension("D")


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(array_ops, "_ops_types", dict(array_ops._ops_types))
    monkeypatch.setattr(array_ops, "_ops_by_namespace", {})
    monkeypatch.setattr(array_ops, "_ops_by_array_type", {})


def _strict_field(values, dtype=xs.float64):
    return common._field(xs.asarray(values, dtype=dtype), domain=common.domain({D: len(values)}))


def test_unregistered_namespace_uses_standard_ops():
    field = _strict_field([1.0, 2.0])

    assert type(array_ops.of(field.ndarray)) is array_ops.ArrayOps
    assert field.array_ns is xs
    assert isinstance(field, nd_array_field.NdArrayField)


def test_registered_namespaces():
    assert type(array_ops.of(np.empty(1))) is array_libraries.numpy_ops.NumPyOps
    assert array_ops.of(np.empty(1)).array_ns is np


def test_elementwise_function_override(isolated_registry):
    class Ops(array_ops.ArrayOps):
        def _elementwise_function(self, name):
            if name == "sin":
                return lambda x: x + 100.0
            return super()._elementwise_function(name)

    array_ops.register(xs.__name__, Ops)
    field = _strict_field([1.0, 2.0])

    assert np.allclose(fbuiltins.sin(field).asnumpy(), [101.0, 102.0])
    assert np.allclose(fbuiltins.cos(field).asnumpy(), np.cos([1.0, 2.0]))


def test_builtin_override(isolated_registry):
    def where_override(cond, x, y):
        return "overridden"

    class Ops(array_ops.ArrayOps):
        builtin_overrides = {fbuiltins.where: where_override}

    array_ops.register(xs.__name__, Ops)
    field = _strict_field([1.0, 2.0])
    cond = _strict_field([True, False], dtype=xs.bool)

    assert fbuiltins.where(cond, field, field) == "overridden"
    assert np.allclose(fbuiltins.minimum(field, 1.5).asnumpy(), [1.0, 1.5])


def test_later_registration_takes_precedence(isolated_registry):
    class Ops(array_libraries.numpy_ops.NumPyOps):
        pass

    array_ops.register(compat_numpy.__name__, Ops)

    assert type(array_ops.of(np.empty(1))) is Ops


@pytest.mark.requires_torch
def test_concurrent_first_use_gets_one_torch_ops(isolated_registry):
    import torch

    from gt4py.next.embedded.array_libraries import torch_ops

    num_threads = 8
    barrier = threading.Barrier(num_threads)
    results = [None] * num_threads

    def first_use(i):
        barrier.wait()
        results[i] = array_ops.of(torch.empty(1))

    threads = [threading.Thread(target=first_use, args=(i,)) for i in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(type(ops) is torch_ops.TorchOps for ops in results)
    assert all(ops is results[0] for ops in results)


@pytest.mark.requires_torch
def test_torch_ops_registered_without_importing_torch():
    code = (
        "import sys, gt4py.next\n"
        "assert 'gt4py.next.embedded.array_libraries.torch_ops' in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "import torch\n"
        "from gt4py.next.embedded import array_ops\n"
        "assert type(array_ops.of(torch.empty(1))).__name__ == 'TorchOps'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
