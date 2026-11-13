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
import types

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
    monkeypatch.setattr(array_ops, "_ops_by_namespace", dict(array_ops._ops_by_namespace))
    monkeypatch.setattr(array_ops, "_ops_by_array_type", {})
    monkeypatch.setattr(array_ops, "_registering_modules", dict(array_ops._registering_modules))


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

    array_ops.register(Ops(xs))
    field = _strict_field([1.0, 2.0])

    assert np.allclose(fbuiltins.sin(field).asnumpy(), [101.0, 102.0])
    assert np.allclose(fbuiltins.cos(field).asnumpy(), np.cos([1.0, 2.0]))


def test_builtin_override(isolated_registry):
    def where_override(cond, x, y):
        return "overridden"

    class Ops(array_ops.ArrayOps):
        builtin_overrides = {fbuiltins.where: where_override}

    array_ops.register(Ops(xs))
    field = _strict_field([1.0, 2.0])
    cond = _strict_field([True, False], dtype=xs.bool)

    assert fbuiltins.where(cond, field, field) == "overridden"
    assert np.allclose(fbuiltins.minimum(field, 1.5).asnumpy(), [1.0, 1.5])


def test_later_registration_takes_precedence(isolated_registry):
    class Ops(array_libraries.numpy_ops.NumPyOps):
        pass

    array_ops.register(Ops(compat_numpy))

    assert type(array_ops.of(np.empty(1))) is Ops


def test_registration_on_first_use_waits_for_a_concurrent_registration(
    isolated_registry, monkeypatch
):
    xp = types.ModuleType("lazily_registered_namespace")
    array_ops.register_on_first_use(xp.__name__, "lazily_registered_ops")
    other_thread_result = []
    other_thread = threading.Thread(
        target=lambda: other_thread_result.append(array_ops.for_namespace(xp))
    )

    class Ops(array_ops.ArrayOps):
        pass

    def import_module(name):
        assert name == "lazily_registered_ops"
        other_thread.start()
        other_thread.join(timeout=0.5)
        array_ops.register(Ops(xp))

    monkeypatch.setattr(array_ops, "importlib", types.SimpleNamespace(import_module=import_module))

    assert type(array_ops.for_namespace(xp)) is Ops
    other_thread.join()
    assert type(other_thread_result[0]) is Ops


def test_registration_on_first_use_raises_again_after_a_failed_import(
    isolated_registry, monkeypatch
):
    xp = types.ModuleType("lazily_registered_namespace")
    array_ops.register_on_first_use(xp.__name__, "lazily_registered_ops")

    def import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(array_ops, "importlib", types.SimpleNamespace(import_module=import_module))

    for _ in range(2):
        with pytest.raises(ImportError, match="lazily_registered_ops"):
            array_ops.for_namespace(xp)


@pytest.mark.requires_torch
def test_torch_ops_registered_on_first_tensor():
    code = (
        "import sys, gt4py.next, torch\n"
        "from gt4py.next.embedded import array_ops\n"
        "assert 'gt4py.next.embedded.array_libraries.torch_ops' not in sys.modules\n"
        "assert type(array_ops.of(torch.empty(1))).__name__ == 'TorchOps'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.requires_torch
def test_torch_not_imported_with_gt4py():
    subprocess.run(
        [sys.executable, "-c", "import sys, gt4py.next\nassert 'torch' not in sys.modules"],
        check=True,
    )
