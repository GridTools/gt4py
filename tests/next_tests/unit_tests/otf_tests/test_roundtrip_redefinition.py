# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
`resolve` must answer with the *current* class after a notebook-style redefinition.

It used to be memoized, so the roundtrip backend (which resolves tags back to classes to run
embedded) kept the previous `KDim` after the defining module was re-executed, and the second
run failed with a `KeyError` on the column axis.
"""

import importlib
import sys
import textwrap

import numpy as np

import gt4py.next as gtx
from gt4py.next.program_processors.runners import roundtrip


_SOURCE = textwrap.dedent(
    """
    import gt4py.next as gtx


    class KDim(gtx.CartesianAxisIndex, kind=gtx.DimensionKind.VERTICAL): ...


    @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
    def cumsum(state: float, x: float) -> float:
        return state + x


    @gtx.program
    def prog(x: gtx.Field[gtx.Dims[KDim], float], out: gtx.Field[gtx.Dims[KDim], float]):
        cumsum(x, out=out)
    """
)


def _run(kdim, program) -> np.ndarray:
    x = gtx.as_field([kdim], np.ones(4))
    out = gtx.as_field([kdim], np.zeros(4))
    program(x, out, offset_provider={})
    return out.asnumpy()


def test_rerun_module_with_roundtrip(tmp_path, monkeypatch):
    (tmp_path / "nbmod_spike_scan.py").write_text(_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "nbmod_spike_scan", raising=False)
    module = importlib.import_module("nbmod_spike_scan")
    first_kdim, first_prog = module.KDim, module.prog.with_backend(roundtrip.default)
    assert np.allclose(_run(first_kdim, first_prog), [1, 2, 3, 4])

    module = importlib.reload(module)  # the notebook re-run: a new `KDim` class, same ref
    assert module.KDim is not first_kdim
    second_prog = module.prog.with_backend(roundtrip.default)
    assert np.allclose(_run(module.KDim, second_prog), [1, 2, 3, 4])
    # the program loaded in the first run is still alive and still works on its own classes:
    # loading the second one must not have re-bound the first one's (cached) module
    assert np.allclose(_run(first_kdim, first_prog), [1, 2, 3, 4])


def test_program_first_loaded_after_the_rerun(tmp_path, monkeypatch):
    """The other order: the first program is compiled and loaded only after the re-run."""
    (tmp_path / "nbmod_spike_scan2.py").write_text(_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "nbmod_spike_scan2", raising=False)
    module = importlib.import_module("nbmod_spike_scan2")
    first_kdim, first_prog = module.KDim, module.prog.with_backend(roundtrip.default)

    module = importlib.reload(module)
    second_prog = module.prog.with_backend(roundtrip.default)
    assert np.allclose(_run(module.KDim, second_prog), [1, 2, 3, 4])
    # loaded now, when its tags import the *new* classes: it must bind its own
    assert np.allclose(_run(first_kdim, first_prog), [1, 2, 3, 4])
