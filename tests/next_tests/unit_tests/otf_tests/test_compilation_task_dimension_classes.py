# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Which user dimension classes does a shipped compilation task still reference?

The `ProcessRunner` compiles in a spawn worker, which cannot import classes declared in an
interactive `__main__` (a notebook). Below the frontend, dimensions are `DimensionRef` values,
so the compile task should not need the classes at all. These tests pickle exactly what
`ProcessRunner.submit` ships -- `(task.executor, task.construct_compilable(True))` -- and record
every dimension class referenced from an interactive-like `__main__`.
"""

import io
import pickle
import sys
import types
import warnings

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.otf import arguments, compilation_tasks, runners


def _notebook_dim(name: str, base: type, **kwargs) -> gtx.Dimension:
    # As if declared in a notebook cell: module `__main__`, which a spawn worker cannot import.
    return common.DimensionMeta(
        name, (base,), {"__module__": "__main__", "__qualname__": name, "__slots__": ()}, **kwargs
    )


NbIDim = _notebook_dim("NbIDim", gtx.CartesianAxisIndex)
NbKDim = _notebook_dim("NbKDim", gtx.CartesianAxisIndex, kind=gtx.DimensionKind.VERTICAL)
NbVertex = _notebook_dim("NbVertex", gtx.DimensionIndex)
NbEdge = _notebook_dim("NbEdge", gtx.DimensionIndex)
NbV2EDim = _notebook_dim("NbV2EDim", gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL)
NbKHalfDim = common.Staggered[NbKDim]

NbV2E = gtx.FieldOffset("NbV2E", source=NbEdge, target=(NbVertex, NbV2EDim))


@gtx.field_operator
def _cartesian_shift(
    a: gtx.Field[gtx.Dims[NbIDim, NbKDim], gtx.float64],
) -> gtx.Field[gtx.Dims[NbIDim, NbKDim], gtx.float64]:
    return a(NbIDim + 1) + a(NbKHalfDim + 0.5)(NbKDim - 0.5)


@gtx.program
def cartesian_program(
    a: gtx.Field[gtx.Dims[NbIDim, NbKDim], gtx.float64],
    out: gtx.Field[gtx.Dims[NbIDim, NbKDim], gtx.float64],
):
    _cartesian_shift(a, out=out, domain={NbIDim: (0, 4), NbKDim: (0, 3)})


@gtx.field_operator
def _neighbor_shift(
    e: gtx.Field[gtx.Dims[NbEdge], gtx.float64],
) -> gtx.Field[gtx.Dims[NbVertex], gtx.float64]:
    return e(NbV2E[0])


@gtx.program
def unstructured_program(
    e: gtx.Field[gtx.Dims[NbEdge], gtx.float64],
    out: gtx.Field[gtx.Dims[NbVertex], gtx.float64],
):
    _neighbor_shift(e, out=out)


@pytest.fixture
def interactive_main(monkeypatch):
    main = types.ModuleType("__main__")  # no `__file__`: a notebook / REPL `__main__`
    for dim in (NbIDim, NbKDim, NbVertex, NbEdge, NbV2EDim):
        setattr(main, dim.__qualname__, dim)
    monkeypatch.setitem(sys.modules, "__main__", main)
    return main


def _main_dimensions_referenced(obj: object) -> list[str]:
    found: list[str] = []

    class _Scanner(pickle.Pickler):
        def reducer_override(self, o: object):
            if isinstance(o, common.DimensionMeta) and o.__module__ == "__main__":
                found.append(o.__qualname__)
            return NotImplemented

    _Scanner(io.BytesIO()).dump(obj)
    return found


def _shipped(program, *args, offset_provider):
    task = compilation_tasks.make_compilation_task(
        gtx.gtfn_cpu,
        program.definition_stage,
        arguments.CompileTimeArgs.from_concrete(*args, offset_provider=offset_provider),
    )
    return task.executor, task.construct_compilable(True)


def test_cartesian_task_references_no_dimension_class(interactive_main):
    a = gtx.zeros({NbIDim: 4, NbKDim: 3})
    shipped = _shipped(cartesian_program, a, a, offset_provider={})

    assert _main_dimensions_referenced(shipped) == []
    # ...which is exactly the condition under which `ProcessRunner` falls back to compiling in
    # the calling thread
    assert runners._interactive_main_reference(shipped) is None


def test_unstructured_task_still_references_connectivity_dimensions(interactive_main):
    """
    The runtime offset provider is still shipped (`CompileTimeArgs.offset_provider`, needed by
    the worker-side ITIR passes for domain sizes and `prune_empty_concat_where`), and a
    connectivity's `domain` and `codomain` are dimension classes.
    """
    e = gtx.zeros({NbEdge: 3})
    v = gtx.zeros({NbVertex: 2})
    v2e = gtx.as_connectivity(
        domain={NbVertex: 2, NbV2EDim: 1}, codomain=NbEdge, data=np.array([[0], [2]], np.int32)
    )
    executor, compilable = _shipped(unstructured_program, e, v, offset_provider={"NbV2E": v2e})

    assert _main_dimensions_referenced(compilable.data) == []  # the IR is self-contained
    assert _main_dimensions_referenced(compilable.args.args) == []  # so are the arg types
    assert sorted(set(_main_dimensions_referenced(compilable.args.offset_provider))) == [
        "NbEdge",
        "NbV2EDim",
        "NbVertex",
    ]


def test_cartesian_task_compiles_in_a_spawn_worker(interactive_main, tmp_path):
    """The real thing: a spawn worker, which cannot import `__main__`, unpickles and compiles."""
    a = gtx.zeros({NbIDim: 4, NbKDim: 3})
    task = compilation_tasks.make_compilation_task(
        gtx.gtfn_cpu,
        cartesian_program.definition_stage,
        arguments.CompileTimeArgs.from_concrete(a, a, offset_provider={}),
    )
    runner = runners.ProcessRunner(max_workers=1, shared_session_cache_dir=str(tmp_path))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*in the calling thread.*")
            future = runner.submit(task)  # no fallback to the calling thread
        artifact = future.result(timeout=600)
    finally:
        runner.shutdown(wait=True)
    assert artifact is not None
    assert callable(gtx.gtfn_cpu.load_artifact(artifact))
