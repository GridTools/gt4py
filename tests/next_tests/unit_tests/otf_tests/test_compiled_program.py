# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import collections
import contextvars
import gc
import weakref

import pytest

from gt4py import eve, next as gtx
from gt4py.next import utils
from gt4py.next import errors, backend, broadcast, common
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import toolchain, arguments, compiled_program
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners import gtfn


def test_static_arg_from_enum():
    class SomeEnum(eve.IntEnum):
        FOO = 1

    static_arg = arguments.StaticArg(value=SomeEnum.FOO)
    assert static_arg.value == 1 and type(static_arg.value) is int


def test_static_arg_from_enum_tuple():
    class SomeEnum(eve.IntEnum):
        FOO = 1

    static_arg = arguments.StaticArg(value=(SomeEnum.FOO, SomeEnum.FOO))
    assert static_arg.value == (1, 1) and all(type(val) is int for val in static_arg.value)


def test_static_args_non_scalar_type():
    with pytest.raises(
        errors.DSLTypeError,
        match="only scalars.*can be static",
    ):
        static_arg = arguments.StaticArg(value=1)
        static_arg.validate(
            "foo", ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
        )


def test_sanitize_static_args_wrong_type():
    with pytest.raises(
        errors.DSLTypeError,
        match="expected 'int32'.*has.*'int64'",
    ):
        static_arg = arguments.StaticArg(value=gtx.int64(1))
        static_arg.validate("foo", ts.ScalarType(kind=ts.ScalarKind.INT32))


TDim = gtx.Dimension("TDim")


@pytest.fixture
def testee_prog():
    @gtx.field_operator
    def fop(cond: bool):
        return broadcast(cond, (TDim,))

    @gtx.program(backend=gtfn.run_gtfn)
    def prog(
        cond: bool,
        out: gtx.Field[gtx.Dims[TDim], bool],
    ):
        fop(cond, out=out)

    return prog


def _verify_program_has_expected_true_value(program: itir.Program):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    assert isinstance(program.body[0].expr.args[0], itir.Literal)
    assert program.body[0].expr.args[0].value  # is True


def test_inlining_of_scalars_works(testee_prog):
    input_pair = toolchain.ConcreteArtifact(
        data=testee_prog.definition_stage,
        args=arguments.CompileTimeArgs(
            args=list(testee_prog.past_stage.past_node.type.definition.pos_or_kw_args.values()),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={
                arguments.StaticArg: {"cond": arguments.StaticArg(value=True)}
            },
        ),
    )

    transformed = backend.DEFAULT_TRANSFORMS(input_pair).data
    _verify_program_has_expected_true_value(transformed)


def test_inlining_of_scalar_works_integration(testee_prog):
    """
    Test that `.compile` replaces the scalar arg in the program.
    Unlike the previous test, this test uses a full backend and makes sure the replacement step is there.
    """
    # Note: this is more an integration test, but does not execute a program, like the other integration tests.

    hijacked_program = None

    def pirate(program: toolchain.ConcreteArtifact):
        # Replaces the gtfn otf_workflow: and steals the compilable program,
        # then returns a dummy "CompiledProgram" that does nothing.
        nonlocal hijacked_program
        hijacked_program = program
        return lambda *args, **kwargs: None

    hacked_gtfn_backend = gtfn.GTFNBackendFactory(name_postfix="_custom", executor=pirate)

    testee = testee_prog.with_backend(hacked_gtfn_backend).compile(cond=[True], offset_provider={})
    testee(
        cond=True,
        out=gtx.zeros(domain={TDim: 1}, dtype=bool),
        offset_provider={},
    )

    _verify_program_has_expected_true_value(hijacked_program.data)


def test_different_static_args_work_after_backend_change(testee_prog):
    prg1 = testee_prog.with_backend(gtfn.run_gtfn)
    prg2 = testee_prog.with_backend(gtfn.run_gtfn)

    # compile with static args
    prg1.compile(cond=[True], offset_provider={})

    # compile without static args
    prg2.compile(offset_provider={})


def test_different_static_args_work_after_static_params_change(testee_prog):
    testee_prog2 = testee_prog.with_compilation_options(static_params=["cond"])

    # compile without static args
    testee_prog.compile(offset_provider={})

    # compile with static args
    testee_prog2.compile(cond=[True], offset_provider={})


def test_different_static_args_break_same_prg_after_static_params_change(testee_prog):
    prg = testee_prog.with_compilation_options(static_params=[])

    # compile without static args
    prg.compile(offset_provider={})

    # compile with different static args
    with pytest.raises(
        ValueError,
        match="Argument descriptor StaticArg must be the same for all compiled programs",
    ):
        prg.compile(cond=[True], offset_provider={})


def _verify_program_has_expected_domain(
    program: itir.Program, expected_domain: gtx.Domain, uids: utils.IDGeneratorPool
):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    domain = CollapseTuple.apply(program.body[0].domain, within_stencil=False, uids=uids)
    assert domain == im.domain(common.GridType.CARTESIAN, expected_domain)


def test_inlining_of_static_domain_works(testee_prog, uids: utils.IDGeneratorPool):
    domain = gtx.Domain(dims=(TDim,), ranges=(gtx.UnitRange(0, 1),))
    input_pair = toolchain.ConcreteArtifact(
        data=testee_prog.definition_stage,
        args=arguments.CompileTimeArgs(
            args=list(testee_prog.past_stage.past_node.type.definition.pos_or_kw_args.values()),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={
                arguments.FieldDomainDescriptor: {"out": arguments.FieldDomainDescriptor(domain)}
            },
        ),
    )

    transformed = backend.DEFAULT_TRANSFORMS(input_pair).data
    _verify_program_has_expected_domain(transformed, domain, uids)


def test_make_param_context_from_func_type_for_named_collections():
    int32_t, int64_t = (
        ts.ScalarType(kind=ts.ScalarKind.INT32),
        ts.ScalarType(kind=ts.ScalarKind.INT64),
    )

    @dataclasses.dataclass
    class DataclassNamedCollection:
        u: gtx.Field[gtx.Dims[TDim], int32_t]
        v: gtx.Field[gtx.Dims[TDim], int64_t]

    nc_type = ts.NamedCollectionType(
        types=[int32_t, int64_t], keys=["a", "b"], original_python_type="DUMMY"
    )
    func_type = ts.FunctionType(
        pos_only_args=[],
        pos_or_kw_args={"inp": nc_type},
        kw_only_args={},
        returns=nc_type,
    )
    context = compiled_program._make_param_context_from_func_type(func_type)
    # both `extract` and `_make_param_context_from_func_type` need to use the same structure
    assert arguments.extract(DataclassNamedCollection(int32_t, int64_t)) == context["inp"]


class _DummyPool:
    def __init__(self, root):
        self.root = root


def test_metrics_source_key_caches_per_pool_and_key():
    cache_token = compiled_program._metrics_source_key_cache.set({})
    counter_token = compiled_program._pools_per_root.set(
        collections.Counter({("prog", "backend"): 3})
    )
    try:
        pool = _DummyPool(("prog", "backend"))
        key = (("static",), 11, None)

        first = compiled_program.metrics_source_key(pool, key)
        # Change counter after first call; second call must come from cache.
        compiled_program._pools_per_root.get()[pool.root] = 99
        second = compiled_program.metrics_source_key(pool, key)

        assert first == second
        assert first == f"prog<backend>#3[{hash(key)}]"
        assert compiled_program._metrics_source_key_cache.get()[(id(pool), key)] == first
    finally:
        compiled_program._metrics_source_key_cache.reset(cache_token)
        compiled_program._pools_per_root.reset(counter_token)


def test_metrics_source_key_uses_contextvars_isolation():
    pool = _DummyPool(("prog", "backend"))
    key = (("static",), 12, None)

    base_cache_token = compiled_program._metrics_source_key_cache.set({})
    base_counter_token = compiled_program._pools_per_root.set(collections.Counter({pool.root: 1}))
    try:
        key_in_base = compiled_program.metrics_source_key(pool, key)

        def _run_in_new_context():
            compiled_program._metrics_source_key_cache.set({})
            compiled_program._pools_per_root.set(collections.Counter({pool.root: 7}))
            return compiled_program.metrics_source_key(pool, key)

        key_in_other_ctx = contextvars.Context().run(_run_in_new_context)

        assert key_in_base != key_in_other_ctx
        assert key_in_base == f"prog<backend>#1[{hash(key)}]"
        assert key_in_other_ctx == f"prog<backend>#7[{hash(key)}]"
        # Base context cache remains its own value.
        assert compiled_program._metrics_source_key_cache.get()[(id(pool), key)] == key_in_base
    finally:
        compiled_program._metrics_source_key_cache.reset(base_cache_token)
        compiled_program._pools_per_root.reset(base_counter_token)


def test_metrics_source_key_finalizer_removes_cache_entry_when_pool_is_deleted():
    cache_token = compiled_program._metrics_source_key_cache.set({})
    counter_token = compiled_program._pools_per_root.set(
        collections.Counter({("prog", "backend"): 2})
    )
    try:
        pool = _DummyPool(("prog", "backend"))
        key = (("static",), 13, None)

        compiled_program.metrics_source_key(pool, key)
        entry = (id(pool), key)
        assert entry in compiled_program._metrics_source_key_cache.get()

        pool_ref = weakref.ref(pool)
        del pool
        gc.collect()

        assert pool_ref() is None
        assert entry not in compiled_program._metrics_source_key_cache.get()
    finally:
        compiled_program._metrics_source_key_cache.reset(cache_token)
        compiled_program._pools_per_root.reset(counter_token)
