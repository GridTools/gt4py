import ctypes

import pytest

from functional.backend.codegen import gtfn
from functional.iterator import ir
from functional.backend import defs


@pytest.fixture
def fencil_example():
    domain = ir.FunCall(fun=ir.SymRef(id="cartesian_domain"),
                        args=[ir.FunCall(fun=ir.SymRef(id="named_range"),
                                         args=[
                                             ir.AxisLiteral(value="X"),
                                             ir.Literal(value="0", type="int"),
                                             ir.Literal(value="10", type="int")
                                         ])
                              ])
    itir = ir.FencilDefinition(id="example",
                               params=[ir.Sym(id="buf"), ir.Sym(id="sc")],
                               function_definitions=[
                                   ir.FunctionDefinition(id="stencil",
                                                         params=[
                                                             ir.Sym(id="buf"),
                                                             ir.Sym(id="sc")
                                                         ],
                                                         expr=ir.Literal(value="1", type="float"))
                               ],
                               closures=[
                                   ir.StencilClosure(domain=domain,
                                                     stencil=ir.SymRef(id="stencil"),
                                                     output=ir.SymRef(id="buf"),
                                                     inputs=[ir.SymRef(id="buf"), ir.SymRef(id="sc")])
                               ])
    params = [
        defs.BufferParameter("buf", 1, ctypes.c_float),
        defs.ScalarParameter("sc", ctypes.c_float)
    ]
    return itir, params


def test_codegen(fencil_example):
    itir, parameters = fencil_example
    module = gtfn.create_source_module(itir, parameters)
    assert module.entry_point.name == itir.id
    assert any(d.name == "gridtools" for d in module.library_deps)
    assert all(fp.name == ip.id for fp, ip in zip(module.entry_point.parameters, itir.params))
