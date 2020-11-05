import ast
import pathlib
from typing import Any, Iterator

import astor
import black
import numpy
import pytest

from gt4py.utils import make_module_from_file
from gtc import common, gtir, pnir
from gtc.pnir_to_ast import PnirToAst


def to_snippet(node: ast.AST) -> str:
    return astor.to_source(node).strip()


def to_literal(node: ast.AST) -> Any:
    return eval(astor.to_source(node).strip("\n()"))


def eval_const(node: ast.AST) -> Any:
    return eval(
        compile(
            ast.fix_missing_locations(ast.Interactive(body=[ast.Expr(value=node)])),
            filename="<none>",
            mode="single",
        )
    )


@pytest.fixture
def pnir_to_ast() -> Iterator[PnirToAst]:
    yield PnirToAst()


def test_literal(pnir_to_ast: PnirToAst) -> None:
    false = pnir_to_ast.visit(gtir.Literal(value="False", dtype=common.DataType.BOOL))
    assert isinstance(false, ast.NameConstant)
    assert false.value is False
    inum = pnir_to_ast.visit(gtir.Literal(value="42", dtype=common.DataType.INT8))
    assert isinstance(inum, ast.Num)
    assert inum.n == 42
    fnum = pnir_to_ast.visit(gtir.Literal(value="1.7654", dtype=common.DataType.FLOAT64))
    assert isinstance(fnum, ast.Num)
    assert fnum.n == 1.7654


def test_offset(pnir_to_ast: PnirToAst) -> None:
    indices = pnir_to_ast.visit(gtir.CartesianOffset(i=-29, j=4, k=0))
    assert isinstance(indices[0], ast.BinOp)
    assert indices[0].left.id == "I"
    assert isinstance(indices[0].op, ast.Sub)
    assert indices[0].right.n == 29
    assert isinstance(indices[1].op, ast.Add)
    assert indices[2].id == "K"


def test_field_access(pnir_to_ast: PnirToAst) -> None:
    subs = pnir_to_ast.visit(
        gtir.FieldAccess(name="a", offset=gtir.CartesianOffset(i=0, j=-1, k=1))
    )
    assert isinstance(subs, ast.Subscript)
    assert subs.value.id == "a"
    assert subs.slice.value.elts[0].id == "I"


def test_binary_op(pnir_to_ast: PnirToAst) -> None:
    bin_op = pnir_to_ast.visit(
        gtir.BinaryOp(
            left=gtir.FieldAccess.centered(name="a"),
            right=gtir.FieldAccess.centered(name="b"),
            op=common.BinaryOperator.DIV,
        )
    )
    assert isinstance(bin_op, ast.BinOp)
    assert isinstance(bin_op.op, ast.Div)
    assert bin_op.left.value.id == "a"
    assert bin_op.right.value.id == "b"


def test_assign_stmt(pnir_to_ast: PnirToAst) -> None:
    assign = pnir_to_ast.visit(
        gtir.AssignStmt(
            left=gtir.FieldAccess.centered(name="a"), right=gtir.FieldAccess.centered(name="b")
        )
    )
    assert isinstance(assign, ast.Assign)
    assert len(assign.targets) == 1
    assert assign.targets[0].value.id == "a"
    assert assign.value.value.id == "b"


def test_ij_loop(pnir_to_ast: PnirToAst) -> None:
    ij_loop = pnir.IJLoop(
        body=[
            gtir.AssignStmt(
                left=gtir.FieldAccess.centered(name="a"),
                right=gtir.BinaryOp(
                    left=gtir.FieldAccess(name="b", offset=gtir.CartesianOffset(i=1, j=0, k=0)),
                    right=gtir.Literal(value="1", dtype=common.DataType.INT32),
                    op=common.BinaryOperator.ADD,
                ),
            ),
            gtir.AssignStmt(
                left=gtir.FieldAccess.centered(name="b"),
                right=gtir.BinaryOp(
                    left=gtir.FieldAccess(name="a", offset=gtir.CartesianOffset(i=0, j=0, k=-1)),
                    right=gtir.FieldAccess.centered(name="b"),
                    op=common.BinaryOperator.MUL,
                ),
            ),
        ]
    )
    j_for = pnir_to_ast.visit(ij_loop)
    print(astor.dump_tree(j_for))
    print(astor.code_gen.to_source(j_for))
    assert isinstance(j_for, ast.For)
    assert j_for.iter.args[0].value.id == "_domain_"
    assert j_for.iter.args[0].slice.value.n == 1
    assert j_for.target.id == "J"
    i_for = j_for.body[0]
    assert isinstance(i_for, ast.For)
    assert i_for.target.id == "I"
    body_1, body_2 = i_for.body
    assert isinstance(body_1, ast.Assign)
    assert isinstance(body_1.targets[0], ast.Subscript)
    assert isinstance(body_2, ast.Assign)


def test_axis_bound(pnir_to_ast: PnirToAst) -> None:
    lower_0 = pnir_to_ast.visit(gtir.AxisBound.start())
    assert isinstance(lower_0, ast.Num)
    assert lower_0.n == 0
    lower_1 = pnir_to_ast.visit(gtir.AxisBound.from_start(1))
    assert isinstance(lower_1, ast.Num)
    assert to_literal(lower_1) == 1
    upper_0 = pnir_to_ast.visit(gtir.AxisBound.end())
    assert isinstance(upper_0, ast.Subscript)
    assert to_snippet(upper_0.value) == "_domain_"
    assert to_literal(upper_0.slice.value) == 2
    upper_1 = pnir_to_ast.visit(gtir.AxisBound.from_end(1))
    assert isinstance(upper_1, ast.BinOp)
    assert to_snippet(upper_1.left.value) == "_domain_"
    assert to_literal(upper_1.left.slice.value) == 2
    assert to_literal(upper_1.right) == 1
    assert isinstance(upper_1.op, ast.Sub)


def test_k_loop(pnir_to_ast: PnirToAst) -> None:
    k_loop = pnir.KLoop(
        lower=gtir.AxisBound.from_start(1),
        upper=gtir.AxisBound.from_end(3),
        ij_loops=[
            pnir.IJLoop(
                body=[
                    gtir.AssignStmt(
                        left=gtir.FieldAccess.centered(name="a"),
                        right=gtir.FieldAccess.centered(name="b"),
                    )
                ]
            ),
            pnir.IJLoop(
                body=[
                    gtir.AssignStmt(
                        left=gtir.FieldAccess.centered(name="b"),
                        right=gtir.FieldAccess.centered(name="c"),
                    )
                ]
            ),
        ],
    )
    k_for = pnir_to_ast.visit(k_loop)
    print(astor.dump_tree(k_for))
    print(astor.code_gen.to_source(k_for))
    assert isinstance(k_for, ast.For)
    assert to_snippet(k_for.target) == "K"


def test_run_function(pnir_to_ast: PnirToAst) -> None:
    run_function = pnir.RunFunction(
        field_params=["a", "b", "c"],
        scalar_params=["d"],
        k_loops=[
            pnir.KLoop(
                lower=gtir.AxisBound.start(),
                upper=gtir.AxisBound.from_end(3),
                ij_loops=[
                    pnir.IJLoop(
                        body=[
                            gtir.AssignStmt(
                                left=gtir.FieldAccess.centered(name="a"),
                                right=gtir.FieldAccess.centered(name="b"),
                            ),
                        ],
                    ),
                ],
            ),
            pnir.KLoop(
                lower=gtir.AxisBound.from_end(2),
                upper=gtir.AxisBound.end(),
                ij_loops=[
                    pnir.IJLoop(
                        body=[
                            gtir.AssignStmt(
                                left=gtir.FieldAccess.centered(name="b"),
                                right=gtir.FieldAccess.centered(name="c"),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    func_def = pnir_to_ast.visit(run_function)
    assert isinstance(func_def, ast.FunctionDef)
    assert func_def.name == "run"
    assert [arg.arg for arg in func_def.args.args] == ["a", "b", "c"]
    assert [kwarg.arg for kwarg in func_def.args.kwonlyargs] == ["d", "_domain_"]
    assert isinstance(func_def.body[1], ast.For)
    assert len(func_def.body) == 3


def test_module(tmp_path: pathlib.Path, pnir_to_ast: PnirToAst) -> None:
    pnir_module = pnir.Module(
        run=pnir.RunFunction(
            field_params=["a", "b", "c"],
            scalar_params=["d"],
            k_loops=[
                pnir.KLoop(
                    lower=gtir.AxisBound.start(),
                    upper=gtir.AxisBound.from_end(3),
                    ij_loops=[
                        pnir.IJLoop(
                            body=[
                                gtir.AssignStmt(
                                    left=gtir.FieldAccess.centered(name="a"),
                                    right=gtir.FieldAccess.centered(name="b"),
                                ),
                            ],
                        ),
                    ],
                ),
                pnir.KLoop(
                    lower=gtir.AxisBound.from_end(2),
                    upper=gtir.AxisBound.end(),
                    ij_loops=[
                        pnir.IJLoop(
                            body=[
                                gtir.AssignStmt(
                                    left=gtir.FieldAccess.centered(name="a"),
                                    right=gtir.FieldAccess.centered(name="c"),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    )
    ast_module = pnir_to_ast.visit(pnir_module)
    comp_source = black.format_str(astor.to_source(ast_module), mode=black.Mode())
    print(comp_source)
    comp_mod_file = tmp_path / "test_pnir_to_ast_computation.py"
    comp_mod_file.write_text(comp_source)
    computation = make_module_from_file("computation", file_path=comp_mod_file)
    assert computation
    a = numpy.zeros((7, 7, 7))
    b = numpy.full_like(a, 1)
    c = numpy.full_like(a, 2)
    with pytest.raises(TypeError):
        computation.run(a, b, c)
    computation.run(a, b, c, d=1.0)
    assert (a[:, :, :-3] == b[:, :, :-3]).all()
    assert (a[:, :, -3] == numpy.zeros((7, 7, 1))).all()
    assert (a[:, :, -2:] == c[:, :, -2:]).all()


def test_field_metadata(pnir_to_ast: PnirToAst) -> None:
    meta = gtir.FieldMetadata(
        name="a",
        access=gtir.AccessKind.READ_WRITE,
        boundary=gtir.FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
        dtype=common.DataType.FLOAT64,
    )
    field_info = pnir_to_ast.visit(meta)
    assert isinstance(field_info, ast.Call)


def test_fields_metadata(pnir_to_ast: PnirToAst) -> None:
    meta_builder = (
        gtir.FieldMetadataBuilder()
        .access(gtir.AccessKind.READ_WRITE)
        .dtype(common.DataType.FLOAT64)
    )
    metas = gtir.FieldsMetadata(
        metas={"a": meta_builder.name("a").build(), "b": meta_builder.name("b").build()}
    )
    infos = pnir_to_ast.visit(metas)
    assert set(infos.keys()) == {"a", "b"}
    assert isinstance(infos["b"], ast.Call)


@pytest.fixture
def set_ones_stencil() -> Iterator[pnir.Stencil]:
    yield pnir.Stencil(
        computation=pnir.Module(
            run=pnir.RunFunction(
                field_params=["a"],
                scalar_params=[],
                k_loops=[
                    pnir.KLoop(
                        lower=gtir.AxisBound.start(),
                        upper=gtir.AxisBound.end(),
                        ij_loops=[
                            pnir.IJLoop(
                                body=[
                                    gtir.AssignStmt(
                                        left=gtir.FieldAccess.centered(name="a"),
                                        right=gtir.Literal(value="1", dtype=common.DataType.INT32),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ),
        stencil_obj=pnir.StencilObject(
            name="set_ones",
            params=[gtir.FieldDecl(name="a", dtype=common.DataType.INT32)],
            fields_metadata=gtir.FieldsMetadata(
                metas={
                    "a": (
                        gtir.FieldMetadataBuilder()
                        .name("a")
                        .access(gtir.AccessKind.READ_WRITE)
                        .dtype(common.DataType.INT32)
                        .build()
                    ),
                },
            ),
        ),
    )


def test_stencil_onemod(set_ones_stencil: pnir.Stencil, tmp_path: pathlib.Path) -> None:
    pnir_to_ast = PnirToAst(onemodule=True)
    ast_module_builder = pnir_to_ast.visit(set_ones_stencil)
    ast_module = ast_module_builder.build()
    stencil_source = black.format_str(astor.to_source(ast_module), mode=black.Mode())
    print(stencil_source)
    mod_file = tmp_path / "test_pnir_to_ast_stencil_onemod.py"
    mod_file.write_text(stencil_source)
    stencil = make_module_from_file("stencil", file_path=mod_file)
    assert stencil
    a = numpy.zeros((4, 4, 4), dtype=numpy.int32)
    stencil.set_ones()(a, origin=(0, 0, 0))
    assert (a == numpy.ones_like(a)).all()


def test_stencil_split(set_ones_stencil: pnir.Stencil, tmp_path: pathlib.Path) -> None:
    # file paths
    cur_path = tmp_path / "test_pnir_to_ast_split"
    cur_path.mkdir()
    comp_file = cur_path / "computation.py"
    mod_file = cur_path / "stencil.py"
    # generate asts and python code from there
    pnir_to_ast = PnirToAst(onemodule=False)
    ast_module_builder, ast_comp_module = pnir_to_ast.visit(set_ones_stencil)
    ast_module = ast_module_builder.build()
    stencil_source = black.format_str(astor.to_source(ast_module), mode=black.Mode())
    computation_source = black.format_str(astor.to_source(ast_comp_module), mode=black.Mode())
    print(computation_source)
    print("----")
    print(stencil_source)
    # write and use python sources
    comp_file.write_text(computation_source)
    mod_file.write_text(stencil_source)
    stencil = make_module_from_file("stencil", file_path=mod_file)
    assert stencil
    a = numpy.zeros((4, 4, 4), dtype=numpy.int32)
    stencil.set_ones()(a, origin=(0, 0, 0))
    assert (a == numpy.ones_like(a)).all()
