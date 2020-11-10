import ast
import re
from typing import Iterator

import pytest

from gt4py.backend.gtc_backend.stencil_object_snippet_generators import (
    ComputationCallGenerator,
    DomainInfoGenerator,
    FieldInfoGenerator,
    ParameterInfoGenerator,
    RunBodyGenerator,
)
from gt4py.gtc.common import DataType, LoopOrder
from gt4py.gtc.gtir import (
    AccessKind,
    AssignStmt,
    AxisBound,
    CartesianOffset,
    Computation,
    FieldAccess,
    FieldBoundary,
    FieldDecl,
    FieldMetadata,
    FieldsMetadata,
    HorizontalLoop,
    Stencil,
    VerticalInterval,
    VerticalLoop,
)


def ast_parse(source):
    print(source)
    return ast.parse(source)


@pytest.fixture
def copy_stencil() -> Iterator[Computation]:
    yield Computation(
        name="copy_shift",
        params=[
            FieldDecl(name="a", dtype=DataType.AUTO),
            FieldDecl(name="b", dtype=DataType.AUTO),
        ],
        fields_metadata=FieldsMetadata(
            metas={
                "a": FieldMetadata(
                    name="a",
                    access=AccessKind.READ_WRITE,
                    boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
                    dtype=DataType.AUTO,
                ),
                "b": FieldMetadata(
                    name="b",
                    access=AccessKind.READ_ONLY,
                    boundary=FieldBoundary(i=(1, 0), j=(0, 0), k=(0, 0)),
                    dtype=DataType.AUTO,
                ),
            }
        ),
        stencils=[
            Stencil(
                vertical_loops=[
                    VerticalLoop(
                        loop_order=LoopOrder.PARALLEL,
                        vertical_intervals=[
                            VerticalInterval(
                                start=AxisBound.start(),
                                end=AxisBound.end(),
                                horizontal_loops=[
                                    HorizontalLoop(
                                        stmt=AssignStmt(
                                            left=FieldAccess.centered(name="a"),
                                            right=FieldAccess(
                                                name="b", offset=CartesianOffset(i=-1, j=0, k=0)
                                            ),
                                        )
                                    )
                                ],
                            )
                        ],
                    )
                ]
            )
        ],
    )


def test_field_info_field_boundary() -> None:
    field_boundary = FieldBoundary(i=(0, 1), j=(2, 3), k=(4, 5))
    assert FieldInfoGenerator.apply(field_boundary) == "Boundary((0, 1), (2, 3), (4, 5))"


def test_field_info_field_metadata() -> None:
    field_metadata = FieldMetadata(
        name="a",
        access=AccessKind.READ_ONLY,
        boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
        dtype=DataType.FLOAT64,
    )
    assert FieldInfoGenerator.apply(field_metadata) == (
        "'a': FieldInfo(access=AccessKind.READ_ONLY, "
        "boundary=Boundary((0, 0), (0, 0), (0, 0)), "
        "dtype=dtype('float64'))"
    )


def test_field_info_fields_metadata() -> None:
    fields_metadata = FieldsMetadata(
        metas={
            "a": FieldMetadata(
                name="a",
                access=AccessKind.READ_ONLY,
                boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
                dtype=DataType.FLOAT64,
            ),
            "b": FieldMetadata(
                name="b",
                access=AccessKind.READ_WRITE,
                boundary=FieldBoundary(i=(1, 2), j=(3, 4), k=(5, 6)),
                dtype=DataType.FLOAT64,
            ),
        }
    )
    assert FieldInfoGenerator.apply(fields_metadata) == (
        "{'a': FieldInfo(access=AccessKind.READ_ONLY, "
        "boundary=Boundary((0, 0), (0, 0), (0, 0)), "
        "dtype=dtype('float64')), "
        "'b': FieldInfo(access=AccessKind.READ_WRITE, "
        "boundary=Boundary((1, 2), (3, 4), (5, 6)), "
        "dtype=dtype('float64'))}"
    )


def test_field_info(copy_stencil: Computation) -> None:
    source = FieldInfoGenerator.apply(copy_stencil)
    info_pattern = re.compile(
        (
            r"'(\w.*?)': FieldInfo\(access=AccessKind\.([A-Z_]*?), "
            r"boundary=Boundary\((.*?)\), "
            r"dtype=dtype\((.*?)\)\)"
        )
    )
    dict_pattern = re.compile("^{.*}$")
    assert re.match(dict_pattern, source)
    infos = re.findall(info_pattern, source)
    assert len(infos) == 2
    assert infos[0][0] == "a"
    assert infos[0][1] == "READ_WRITE"
    assert infos[0][2] == "(0, 0), (0, 0), (0, 0)"
    assert infos[1][0] == "b"
    assert infos[1][1] == "READ_ONLY"
    assert infos[1][2] == "(1, 0), (0, 0), (0, 0)"


def test_parameter_info(copy_stencil: Computation) -> None:
    source = ParameterInfoGenerator.apply(copy_stencil)
    assert source == "{}"


def test_computation_call(copy_stencil: Computation) -> None:
    source = ComputationCallGenerator.apply(copy_stencil)
    tree = ast_parse(source)
    call = tree.body[0].value
    assert isinstance(call, ast.Call)
    assert call.func.attr == "run"
    assert [name.id for name in call.args] == ["a", "b", "_domain_"]


def test_run_body(copy_stencil: Computation) -> None:
    source = RunBodyGenerator.apply(copy_stencil)
    tree = ast_parse(source)
    assert isinstance(tree.body[0], ast.Assign)
    assert tree.body[0].targets[0].id == "a_at"
    assert tree.body[0].value.func.id == "_Accessor"
    assert tree.body[0].value.args[0].id == "a"
    assert isinstance(tree.body[1], ast.Assign)
    assert tree.body[1].targets[0].id == "b_at"
    assert tree.body[1].value.func.id == "_Accessor"
    assert tree.body[1].value.args[0].id == "b"


def test_domain_info(copy_stencil: Computation) -> None:
    assert (
        DomainInfoGenerator.apply(copy_stencil)
        == "DomainInfo(parallel_axes=('I', 'J'), sequential_axis='K', ndims=3)"
    )
