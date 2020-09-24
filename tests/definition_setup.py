import itertools
from functools import partial
from typing import Iterator, List, Tuple

import pytest

from gt4py.analysis import TransformData
from gt4py.definitions import BuildOptions
from gt4py.ir.nodes import (
    ArgumentInfo,
    Assign,
    Axis,
    AxisBound,
    AxisInterval,
    BlockStmt,
    ComputationBlock,
    DataType,
    Domain,
    FieldDecl,
    FieldRef,
    IterationOrder,
    LevelMarker,
    Location,
    StencilDefinition,
    StencilImplementation,
)


BodyType = List[Tuple[str, str, Tuple[int, int, int]]]


@pytest.fixture()
def ijk_domain() -> Domain:
    axes = [Axis(name=idx) for idx in ["I", "J", "K"]]
    return Domain(parallel_axes=axes[:2], sequential_axis=axes[2])


@pytest.fixture(params=[IterationOrder.FORWARD, IterationOrder.BACKWARD, IterationOrder.PARALLEL])
def iteration_order(request) -> Iterator[IterationOrder]:
    return request.param


@pytest.fixture(params=[IterationOrder.FORWARD, IterationOrder.BACKWARD])
def non_parallel_iteration_order(request) -> Iterator[IterationOrder]:
    return request.param


@pytest.fixture(params=[(-1, 0, 0), (0, 1, 0), (-1, 2, 0)])
def ij_offset(request):
    yield request.param


def make_offset(offset: Tuple[int, int, int]):
    return {"I": offset[0], "J": offset[1], "K": offset[2]}


def make_assign(
    target: str,
    value: str,
    offset: Tuple[int, int, int] = (0, 0, 0),
    loc_line=0,
    loc_scope="unnamed",
):
    make_loc = partial(Location, scope=loc_scope)
    return Assign(
        target=FieldRef(
            name=target, offset=make_offset((0, 0, 0)), loc=make_loc(line=loc_line, column=0)
        ),
        value=FieldRef(
            name=value, offset=make_offset(offset), loc=make_loc(line=loc_line, column=2)
        ),
        loc=make_loc(line=loc_line, column=1),
    )


def make_definition_multiple(
    name: str,
    fields: List[str],
    domain: Domain,
    info: List[Tuple[BodyType, IterationOrder, Tuple[int, int]]],
) -> StencilDefinition:
    api_signature = [ArgumentInfo(name=n, is_keyword=False) for n in fields]
    bodies = tuple(itertools.chain.from_iterable([definition[0] for definition in info]))
    tmp_fields = {i[0] for i in bodies}.union({i[1] for i in bodies}).difference(fields)
    api_fields = [
        FieldDecl(name=n, data_type=DataType.AUTO, axes=domain.axes_names, is_api=True)
        for n in fields
    ] + [
        FieldDecl(name=n, data_type=DataType.AUTO, axes=domain.axes_names, is_api=False)
        for n in tmp_fields
    ]
    comp_blocks = [
        ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START, offset=interval[0]),
                end=AxisBound(level=LevelMarker.END, offset=interval[1]),
            ),
            iteration_order=iteration_order,
            body=BlockStmt(
                stmts=[
                    make_assign(*assign, loc_scope=name, loc_line=i)
                    for i, assign in enumerate(body)
                ]
            ),
        )
        for body, iteration_order, interval in info
    ]
    return StencilDefinition(
        name=name,
        domain=domain,
        api_signature=api_signature,
        api_fields=api_fields,
        parameters=[],
        computations=comp_blocks,
        docstring="",
    )


def make_definition(
    name: str, fields: List[str], domain: Domain, body: BodyType, iteration_order: IterationOrder
) -> StencilDefinition:
    return make_definition_multiple(
        name,
        fields,
        domain,
        [
            (body, iteration_order, (0, 0)),
        ],
    )


def init_implementation_from_definition(definition: StencilDefinition) -> StencilImplementation:
    return StencilImplementation(
        name=definition.name,
        api_signature=[],
        domain=definition.domain,
        fields={},
        parameters={},
        multi_stages=[],
        fields_extents={},
        unreferenced=[],
        axis_splitters_var=None,
        externals=definition.externals,
        sources=definition.sources,
        docstring=definition.docstring,
    )


def make_transform_data(
    *,
    name: str,
    domain: Domain,
    fields: List[str],
    body: BodyType,
    iteration_order: IterationOrder,
) -> TransformData:
    definition = make_definition(name, fields, domain, body, iteration_order)
    return TransformData(
        definition_ir=definition,
        implementation_ir=init_implementation_from_definition(definition),
        options=BuildOptions(name=name, module=__name__),
    )


def make_transform_data_multiple(
    *,
    name: str,
    domain: Domain,
    fields: List[str],
    info: List[Tuple[BodyType, IterationOrder, Tuple[int, int]]],
) -> TransformData:
    definition = make_definition_multiple(name, fields, domain, info)
    return TransformData(
        definition_ir=definition,
        implementation_ir=init_implementation_from_definition(definition),
        options=BuildOptions(name=name, module=__name__),
    )
