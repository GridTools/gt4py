from typing import Dict, Iterator, List, Optional

import pytest

from gt4py.analysis import TransformData
from gt4py.analysis.passes import (
    ComputeExtentsPass,
    InitInfoPass,
    MergeBlocksPass,
    NormalizeBlocksPass,
)
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


@pytest.fixture(params=["extended", "offset", "allowed_offset"])
def case(request):
    yield request.param


@pytest.fixture()
def length() -> Iterator[int]:
    yield 10


@pytest.fixture()
def axis_names() -> Iterator[List[str]]:
    yield ["I", "J", "K"]


@pytest.fixture()
def axes(axis_names: List[str]) -> Iterator[List[Axis]]:
    yield [Axis(name=idx) for idx in axis_names]


@pytest.fixture()
def domain(axes: List[Axis]) -> Iterator[Domain]:
    yield Domain(parallel_axes=axes[:2], sequential_axis=axes[2])


@pytest.fixture()
def param_names() -> Iterator[List[str]]:
    yield ["in", "out", "inout"]


@pytest.fixture()
def api_signature(param_names: List[str]) -> Iterator[List[ArgumentInfo]]:
    yield [ArgumentInfo(name=n, is_keyword=False) for n in param_names]


@pytest.fixture()
def api_fields(param_names: List[str], axis_names: List[str]) -> Iterator[List[FieldDecl]]:
    yield (
        [
            FieldDecl(name=n, data_type=DataType.AUTO, axes=axis_names, is_api=True)
            for n in param_names
        ]
        + [FieldDecl(name="tmp", data_type=DataType.AUTO, axes=axis_names, is_api=False)]
    )


@pytest.fixture()
def no_offset(axis_names: List[str]) -> Iterator[Dict[str, int]]:
    yield {idx: 0 for idx in axis_names}


@pytest.fixture()
def offset_by_one(axis_names: List[str]) -> Iterator[Dict[str, int]]:
    yield {idx: -1 if idx == "I" else 0 for idx in axis_names}


@pytest.fixture()
def computations_extended(
    no_offset: Dict[str, int], offset_by_one: Dict[str, int]
) -> Iterator[List[ComputationBlock]]:
    """
    Build stencil definition body for write after read with extended compute domain.

    equivalent to
    .. code-block: python

        with computation(PARALLEL), interval(...):
            tmp = inout
            out = tmp[-1, 0, 0]
            inout = in
    """
    yield [
        ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START), end=AxisBound(level=LevelMarker.END)
            ),
            iteration_order=IterationOrder.PARALLEL,
            body=BlockStmt(
                stmts=[
                    Assign(
                        target=FieldRef(
                            name="tmp",
                            offset=no_offset,
                            loc=Location(scope="war_extended_compute", line=0, column=0),
                        ),
                        value=FieldRef(
                            name="inout",
                            offset=no_offset,
                            loc=Location(scope="war_extended_compute", line=0, column=2),
                        ),
                        loc=Location(scope="war_extended_compute", line=0, column=1),
                    ),
                    Assign(
                        target=FieldRef(
                            name="out",
                            offset=no_offset,
                            loc=Location(scope="war_extended_compute", line=1, column=0),
                        ),
                        value=FieldRef(
                            name="tmp",
                            offset=offset_by_one,
                            loc=Location(scope="war_extended_compute", line=1, column=2),
                        ),
                        loc=Location(scope="war_extended_compute", line=1, column=1),
                    ),
                    Assign(
                        target=FieldRef(
                            name="inout",
                            offset=no_offset,
                            loc=Location(scope="war_extended_compute", line=2, column=0),
                        ),
                        value=FieldRef(
                            name="in",
                            offset=no_offset,
                            loc=Location(scope="war_extended_compute", line=2, column=2),
                        ),
                        loc=Location(scope="war_extended_compute", line=2, column=1),
                    ),
                ]
            ),
        )
    ]


@pytest.fixture()
def computations_offset(
    no_offset: Dict[str, int], offset_by_one: Dict[str, int]
) -> Iterator[List[ComputationBlock]]:
    """
    Build stencil definition body for write after read with read offset.

    equivalent to
    .. code-block: python

        with computation(PARALLEL), interval(...):
            out = inout[-1, 0, 0]
            inout = in
    """
    yield [
        ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START), end=AxisBound(level=LevelMarker.END)
            ),
            iteration_order=IterationOrder.PARALLEL,
            body=BlockStmt(
                stmts=[
                    Assign(
                        target=FieldRef(
                            name="out",
                            offset=no_offset,
                            loc=Location(scope="war_offset", line=0, column=0),
                        ),
                        value=FieldRef(
                            name="inout",
                            offset=offset_by_one,
                            loc=Location(scope="war_offset", line=0, column=2),
                        ),
                        loc=Location(scope="war_offset", line=0, column=1),
                    ),
                    Assign(
                        target=FieldRef(
                            name="inout",
                            offset=no_offset,
                            loc=Location(scope="war_offset", line=1, column=0),
                        ),
                        value=FieldRef(
                            name="in",
                            offset=offset_by_one,
                            loc=Location(scope="war_offset", line=1, column=2),
                        ),
                        loc=Location(scope="war_offset", line=1, column=1),
                    ),
                ]
            ),
        )
    ]


@pytest.fixture()
def computations_allowed_offset(
    no_offset: Dict[str, int], offset_by_one: Dict[str, int]
) -> Iterator[List[ComputationBlock]]:
    """
    Build stencil definition body writing to x after reading y with offset.

    equivalent to
    .. code-block: python

        with computation(PARALLEL), interval(...):
            out = in[-1, 0, 0]
            inout = in
    """
    yield [
        ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START), end=AxisBound(level=LevelMarker.END)
            ),
            iteration_order=IterationOrder.PARALLEL,
            body=BlockStmt(
                stmts=[
                    Assign(
                        target=FieldRef(
                            name="out",
                            offset=no_offset,
                            loc=Location(scope="allowed_offset", line=0, column=0),
                        ),
                        value=FieldRef(
                            name="in",
                            offset=offset_by_one,
                            loc=Location(scope="allowed_offset", line=0, column=2),
                        ),
                        loc=Location(scope="allowed_offset", line=0, column=1),
                    ),
                    Assign(
                        target=FieldRef(
                            name="inout",
                            offset=no_offset,
                            loc=Location(scope="allowed_offset", line=1, column=0),
                        ),
                        value=FieldRef(
                            name="in",
                            offset=no_offset,
                            loc=Location(scope="allowed_offset", line=1, column=2),
                        ),
                        loc=Location(scope="allowed_offset", line=1, column=1),
                    ),
                ]
            ),
        )
    ]


@pytest.fixture()
def computations(
    case: str,
    computations_extended: List[ComputationBlock],
    computations_offset: List[ComputationBlock],
    computations_allowed_offset: List[ComputationBlock],
) -> Iterator[Optional[List[ComputationBlock]]]:
    comps = None
    if case == "extended":
        comps = computations_extended
    elif case == "offset":
        comps = computations_offset
    elif case == "allowed_offset":
        comps = computations_allowed_offset
    yield comps


@pytest.fixture()
def expected_nblocks(case: str) -> Iterator[Optional[int]]:
    nblocks = None
    if case == "extended":
        nblocks = 2
    elif case == "offset":
        nblocks = 2
    elif case == "allowed_offset":
        nblocks = 1
    yield nblocks


@pytest.fixture()
def definition(
    case: str,
    domain: Domain,
    api_signature: List[ArgumentInfo],
    api_fields: List[FieldDecl],
    computations: List[ComputationBlock],
) -> StencilDefinition:
    yield StencilDefinition(
        name=case,
        domain=domain,
        api_signature=api_signature,
        api_fields=api_fields,
        parameters=[],
        computations=computations,
        docstring="",
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


@pytest.fixture()
def transform_data(
    case: str,
    definition: StencilDefinition,
) -> TransformData:
    yield TransformData(
        definition_ir=definition,
        implementation_ir=init_implementation_from_definition(definition),
        options=BuildOptions(name=case, module=__name__),
    )


@pytest.fixture()
def transform_data_after_init_pass(transform_data: TransformData):
    init_pass = InitInfoPass()
    yield init_pass.apply(transform_data)


@pytest.fixture()
def transform_data_after_normalize_blocks_pass(transform_data_after_init_pass: TransformData):
    normalize_blocks_pass = NormalizeBlocksPass()
    yield normalize_blocks_pass.apply(transform_data_after_init_pass)


@pytest.fixture()
def transform_data_after_compute_extents_pass(
    transform_data_after_normalize_blocks_pass: TransformData,
):
    compute_extents_pass = ComputeExtentsPass()
    yield compute_extents_pass.apply(transform_data_after_normalize_blocks_pass)


def test_merge_write_after_read(
    transform_data_after_compute_extents_pass: TransformData, expected_nblocks
):
    merge_blocks_pass = MergeBlocksPass()
    data = merge_blocks_pass.apply(transform_data_after_compute_extents_pass)

    assert len(data.blocks) == expected_nblocks
