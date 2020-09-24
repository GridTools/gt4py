from functools import partial
from typing import Iterator, List, Tuple, Set, Union

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
    Expr,
    FieldDecl,
    FieldRef,
    IterationOrder,
    LevelMarker,
    Location,
    StencilDefinition,
    StencilImplementation,
    Statement,
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


# TODO: clean up unused setup functions

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


class TObject:
    def __init__(self, loc: Location):
        self.loc = loc
        self.children = []

    @property
    def width(self) -> int:
        return sum(child.width for child in self.children) + 1 if self.children else 1

    @property
    def height(self) -> int:
        return sum(child.height for child in self.children) + 1 if self.children else 1

    def register_child(self, child: "TObject") -> None:
        child.loc = Location(
            line=self.loc.line + self.height,
            column=self.loc.column + self.width,
            scope=self.child_scope
        )
        self.children.append(child)

    @property
    def field_names(self) -> Set[str]:
        return set.union(*(child.field_names for child in self.children))

    @property
    def child_scope(self) -> str:
        return self.loc.scope


class TDefinition(TObject):
    def __init__(self, *, name: str, domain: Domain, fields: List[str]):
        super().__init__(Location(line=0, column=0, scope=name))
        self.name = name
        self.domain = domain
        self.fields = fields
        self.parameters = []
        self.docstring = ""

    def add_blocks(self, *blocks: List[ComputationBlock]) -> "TDefinition":
        for block in blocks:
            self.register_child(block)
        return self

    @property
    def width(self) -> int:
        return 0

    @property
    def height(self) -> int:
        return super().height - 1

    @property
    def api_signature(self) -> List[ArgumentInfo]:
        return [ArgumentInfo(name=n, is_keyword=False) for n in self.fields]

    @property
    def api_fields(self) -> List[FieldDecl]:
        tmp_field_names = self.field_names.difference(self.fields)
        tmp_fields = [
            FieldDecl(name=n, data_type=DataType.AUTO, axes=self.domain.axes_names, is_api=False)
            for n in tmp_field_names
        ]
        return tmp_fields + [
            FieldDecl(name=n, data_type=DataType.AUTO, axes=self.domain.axes_names, is_api=True)
            for n in self.fields
        ]

    def build(self) -> StencilDefinition:
        return StencilDefinition(
            name=self.name,
            domain=self.domain,
            api_signature=self.api_signature,
            api_fields = self.api_fields,
            parameters=self.parameters,
            computations=[block.build() for block in self.children],
            docstring=self.docstring,
        )

    def build_transform(self):
        definition = self.build()
        return TransformData(
            definition_ir=definition,
            implementation_ir=init_implementation_from_definition(definition),
            options=BuildOptions(name=self.name, module=__name__),
        )


class TComputationBlock(TObject):
    def __init__(self, *, order: IterationOrder, start: int, end: int, scope: str = "<unnamed>"):
        super().__init__(Location(line=0, column=0, scope=""))
        self.order = order
        self.start = start
        self.end = end
        self.scope = scope

    def add_statements(self, *stmts: List[Statement]) -> "TComputationBlock":
        for stmt in stmts:
            self.register_child(stmt)
        return self

    @property
    def width(self) -> int:
        return 0

    def build(self) -> ComputationBlock:
        return ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START, offset=self.start),
                end=AxisBound(level=LevelMarker.END, offset=self.end),
            ),
            iteration_order=self.order,
            body=BlockStmt(
                stmts=[stmt.build() for stmt in self.children],
            )
        )

    @property
    def child_scope(self) -> str:
        return f"{self.loc.scope}:{self.scope}"


class TAssign(TObject):
    def __init__(self, target: str, value: Union[str, Expr], offset: Tuple[int, int, int]):
        super().__init__(Location(line=0, column=0))
        self._target = target
        self._value = value
        self.offset = offset

    @property
    def height(self):
        return 1

    @property
    def width(self):
        return self.target.width + 3 + self.value.width

    @property
    def value(self):
        value = self._value
        if isinstance(self._value, str):
            value = TFieldRef(name=self._value, offset=self.offset)
        value.loc = Location(
            line=self.loc.line, column=self.loc.column + self.target.width + 3, scope=self.loc.scope
            )
        return value

    @property
    def field_names(self) -> Set[str]:
        return set.union(self.target.field_names, self.value.field_names)

    @property
    def target(self):
        return TFieldRef(name=self._target, loc=Location(
            line=self.loc.line, column=self.loc.column, scope=self.loc.scope
        ))

    def build(self) -> Assign:
        return Assign(
            target=self.target.build(),
            value=self.value.build(),
            loc=Location(line=self.loc.line, column=self.loc.column + self.target.width + 1, scope=self.loc.scope)
        )


class TFieldRef(TObject):
    def __init__(self, *, name: str, offset: Tuple[int, int, int] = (0, 0, 0), loc: Location = None):
        super().__init__(loc or Location(line=0, column=0))
        self.name = name
        self.offset = make_offset(offset)

    def build(self):
        return FieldRef(
            name=self.name,
            offset=self.offset,
            loc=self.loc,
        )

    @property
    def height(self) -> int:
        return 1

    @property
    def width(self) -> int:
        return len(self.name)

    @property
    def field_names(self) -> Set[str]:
        return {self.name}


def make_definition_multiple(
    name: str,
    fields: List[str],
    domain: Domain,
    info: List[Tuple[BodyType, IterationOrder, Tuple[int, int]]],
) -> StencilDefinition:
    api_signature = [ArgumentInfo(name=n, is_keyword=False) for n in fields]
    bodies = []
    for definition in info:
        bodies.extend(definition[0])
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
