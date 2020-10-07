from typing import Iterator, List, Set, Tuple, Union

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
    VarDecl,
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


def init_implementation_from_definition(definition: StencilDefinition) -> StencilImplementation:
    return StencilImplementation(
        name=definition.name,
        api_signature=[],
        domain=definition.domain,
        fields={},
        parameters={},
        splitters=[],
        multi_stages=[],
        fields_extents={},
        unreferenced=[],
        axis_splitters_var=None,
        externals=definition.externals,
        sources=definition.sources,
        docstring=definition.docstring,
    )


class TObject:
    def __init__(self, loc: Location, parent: "TObject" = None):
        self.loc = loc
        self.children = []
        self.parent = parent

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
            scope=self.child_scope,
        )
        child.parent = self
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

    def add_blocks(self, *blocks: "TComputationBlock") -> "TDefinition":
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

    @property
    def splitters(self) -> List[VarDecl]:
        return []

    def build(self) -> StencilDefinition:
        return StencilDefinition(
            name=self.name,
            domain=self.domain,
            api_signature=self.api_signature,
            api_fields=self.api_fields,
            parameters=self.parameters,
            computations=[block.build() for block in self.children],
            docstring=self.docstring,
            splitters=self.splitters,
            loc=self.loc,
        )

    def build_transform(self):
        definition = self.build()
        return TransformData(
            definition_ir=definition,
            implementation_ir=init_implementation_from_definition(definition),
            options=BuildOptions(name=self.name, module=__name__),
        )


class TComputationBlock(TObject):
    def __init__(
        self, *, order: IterationOrder, start: int = 0, end: int = 0, scope: str = "<unnamed>"
    ):
        super().__init__(Location(line=0, column=0))
        self.order = order
        self.start = start
        self.end = end
        self.scope = scope

    def add_statements(self, *stmts: "TStatement") -> "TComputationBlock":
        for stmt in stmts:
            self.register_child(stmt)
        return self

    @property
    def width(self) -> int:
        return 0

    def build(self) -> ComputationBlock:
        self.loc.scope = self.parent.child_scope
        return ComputationBlock(
            interval=AxisInterval(
                start=AxisBound(level=LevelMarker.START, offset=self.start),
                end=AxisBound(level=LevelMarker.END, offset=self.end),
            ),
            iteration_order=self.order,
            body=BlockStmt(
                stmts=[stmt.build() for stmt in self.children],
            ),
            loc=self.loc,
        )

    @property
    def child_scope(self) -> str:
        return f"{self.loc.scope}:{self.scope}"


class TStatement(TObject):
    pass


class TAssign(TStatement):
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
            line=self.loc.line,
            column=self.loc.column + self.target.width + 3,
            scope=self.loc.scope,
        )
        value.parent = self
        return value

    @property
    def field_names(self) -> Set[str]:
        return set.union(self.target.field_names, self.value.field_names)

    @property
    def target(self):
        return TFieldRef(
            name=self._target,
            parent=self,
            loc=Location(line=self.loc.line, column=self.loc.column, scope=self.loc.scope),
        )

    def build(self) -> Assign:
        self.loc.scope = self.parent.child_scope
        return Assign(
            target=self.target.build(),
            value=self.value.build(),
            loc=Location(
                line=self.loc.line,
                column=self.loc.column + self.target.width + 1,
                scope=self.loc.scope,
            ),
        )


class TFieldRef(TObject):
    def __init__(
        self,
        *,
        name: str,
        offset: Tuple[int, int, int] = (0, 0, 0),
        loc: Location = None,
        parent: TObject = None,
    ):
        super().__init__(loc or Location(line=0, column=0), parent=parent)
        self.name = name
        self.offset = make_offset(offset)

    def build(self):
        self.loc.scope = self.parent.child_scope
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
