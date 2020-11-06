from typing import List, Union

from eve import Str

from gtc2.gtir import AxisBound, Expr, FieldDecl, FieldsMetadata, LocNode, Stmt


class IJLoop(LocNode):
    body: List[Union[Expr, Stmt]]


class IndexFromStart(LocNode):
    idx: int


class IndexFromEnd(LocNode):
    idx: int


class KLoop(LocNode):
    lower: AxisBound
    upper: AxisBound
    ij_loops: List[IJLoop]


class RunFunction(LocNode):
    field_params: List[Str]
    scalar_params: List[Str]
    k_loops: List[KLoop]


class Module(LocNode):
    run: RunFunction


class StencilObject(LocNode):
    name: Str
    params: List[FieldDecl]
    fields_metadata: FieldsMetadata


class Stencil(LocNode):
    computation: Module
    stencil_obj: StencilObject
