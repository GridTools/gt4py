import ast
import copy
import pathlib
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, cast

from . import gtir


BOUNDARY_TYPE = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


CALL_RUN_TPL = """
self._call_run(
    field_args=field_args,
    parameter_args=parameter_args,
    domain=domain,
    origin=origin,
    validate_args=validate_args,
    exec_info=exec_info,
)"""

ACCESSOR_TPL = """
class _Accessor:
    def __init__(self, array, origin):
        self.array = array
        self.origin = origin

    def _shift(self, index):
        return tuple(i + offset for i, offset in zip(index, self.origin))

    def __getitem__(self, index):
        return self.array[self._shift(index)]

    def __setitem__(self, index, value):
        self.array[self._shift(index)] = value"""


def parse_snippet(snippet: str) -> List[ast.AST]:
    tree = ast.parse(snippet)
    # if multiline just return all
    if len(tree.body) > 1:
        return cast(List[ast.AST], tree.body)
    # grab the top level item if there is only one
    top = tree.body[0]
    # expressions get wrapped, return the content
    if isinstance(top, ast.Expr):
        return [top.value]
    # statements are returned unchanged
    return [top]


def parse_node(snippet: str) -> ast.AST:
    nodes = parse_snippet(snippet)
    if len(nodes) > 1:
        raise ValueError("Snippet '%s' got parsed into more than one node.", snippet)
    return nodes[0]


def assign_to_gt_name(name: str, node: Optional[ast.AST]):
    return ast.Assign(
        targets=[ast.Name(id=f"_gt_{name}_")],
        value=node or ast.NameConstant(value=None),
    )


class ImportBuilder:
    def __init__(self):
        self._names: List[str] = None
        self._asname: str = None
        self._from: str = None

    def import_(self, *args: str) -> "ImportBuilder":
        if self._asname is not None and len(args) > 1:
            raise ValueError("can not import multiple names 'as'")
        self._name = args
        return self

    def as_(self, asname: str) -> "ImportBuilder":
        if len(self._name) > 1:
            raise ValueError("can not import multiple names 'as'")
        self._asname = asname
        return self

    def from_(self, fromname: str) -> "ImportBuilder":
        self._from = fromname
        return self

    def build(self) -> Union[ast.Import, ast.ImportFrom]:
        names = [ast.alias(name=name, asname=self._asname) for name in self._names]
        if self._from:
            return ast.ImportFrom(module=self._from, names=names)
        return ast.Import(names=names)


class FieldInfoBuilder:
    def __init__(self):
        self._access: gtir.AccessKind = None
        self._boundary: ast.Call = None
        self._dtype: str = None

    def access(self, access_kind: gtir.AccessKind) -> "FieldInfoBuilder":
        self._access = access_kind
        return self

    def boundary(self, boundary: BOUNDARY_TYPE) -> "FieldInfoBuilder":
        boundary_ast = parse_node(repr(boundary))
        self._boundary = ast.Call(
            func=ast.Name(id="Boundary"),
            args=[boundary_ast],
            keywords=[],
        )
        return self

    def dtype(self, dtype_name: str) -> "FieldInfoBuilder":
        self._dtype = dtype_name
        return self

    def build(self) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="FieldInfo"),
            args=[],
            keywords=[
                ast.keyword(
                    arg="access",
                    value=ast.Attribute(value=ast.Name(id="AccessKind"), attr=self._access.name),
                ),
                ast.keyword(
                    arg="boundary",
                    value=self._boundary,
                ),
                ast.keyword(
                    arg="dtype",
                    value=parse_node(f"np.{self._dtype}"),
                ),
            ],
        )


class FunctionDefBuilder:
    def __init__(self):
        self._name: str = None
        self._args: List[ast.arg] = []
        self._defaults: List[ast.AST] = []
        self._kwonlyargs: List[ast.arg] = []
        self._kw_defaults: List[ast.AST] = []
        self._decorator_list: List[ast.AST] = []
        self._returns: ast.AST = None
        self._body: List[ast.AST] = []

    def name(self, name: str) -> "FunctionDefBuilder":
        self._name = name
        return self

    def args(self, *defaultless: str, **withdefaults: ast.AST) -> "FunctionDefBuilder":
        self._args.extend([ast.arg(arg=name, annotation=None) for name in defaultless])
        self._args.extend([ast.arg(arg=name, annotation=None) for name in withdefaults])
        self._defaults.extend(withdefaults.values())
        return self

    def kwonly(self, *defaultless: str, **withdefaults: ast.AST) -> "FunctionDefBuilder":
        self._kwonlyargs.extend([ast.arg(arg=name, annotation=None) for name in defaultless])
        self._kwonlyargs.extend([ast.arg(arg=name, annotation=None) for name in withdefaults])
        self._kw_defaults.extend(withdefaults.values())
        return self

    def add_body(self, node: ast.AST) -> "FunctionDefBuilder":
        self._body.append(node)
        return self

    def build(self) -> ast.FunctionDef:
        return ast.FunctionDef(
            name=self._name,
            args=ast.arguments(
                args=self._args,
                defaults=self._defaults,
                kwonlyargs=self._kwonlyargs,
                kw_defaults=self._kw_defaults,
                vararg=None,
                kwarg=None,
            ),
            body=self._body,
            decorator_list=self._decorator_list,
            returns=self._returns,
        )


class StrDictBuilder:
    def __init__(self):
        self._keys = []
        self._values = []

    def add_item(self, key: str, node: ast.AST) -> "StrDictBuilder":
        self._keys.append(parse_node(f"'{key}'"))
        self._values.append(node)
        return self

    def add_items(self, *args: Tuple[str, ast.AST]) -> "StrDictBuilder":
        for key, node in args:
            self.add_item(key, node)
        return self

    def build(self) -> ast.Dict:
        node = cast(ast.Dict, parse_node("{}"))
        node.keys = self._keys
        node.values = self._values
        return node


class StencilClassBuilder:

    DEFAULT_DOMAIN_INFO = parse_node(
        "DomainInfo(parallel_axes=('I', 'J'), sequential_axis='K', ndims=3)"
    )

    def __init__(self):
        self._name: str = None
        self._run_body: List[ast.AST] = []
        self._field_names: List[str] = []
        self._parameter_names: List[str] = []
        self._class_attrs: OrderedDict[str, ast.AST] = OrderedDict(
            [
                ("backend", None),
                ("source", None),
                ("domain_info", copy.deepcopy(self.DEFAULT_DOMAIN_INFO)),
                ("field_info", parse_node("{}")),
                ("parameter_info", parse_node("{}")),
                ("constants", parse_node("{}")),
                ("options", parse_node("{}")),
            ]
        )

    def name(self, name: str) -> "StencilClassBuilder":
        self._name = name
        return self

    def field_names(self, *names: str) -> "StencilClassBuilder":
        self._field_names.extend(names)
        return self

    def parameter_names(self, *names: str) -> "StencilClassBuilder":
        self._parameter_names = list(names)
        return self

    @property
    def signature(self) -> List[str]:
        return self._field_names + self._parameter_names

    @property
    def signature_args(self) -> List[ast.arg]:
        return [ast.arg(arg=name, annotation=None) for name in self.signature]

    @property
    def accessors(self) -> List[ast.Assign]:
        return [
            cast(ast.Assign, parse_node(f"{name}_at = _Accessor({name}, _origin_['{name}'])"))
            for name in self._field_names
        ]

    def add_run_line(self, node: ast.AST) -> "StencilClassBuilder":
        self._run_body.append(node)
        return self

    def backend(self, backend_name: str) -> "StencilClassBuilder":
        self._class_attrs["backend"] = ast.Str(s=backend_name)
        return self

    def source(self, source: str) -> "StencilClassBuilder":
        self._class_attrs["source"] = ast.Str(s=source)
        return self

    def add_gt_field_info(self, *, name: str, field_info: ast.Call) -> "StencilClassBuilder":
        field_info_dict = cast(ast.Dict, self._class_attrs["field_info"])
        field_info_dict.keys.append(ast.Str(s=name))
        field_info_dict.values.append(field_info)
        return self

    @property
    def class_attrs(self) -> List[ast.Assign]:
        return [assign_to_gt_name(key, value) for key, value in self._class_attrs.items()]

    def build_call_def(self) -> ast.FunctionDef:
        exec_info = parse_node(
            "\n".join(
                [
                    "if exec_info is not None:",
                    "    exec_info['call_start_time'] = time.perf_counter()",
                ]
            )
        )

        return (
            FunctionDefBuilder()
            .name("__call__")
            .args(
                "self",
                *self.signature,
                domain=parse_node("None"),
                origin=parse_node("None"),
                validate_args=parse_node("True"),
                exec_info=parse_node("None"),
            )
            .add_body(exec_info)
            .add_body(
                ast.Assign(
                    targets=[parse_node("field_args")],
                    value=StrDictBuilder()
                    .add_items(*((name, parse_node(name)) for name in self._field_names))
                    .build(),
                )
            )
            .add_body(
                ast.Assign(
                    targets=[parse_node("parameter_args")],
                    value=StrDictBuilder()
                    .add_items(*((name, parse_node(name)) for name in self._parameter_names))
                    .build(),
                )
            )
            .add_body(ast.Expr(value=parse_node(CALL_RUN_TPL)))
            .build()
        )

    def build_run_def(self) -> ast.FunctionDef:
        funcdef = cast(
            ast.FunctionDef,
            parse_node(
                "\n".join(
                    [
                        "def run(self, _domain_, _origin_, exec_info, *, placeholder):",
                        "    if exec_info is not None:",
                        "        exec_info['domain'] = _domain_",
                        "        exec_info['origin'] = _origin_",
                        "        exec_info['run_start_time'] = time.perf_counter()",
                    ]
                )
            ),
        )
        funcdef.args.kwonlyargs = self.signature_args
        funcdef.args.kw_defaults = []
        funcdef.body = [cast(ast.stmt, node) for node in [*self.accessors, *self._run_body]]
        return funcdef

    def build_properties(self) -> List[ast.FunctionDef]:
        if any(value is None for value in self._class_attrs.values()):
            none_keys = [key for key, value in self._class_attrs.items() if value is None]
            raise ValueError(f"Missing stencil class attributes: {none_keys}")
        return [
            cast(
                ast.FunctionDef,
                parse_node(f"@property\ndef {name}(self):\n    return self._gt_{name}_"),
            )
            for name in self._class_attrs
        ]

    def build(self) -> ast.ClassDef:
        return ast.ClassDef(
            name=self._name,
            bases=[ast.Name(id="StencilObject")],
            body=[
                *self.class_attrs,
                *self.build_properties(),
                self.build_call_def(),
                self.build_run_def(),
            ],
            decorator_list=[],
        )


class StencilModuleBuilder:
    DEFAULT_IMPORTS = [
        parse_node("import time"),
        parse_node("import numpy as np"),
        parse_node("from numpy import dtype"),
        parse_node(
            "from gt4py.stencil_object import "
            "AccessKind, "
            "Boundary, "
            "DomainInfo, "
            "FieldInfo, "
            "ParameterInfo, "
            "StencilObject"
        ),
    ]
    DEFAULT_NAME = "stencil"

    def __init__(self):
        self._imports = self.DEFAULT_IMPORTS
        self._paths = []
        self._name = self.DEFAULT_NAME
        self._stencil_class = StencilClassBuilder()

    def stencil_class(self, builder: StencilClassBuilder) -> "StencilModuleBuilder":
        self._stencil_class = builder
        return self

    def name(self, name: str) -> "StencilModuleBuilder":
        self._name = name
        return self

    def import_(self, import_node: Union[ast.Import, ast.ImportFrom]) -> "StencilModuleBuilder":
        self._imports.append(import_node)
        return self

    def add_paths(self, *paths: pathlib.Path) -> "StencilModuleBuilder":
        self._paths.extend(paths)
        return self

    def build(self) -> ast.Module:
        add_paths = []
        if self._paths:
            add_paths = parse_snippet(
                f"import sys\nsys.path.extend([{', '.join([repr(str(path)) for path in self._paths])}])\n\n"
            )
        return ast.Module(
            body=add_paths
            + self._imports
            + [parse_node(ACCESSOR_TPL)]
            + [self._stencil_class.name(self._name).build()]
        )
