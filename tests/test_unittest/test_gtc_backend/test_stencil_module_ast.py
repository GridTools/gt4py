import ast

import astor
import black
import numpy
import pytest

from gt4py.backend.gtc_backend.stencil_module_builder import (
    StencilClassBuilder,
    StencilModuleBuilder,
    parse_node,
    parse_snippet,
)
from gt4py.stencil_object import StencilObject
from gt4py.utils import make_module_from_file


def test_parse_snippet():
    # test multi item
    import_a, import_b, hello = parse_snippet("import a\nimport b\nprint('hello world')")
    assert isinstance(import_a, ast.Import)
    assert isinstance(import_b, ast.Import)
    # wrapper expression kept in multiline
    assert isinstance(hello, ast.Expr)
    # expression content returned for single expression
    assert isinstance(parse_snippet("print('hello world')")[0], ast.Call)
    assert isinstance(parse_snippet("hello")[0], ast.Name)
    # statements
    assert isinstance(parse_snippet("a = b")[0], ast.Assign)


def test_parse_node():
    with pytest.raises(ValueError):
        import_a, import_b, hello = parse_node("import a\nimport b\nprint('hello world')")
    assert isinstance(parse_node("{}"), ast.Dict)


def test_stencil_module(tmp_path):
    module_file = tmp_path / "test_stencil_module.py"
    ast_mod = (
        StencilModuleBuilder()
        .name("test_stencil")
        .stencil_class(
            StencilClassBuilder().backend("test_backend").source("pass").field_names("a", "b")
        )
        .build()
    )

    print(astor.dump_tree(ast_mod))
    print(black.format_str(astor.to_source(ast_mod), mode=black.Mode()))
    module_file.write_text(astor.to_source(ast_mod))
    module = make_module_from_file("test_stencil_module", module_file)
    stencil = module.test_stencil()
    assert isinstance(stencil, StencilObject)
    stencil(numpy.array([[[1, 2]]]), numpy.array([[[0, 0]]]))
