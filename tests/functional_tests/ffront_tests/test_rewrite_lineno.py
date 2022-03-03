import ast
import textwrap

from eve.pattern_matching import ObjectPattern as P
from functional.ffront.ast_passes.rewrite_lineno import RewriteLineNumbers
from functional.ffront.source_utils import SourceDefinition


def square(a):
    return a ** 2


def test_rewrite_lineno():
    source, filename, starting_line = SourceDefinition.from_function(square)
    ast_node = ast.parse(textwrap.dedent(source)).body[0]
    new_ast_node = RewriteLineNumbers.apply(ast_node, starting_line)
    expected_pattern = P(
        ast.FunctionDef,
        args=P(ast.arguments, args=[P(ast.arg, lineno=8)]),
        body=[P(ast.Return, value=P(ast.BinOp, op=P(ast.Pow, lineno=9), lineno=9))],
        lineno=8,
    )
    expected_pattern.match(new_ast_node, raise_exception=True)
