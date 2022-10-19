import ast
import textwrap
import pytest
from functional.ffront.ast_passes.simple_assign import UnpackedIterablePass


def getlines(node: ast.AST) -> list[str]:
    return ast.unparse(node).splitlines()


def unpack_iterable(code: str) -> ast.AST:
    return UnpackedIterablePass.apply(ast.parse(textwrap.dedent(code)))


def test_unpack_iterable():
    """Unpacking tuples using a starred expression should correctly assign the starred subexpression to a tuple of its corresponding targets.
    """
    lines = getlines(
        unpack_iterable(
            """
            a, *b, c = (1, 2, 3, 4, 5)
            """
        )
    )
    assert len(lines) == 3
    assert lines[0] == "a = (1, 2, 3, 4, 5)[0]"
    assert lines[1] == "b = (1, 2, 3, 4, 5)[1:3]"
    assert lines[2] == "c = (1, 2, 3, 4, 5)[-1]"


def test_unpack_iterable_multiple_starred():
    """Only one starred operator is allowed per assignment.
    """
    with pytest.raises(ValueError):
        lines = getlines(
            unpack_iterable(
                """
                a, *b, *c = (1, 2, 3, 4, 5)
                """
            )
        )
