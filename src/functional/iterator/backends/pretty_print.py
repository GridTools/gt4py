from typing import Any

from functional.iterator import ir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat


def pretty_print_and_check(root: ir.Node, *args: Any, **kwargs: Any) -> None:
    pretty = pformat(root)
    print(pretty)
    parsed = pparse(pretty)

    assert parsed == root
