from typing import Any

from functional.iterator import ir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat
from functional.iterator.processor_interface import fencil_formatter


@fencil_formatter
def pretty_format_and_check(root: ir.Node, *args: Any, **kwargs: Any) -> str:
    pretty = pformat(root)
    parsed = pparse(pretty)
    assert parsed == root
    return pretty
