from functional.iterator.backends import backend
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat


def pretty_print_and_check(root, *args, **kwargs):
    pretty = pformat(root)
    print(pretty)
    assert pparse(pretty) == root


backend.register_backend("pretty_print", pretty_print_and_check)
