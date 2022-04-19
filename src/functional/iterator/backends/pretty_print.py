from functional.iterator.backends import backend
from functional.iterator.pretty_printer import pprint


backend.register_backend("pretty_print", lambda prog, *args, **kwargs: pprint(prog))
