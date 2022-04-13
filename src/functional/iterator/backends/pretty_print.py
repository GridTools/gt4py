from functional.iterator.backends import backend
from functional.iterator.pretty_printer import pretty_print


backend.register_backend("pretty_print", lambda prog, *args, **kwargs: pretty_print(prog))
