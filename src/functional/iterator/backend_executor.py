from devtools import debug

from functional.iterator.backends import backend
from functional.iterator.ir import FencilDefinition


def execute_fencil(fencil: FencilDefinition, *args, **kwargs):
    assert "backend" in kwargs

    if "debug" in kwargs and kwargs["debug"]:
        debug(fencil)

    if not len(args) == len(fencil.params):
        raise RuntimeError("Incorrect number of arguments")

    if kwargs["backend"] in backend._BACKENDS:
        b = backend.get_backend(kwargs["backend"])
        b(fencil, *args, **kwargs)
    else:
        raise RuntimeError(f"Backend {kwargs['backend']} is not registered.")
