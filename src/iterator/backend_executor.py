from devtools import debug

from iterator.backends import backend
from iterator.ir import Program


def execute_program(prog: Program, *args, **kwargs):
    assert "backend" in kwargs
    assert len(prog.fencil_definitions) == 1

    if "debug" in kwargs and kwargs["debug"]:
        debug(prog)

    if not len(args) == len(prog.fencil_definitions[0].params):
        raise RuntimeError("Incorrect number of arguments")

    if kwargs["backend"] in backend._BACKENDS:
        b = backend.get_backend(kwargs["backend"])
        b(prog, *args, **kwargs)
    else:
        raise RuntimeError(f"Backend {kwargs['backend']} is not registered.")
