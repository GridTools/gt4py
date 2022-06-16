from typing import Callable

from devtools import debug

from functional.iterator.ir import FencilDefinition


def execute_fencil(
    fencil: FencilDefinition,
    *args,
    backend: Callable,
    **kwargs,
):
    if "debug" in kwargs and kwargs["debug"]:
        debug(fencil)

    if not len(args) == len(fencil.params):
        raise RuntimeError("Incorrect number of arguments")

    backend(fencil, *args, **kwargs)
