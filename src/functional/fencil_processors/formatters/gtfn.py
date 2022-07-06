from functional.fencil_processors.codegens.gtfn.gtfn_backend import _guess_grid_type, generate
from functional.iterator import ir as itir
from functional.iterator.processor_interface import fencil_formatter


@fencil_formatter
def format_sourcecode(fencil: itir.FencilDefinition, *arg, **kwargs) -> str:
    return generate(fencil, grid_type=_guess_grid_type(**kwargs), **kwargs)
