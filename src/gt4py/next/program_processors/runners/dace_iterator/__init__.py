import gt4py.next.iterator.ir as itir
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.iterator.transforms import apply_common_transforms
from .itir_to_sdfg import ItirToSDFG
from gt4py.next.type_system import type_translation

@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    offset_provider = kwargs["offset_provider"]
    arg_types = [type_translation.from_value(arg) for arg in args]

    program = apply_common_transforms(program, offset_provider=offset_provider, force_inline_lift=True)
    sdfg_gen = ItirToSDFG(param_types=arg_types, offset_provider=offset_provider)
    sdfg_gen.visit(program)

    sdfg = sdfg_gen.sdfg
    sdfg.view()

    raise ValueError("")
