import gt4py.next.iterator.ir as itir
from gt4py.next.program_processors.processor_interface import program_executor


@program_executor
def run_dace_fieldview(program: itir.FencilDefinition, *args, **kwargs) -> None:
    raise ValueError("")
