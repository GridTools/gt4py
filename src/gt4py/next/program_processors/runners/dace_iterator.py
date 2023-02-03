import functional.iterator.ir as itir
from functional.program_processors.processor_interface import program_executor


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    raise ValueError("")
