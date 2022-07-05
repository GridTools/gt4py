from eve.concepts import Node
from functional.fencil_processors import roundtrip
from functional.iterator.processor_interface import fencil_executor


@fencil_executor
def executor(ir: Node, *args, **kwargs):
    roundtrip.executor(ir, *args, dispatch_backend=roundtrip.executor, **kwargs)
