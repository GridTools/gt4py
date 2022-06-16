from eve.concepts import Node
from functional.iterator.backends import roundtrip


def executor(ir: Node, *args, **kwargs):
    roundtrip.executor(ir, *args, dispatch_backend=roundtrip.executor, **kwargs)
