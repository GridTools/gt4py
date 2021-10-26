from eve.concepts import Node
from functional.iterator.backends import backend, roundtrip


def executor(ir: Node, *args, **kwargs):
    roundtrip.executor(ir, *args, dispatch_backend=roundtrip._BACKEND_NAME, **kwargs)


backend.register_backend("double_roundtrip", executor)
