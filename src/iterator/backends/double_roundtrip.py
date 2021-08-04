from eve.concepts import Node
from iterator.backends import backend, embedded


def executor(ir: Node, *args, **kwargs):
    embedded.executor(ir, *args, dispatch_backend=embedded._BACKEND_NAME, **kwargs)


backend.register_backend("double_roundtrip", executor)
