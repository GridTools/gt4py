_BACKENDS = {}


def register_backend(name, backend):
    _BACKENDS[name] = backend


def get_backend(name):
    return _BACKENDS[name]
