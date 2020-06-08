from gt4py import backend as gt_backend


class StorageDefaults:
    def __init__(self, *, gpu=None, alignment=None, layout_map=None):
        self.alignment = alignment
        self.layout_map = layout_map
        self.gpu = True if gpu else False


REGISTRY = dict(
    C=StorageDefaults(layout_map=lambda ndims: tuple(range(ndims))),
    F=StorageDefaults(layout_map=lambda ndims: tuple(reversed(range(ndims)))),
)


def register(name, defaults):

    if not isinstance(name, str):
        raise TypeError("default parameter key must be a string.")
    if name in REGISTRY:
        raise ValueError(f"default parameters with key '{name}' already registered")

    if not isinstance(defaults, StorageDefaults):
        raise TypeError("invalid type for parameter defaults")

    REGISTRY[name] = defaults


def get_default_parameters(key):
    if key not in REGISTRY:
        raise ValueError(f"no default parameters known for 'key' \"{key}\"")
    return REGISTRY[key]
