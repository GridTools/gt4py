# -*- coding: utf-8 -*-
class StorageDefaults:
    def __init__(self, *, device="cpu", alignment_size=None, layout=None):
        self.alignment_size = alignment_size
        self.layout = layout
        self.device = device


REGISTRY = dict(
    C=StorageDefaults(layout=lambda dims: tuple(range(len(dims)))),
    F=StorageDefaults(layout=lambda dims: tuple(reversed(range(len(dims))))),
)


def register_storage_defaults(name, defaults):

    if not isinstance(name, str):
        raise TypeError("default parameter key must be a string.")
    if name in REGISTRY:
        raise ValueError(f"default parameters with key '{name}' already registered")

    if not isinstance(defaults, StorageDefaults):
        raise TypeError("invalid type for parameter defaults")

    REGISTRY[name] = defaults


def get_default_parameters(key) -> StorageDefaults:
    if key not in REGISTRY:
        raise ValueError(f"no default parameters known for 'key' \"{key}\"")
    return REGISTRY[key]
