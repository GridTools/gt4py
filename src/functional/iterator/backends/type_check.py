from functional.iterator import type_inference
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms
from functional.iterator.transforms.global_tmps import FencilWithTemporaries


def check(root, *args, **kwargs):
    print(type_inference.pretty_str(type_inference.infer(root)))
    transformed = apply_common_transforms(
        root, lift_mode=kwargs.get("lift_mode"), offset_provider=kwargs["offset_provider"]
    )
    if isinstance(transformed, FencilWithTemporaries):
        transformed = transformed.fencil
    print(type_inference.pretty_str(type_inference.infer(transformed)))


backend.register_backend("type_check", check)
