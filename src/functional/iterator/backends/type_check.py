from functional.iterator import type_inference
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms, global_tmps


def check(root, *args, **kwargs):
    type_inference.pprint(type_inference.infer(root))
    transformed = apply_common_transforms(
        root, lift_mode=kwargs.get("lift_mode"), offset_provider=kwargs["offset_provider"]
    )
    if isinstance(transformed, global_tmps.FencilWithTemporaries):
        transformed = transformed.fencil
    type_inference.pprint(type_inference.infer(transformed))


backend.register_backend("type_check", check)
