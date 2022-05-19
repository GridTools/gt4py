from functional.iterator import type_inference
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


def check(root, *args, **kwargs):
    type_inference.pprint(type_inference.infer(root))
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
    )
    type_inference.pprint(type_inference.infer(transformed))


backend.register_backend("type_check", check)
