from functional.iterator import type_inference
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


def check(root, *args, **kwargs):
    print(type_inference.pretty_str(type_inference.infer(root)))
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
    )
    print(type_inference.pretty_str(type_inference.infer(transformed)))


backend.register_backend("type_check", check)
