from functional.iterator import type_inference
from functional.iterator.processor_interface import fencil_formatter
from functional.iterator.transforms import apply_common_transforms, global_tmps


@fencil_formatter
def check(root, *args, **kwargs) -> str:
    type_inference.pprint(type_inference.infer(root))
    transformed = apply_common_transforms(
        root, lift_mode=kwargs.get("lift_mode"), offset_provider=kwargs["offset_provider"]
    )
    if isinstance(transformed, global_tmps.FencilWithTemporaries):
        transformed = transformed.fencil
    return type_inference.pformat(type_inference.infer(transformed))
