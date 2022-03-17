from functional.iterator import lisp
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


def ir_to_lisp_with_common_transforms(root, **kwargs):
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
    )
    lisp_str = lisp.ir_to_lisp(transformed, **kwargs)
    assert lisp.lisp_to_ir(lisp_str) == transformed
    lisp_str = lisp.pretty_format(lisp_str)
    return lisp_str


backend.register_backend(
    "lisp", lambda prog, *args, **kwargs: print(ir_to_lisp_with_common_transforms(prog, **kwargs))
)
