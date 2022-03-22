from functional.iterator import lisp
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


def ir_to_lisp_with_common_transforms(root, **kwargs):
    def verify_rountrip(ir):
        lisp_str = lisp.ir_to_lisp(ir, **kwargs)
        roundtrip_ir = lisp.lisp_to_ir(lisp_str)
        assert roundtrip_ir == ir
        roundtrip_ir = lisp.lisp_to_ir_using_lisp(lisp_str)
        assert roundtrip_ir == ir
        lisp_str = lisp.pretty_format(lisp_str)
        return lisp_str

    verify_rountrip(root)
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
    )
    return verify_rountrip(transformed)


backend.register_backend(
    "lisp", lambda prog, *args, **kwargs: print(ir_to_lisp_with_common_transforms(prog, **kwargs))
)
