from typing import Any

from functional.iterator import ir, lisp
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


def ir_to_lisp_with_common_transforms(root: ir.Node, **kwargs: Any) -> str:
    def verify_rountrip(iir: ir.Node) -> str:
        lisp_str = lisp.ir_to_lisp(iir, **kwargs)
        roundtrip_ir = lisp.lisp_to_ir(lisp_str)
        assert roundtrip_ir == iir
        try:
            roundtrip_ir = lisp.lisp_to_ir_using_lisp(lisp_str)
            assert roundtrip_ir == iir
        except FileNotFoundError:
            # No Scheme binary available
            pass
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
