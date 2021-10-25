from typing import Any

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import TemplatedGenerator
from iterator.backends import backend
from iterator.transforms import apply_common_transforms


class ToLispLike(TemplatedGenerator):
    Sym = as_fmt("{id}")
    FunCall = as_fmt("({fun} {' '.join(args)})")
    IntLiteral = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}")
    StringLiteral = as_fmt("{value}")
    SymRef = as_fmt("{id}")
    Program = as_fmt(
        """
    {''.join(function_definitions)}
    {''.join(fencil_definitions)}
    {''.join(setqs)}
    """
    )
    StencilClosure = as_fmt(
        """(
     :domain {domain}
     :stencil {stencil}
     :outputs {' '.join(outputs)}
     :inputs {' '.join(inputs)}
    )
    """
    )
    FencilDefinition = as_fmt(
        """(defen {id}({' '.join(params)})
        {''.join(closures)})
        """
    )
    FunctionDefinition = as_fmt(
        """(defun {id}({' '.join(params)})
        {expr}
        )

"""
    )
    Lambda = as_fmt(
        """(lambda ({' '.join(params)})
         {expr}
          )"""
    )

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        transformed = apply_common_transforms(
            root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
        )
        generated_code = super().apply(transformed, **kwargs)
        try:
            from yasi import indent_code

            indented = indent_code(generated_code, "--dialect lisp")
            return "".join(indented["indented_code"])
        except ImportError:
            return generated_code


backend.register_backend(
    "lisp", lambda prog, *args, **kwargs: print(ToLispLike.apply(prog, **kwargs))
)
