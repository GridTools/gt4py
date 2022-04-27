from typing import Any

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import TemplatedGenerator
from functional.iterator.backends import backend
from functional.iterator.transforms import apply_common_transforms


class ToLispLike(TemplatedGenerator):
    Sym = as_fmt("{id}")
    FunCall = as_fmt("({fun} {' '.join(args)})")
    Literal = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}")
    SymRef = as_fmt("{id}")
    StencilClosure = as_fmt(
        """(
     :domain {domain}
     :stencil {stencil}
     :output {output}
     :inputs {' '.join(inputs)}
    )
    """
    )
    FencilDefinition = as_fmt(
        """
        ({' '.join(function_definitions)})
        (defen {id}({' '.join(params)})
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
