import importlib.util
import pathlib
import tempfile
from typing import Iterable

import numpy as np

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import Node
from functional import iterator
from functional.iterator import ir
from functional.iterator.backends import backend
from functional.iterator.ir import AxisLiteral, OffsetLiteral
from functional.iterator.transforms import apply_common_transforms
from functional.iterator.transforms.global_tmps import FencilWithTemporaries


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    Literal = as_fmt("{value}")
    NoneLiteral = as_fmt("None")
    OffsetLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("(lambda ${','.join(params)}: ${expr})")
    StencilClosure = as_mako("closure(${domain}, ${stencil}, ${output}, [${','.join(inputs)}])")
    FencilDefinition = as_mako(
        """
${''.join(function_definitions)}
@fendef
def ${id}(${','.join(params)}):
    ${'\\n    '.join(closures)}
    """
    )
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )

    # extension required by global_tmps
    def visit_FencilWithTemporaries(self, node, **kwargs):
        params = self.visit(node.params)

        def np_dtype(dtype):
            if isinstance(dtype, int):
                return params[dtype] + ".dtype"
            if isinstance(dtype, tuple):
                return "np.dtype([" + ", ".join(f"('', {np_dtype(d)})" for d in dtype) + "])"
                return np.dtype([("", np_dtype(d)) for d in dtype])
            return f"np.dtype('{dtype}')"

        tmps = "\n    ".join(self.visit(node.tmps, np_dtype=np_dtype))
        args = ", ".join(params + [tmp.id for tmp in node.tmps])
        params = ", ".join(params)
        fencil = self.visit(node.fencil)
        return (
            fencil
            + "\n"
            + f"def {node.fencil.id}_wrapper({params}, **kwargs):\n    "
            + tmps
            + f"\n    {node.fencil.id}({args}, **kwargs)\n"
        )

    def visit_Temporary(self, node, *, np_dtype, **kwargs):
        assert isinstance(node.domain, ir.FunCall) and node.domain.fun == ir.SymRef(id="domain")
        assert all(
            isinstance(r, ir.FunCall) and r.fun == ir.SymRef(id="named_range")
            for r in node.domain.args
        )
        domain_ranges = [self.visit(r.args) for r in node.domain.args]
        axes = ", ".join(label for label, _, _ in domain_ranges)
        origin = "{" + ", ".join(f"{label}: -{start}" for label, start, _ in domain_ranges) + "}"
        shape = "(" + ", ".join(f"{stop}-{start}" for _, start, stop in domain_ranges) + ")"
        dtype = np_dtype(node.dtype)
        return f"{node.id} = np_as_located_field({axes}, origin={origin})(np.empty({shape}, dtype={dtype}))"


_BACKEND_NAME = "roundtrip"


def executor(ir: Node, *args, **kwargs):
    debug = "debug" in kwargs and kwargs["debug"] is True
    lift_mode = kwargs.get("lift_mode")

    ir = apply_common_transforms(ir, lift_mode=lift_mode, offset_provider=kwargs["offset_provider"])

    program = EmbeddedDSL.apply(ir)
    offset_literals: Iterable[str] = (
        ir.iter_tree().if_isinstance(OffsetLiteral).getattr("value").if_isinstance(str).to_set()
    )
    axis_literals: Iterable[str] = (
        ir.iter_tree().if_isinstance(AxisLiteral).getattr("value").to_set()
    )

    header = """
import numpy as np
from functional.iterator.builtins import *
from functional.iterator.runtime import *
from functional.iterator.embedded import np_as_located_field
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as source_file:
        source_file_name = source_file.name
        if debug:
            print(source_file_name)
        offset_literals = [f'{o} = offset("{o}")' for o in offset_literals]
        axis_literals = [f'{o} = CartesianAxis("{o}")' for o in axis_literals]
        source_file.write(header)
        source_file.write("\n".join(offset_literals))
        source_file.write("\n")
        source_file.write("\n".join(axis_literals))
        source_file.write("\n")
        source_file.write(program)

    try:
        spec = importlib.util.spec_from_file_location("module.name", source_file_name)
        foo = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(foo)  # type: ignore
    finally:
        if not debug:
            pathlib.Path(source_file_name).unlink(missing_ok=True)

    fencil_name = ir.fencil.id + "_wrapper" if isinstance(ir, FencilWithTemporaries) else ir.id
    fencil = getattr(foo, fencil_name)
    assert "offset_provider" in kwargs

    new_kwargs = {}
    new_kwargs["offset_provider"] = kwargs["offset_provider"]
    if "column_axis" in kwargs:
        new_kwargs["column_axis"] = kwargs["column_axis"]

    if "dispatch_backend" not in kwargs:
        iterator.builtins.builtin_dispatch.push_key("embedded")
        fencil(*args, **new_kwargs)
        iterator.builtins.builtin_dispatch.pop_key()
    else:
        fencil(
            *args,
            **new_kwargs,
            backend=kwargs["dispatch_backend"],
        )


backend.register_backend(_BACKEND_NAME, executor)
