import importlib.util
import pathlib
import tempfile
from typing import Iterable

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from eve.concepts import Node
from functional import iterator
from functional.iterator.backends import backend
from functional.iterator.ir import AxisLiteral, FencilDefinition, OffsetLiteral
from functional.iterator.transforms import apply_common_transforms


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


# TODO this wrapper should be replaced by an extension of the IR
class WrapperGenerator(EmbeddedDSL):
    def visit_FencilDefinition(self, node: FencilDefinition, *, tmps):
        params = self.visit(node.params)
        non_tmp_params = [param for param in params if param not in tmps]

        body = []
        for tmp, domain in tmps.items():
            axis_literals = [named_range.args[0].value for named_range in domain.args]
            origin = (
                "{"
                + ", ".join(
                    f"{named_range.args[0].value}: -{self.visit(named_range.args[1])}"
                    for named_range in domain.args
                )
                + "}"
            )
            shape = (
                "("
                + ", ".join(
                    f"{self.visit(named_range.args[2])}-{self.visit(named_range.args[1])}"
                    for named_range in domain.args
                )
                + ")"
            )
            body.append(
                f"{tmp} = np_as_located_field({','.join(axis_literals)}, origin={origin})(np.full({shape}, np.nan))"
            )

        body.append(f"{node.id}({','.join(params)}, **kwargs)")
        body_str = "\n    ".join(body)
        return f"\ndef {node.id}_wrapper({','.join(non_tmp_params)}, **kwargs):\n    {body_str}\n"


_BACKEND_NAME = "roundtrip"


def executor(ir: Node, *args, **kwargs):
    debug = "debug" in kwargs and kwargs["debug"] is True
    use_tmps = "use_tmps" in kwargs and kwargs["use_tmps"] is True

    tmps = dict()

    def register_tmp(tmp, domain):
        tmps[tmp] = domain

    ir = apply_common_transforms(
        ir, use_tmps=use_tmps, offset_provider=kwargs["offset_provider"], register_tmp=register_tmp
    )

    program = EmbeddedDSL.apply(ir)
    wrapper = WrapperGenerator.apply(ir, tmps=tmps)
    offset_literals: Iterable[str] = (
        ir.pre_walk_values()
        .if_isinstance(OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_set()
    )
    axis_literals: Iterable[str] = (
        ir.pre_walk_values().if_isinstance(AxisLiteral).getattr("value").to_set()
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
        source_file.write(wrapper)

    try:
        spec = importlib.util.spec_from_file_location("module.name", source_file_name)
        foo = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(foo)  # type: ignore
    finally:
        if not debug:
            pathlib.Path(source_file_name).unlink(missing_ok=True)

    assert isinstance(ir, FencilDefinition)
    fencil_name = ir.id
    fencil = getattr(foo, fencil_name + "_wrapper")
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
