import importlib.util
import tempfile

import iterator
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import Node
from iterator.backends import backend
from iterator.ir import AxisLiteral, FencilDefinition, OffsetLiteral
from iterator.transforms import apply_common_transforms


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    BoolLiteral = as_fmt("{value}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    NoneLiteral = as_fmt("None")
    OffsetLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    StringLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("(lambda ${','.join(params)}: ${expr})")
    StencilClosure = as_mako(
        "closure(${domain}, ${stencil}, [${','.join(outputs)}], [${','.join(inputs)}])"
    )
    FencilDefinition = as_mako(
        """
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
    Program = as_fmt(
        """
{''.join(function_definitions)} {''.join(fencil_definitions)}"""
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
        body = "\n    ".join(body)
        return f"\ndef {node.id}_wrapper({','.join(non_tmp_params)}, **kwargs):\n    {body}\n"


_BACKEND_NAME = "embedded"


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
    offset_literals = (
        ir.iter_tree().if_isinstance(OffsetLiteral).getattr("value").if_isinstance(str).to_set()
    )
    axis_literals = ir.iter_tree().if_isinstance(AxisLiteral).getattr("value").to_set()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=not debug,
    ) as tmp:
        if debug:
            print(tmp.name)
        header = """
import numpy as np
from iterator.builtins import *
from iterator.runtime import *
from iterator.embedded import np_as_located_field
"""
        offset_literals = [f'{o} = offset("{o}")' for o in offset_literals]
        axis_literals = [f'{o} = CartesianAxis("{o}")' for o in axis_literals]
        tmp.write(header)
        tmp.write("\n".join(offset_literals))
        tmp.write("\n")
        tmp.write("\n".join(axis_literals))
        tmp.write("\n")
        tmp.write(program)
        tmp.write(wrapper)
        tmp.flush()

        spec = importlib.util.spec_from_file_location("module.name", tmp.name)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)  # type: ignore

        fencil_name = ir.fencil_definitions[0].id
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
