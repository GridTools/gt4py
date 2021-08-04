from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from eve.concepts import Node
from iterator.ir import AxisLiteral, NoneLiteral, OffsetLiteral
from iterator.backends import backend
import tempfile
import importlib.util
import iterator


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
    Lambda = as_mako("lambda ${','.join(params)}: ${expr}")
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


_BACKEND_NAME = "embedded"


def executor(ir: Node, *args, **kwargs):
    debug = "debug" in kwargs and kwargs["debug"] == True

    program = EmbeddedDSL.apply(ir)
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
from iterator.builtins import *
from iterator.runtime import *
"""
        offset_literals = [f'{l} = offset("{l}")' for l in offset_literals]
        axis_literals = [f'{l} = CartesianAxis("{l}")' for l in axis_literals]
        tmp.write(header)
        tmp.write("\n".join(offset_literals))
        tmp.write("\n")
        tmp.write("\n".join(axis_literals))
        tmp.write("\n")
        tmp.write(program)
        tmp.flush()

        spec = importlib.util.spec_from_file_location("module.name", tmp.name)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        fencil_name = ir.fencil_definitions[0].id
        fencil = getattr(foo, fencil_name)
        assert "offset_provider" in kwargs

        new_kwargs = {}
        new_kwargs["offset_provider"] = kwargs["offset_provider"]
        if "column_axis" in kwargs:
            new_kwargs["column_axis"] = kwargs["column_axis"]

        if not "dispatch_backend" in kwargs:
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
