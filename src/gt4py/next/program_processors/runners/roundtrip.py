# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import tempfile
import textwrap
import types
from collections.abc import Iterable
from typing import Any, Optional

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import (
    backend as next_backend,
    common,
    config,
    custom_layout_allocators as next_allocators,
)
from gt4py.next.ffront import foast_to_gtir, foast_to_past, past_to_itir
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import definitions, stages, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


def _create_tmp(axes: str, origin: str, shape: str, dtype: ts.TypeSpec) -> str:
    if isinstance(dtype, ts.TupleType):
        return f"({','.join(_create_tmp(axes, origin, shape, dt) for dt in dtype.types)},)"
    else:
        assert isinstance(dtype, ts.ScalarType)
        return (
            f"gtx.as_field([{axes}], np.empty({shape}, dtype=np.dtype('{dtype}')), origin={origin})"
        )


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> str:
        if (
            isinstance(node.type, ts.ScalarType)
            and type_info.is_floating_point(node.type)
            and node.value in ["inf", "-inf", "nan"]
        ):
            dtype = node.type.kind.name.lower()
            if node.value == "inf":
                return f"np.{dtype}(np.inf)"
            elif node.value == "-inf":
                return f"-np.{dtype}(np.inf)"
            elif node.value == "nan":
                return f"np.{dtype}(np.nan)"
        return node.value

    NoneLiteral = as_fmt("None")
    OffsetLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("(lambda ${','.join(params)}: ${expr})")
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )
    Program = as_mako(
        """
${''.join(function_definitions)}
@fendef
def ${id}(${','.join(params)}):
    ${'\\n    '.join(declarations)}
    ${'\\n    '.join(body)}
    """
    )
    SetAt = as_mako("set_at(${expr}, ${domain}, ${target})")
    IfStmt = as_mako("""if_stmt(${cond}, 
        lambda: [${','.join(true_branch)}],
        lambda: [${','.join(false_branch)}]
    )""")

    def visit_Temporary(self, node: itir.Temporary, **kwargs: Any) -> str:
        assert (
            isinstance(node.domain, itir.FunCall)
            and isinstance(node.domain.fun, itir.SymRef)
            and node.domain.fun.id in ("cartesian_domain", "unstructured_domain")
        )
        assert all(
            isinstance(r, itir.FunCall) and r.fun == itir.SymRef(id="named_range")
            for r in node.domain.args
        )
        domain_ranges = [self.visit(r.args) for r in node.domain.args]  # type: ignore[attr-defined] # `node.domain` is `FunCall` checked in previous assert
        axes = ", ".join(label for label, _, _ in domain_ranges)
        origin = "{" + ", ".join(f"{label}: -{start}" for label, start, _ in domain_ranges) + "}"
        shape = "(" + ", ".join(f"{stop}-{start}" for _, start, stop in domain_ranges) + ")"
        assert node.dtype
        return f"{node.id} = {_create_tmp(axes, origin, shape, node.dtype)}"


# Caches the generated source by IR hash so re-codegen is skipped within a process.
_SOURCE_CACHE: dict[int, tuple[str, str]] = {}
# Caches the loaded module by source string so re-exec is skipped within a process.
_MODULE_CACHE: dict[str, types.ModuleType] = {}


def _generate_source(
    ir: itir.Program,
    debug: bool,
    use_embedded: bool,
    offset_provider: common.OffsetProvider,
    transforms: itir_transforms.GTIRTransform,
) -> tuple[str, str]:
    """Generate the Python source for an ITIR program. Returns ``(source_code, entry_point_name)``."""
    # TODO(tehrengruber): just a temporary solution until we have a proper generic
    #  caching mechanism
    cache_key = hash(
        (
            ir,
            transforms,
            debug,
            use_embedded,
            tuple(common.offset_provider_to_type(offset_provider).items()),
        )
    )
    if cache_key in _SOURCE_CACHE:
        if debug:
            print(f"Using cached source for key {cache_key}")
        return _SOURCE_CACHE[cache_key]

    ir = transforms(ir, offset_provider=offset_provider)

    program = EmbeddedDSL.apply(ir)

    # format output in debug mode for better debuggability
    # (e.g. line numbers, overview in the debugger).
    if debug:
        program = codegen.format_python_source(program)

    offset_literals: Iterable[str] = (
        ir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_set()
    )
    axis_literals_set: Iterable[itir.AxisLiteral] = (
        ir.pre_walk_values().if_isinstance(itir.AxisLiteral).to_set()
    )

    if use_embedded:
        builtins_import = "from gt4py.next.iterator.embedded import *"
    else:
        builtins_import = "from gt4py.next.iterator.builtins import *"

    header = textwrap.dedent(
        f"""
        import numpy as np
        import gt4py.next as gtx
        {builtins_import}
        from gt4py.next.iterator.runtime import *
        """
    )

    offset_literals_src = "\n".join(f'{o} = offset("{o}")' for o in offset_literals)
    axis_literals_src = "\n".join(
        f'{o.value} = gtx.Dimension("{o.value}", kind=gtx.DimensionKind("{o.kind}"))'
        for o in axis_literals_set
    )
    source_code = f"{header}{offset_literals_src}\n{axis_literals_src}\n{program}"

    assert isinstance(ir, itir.Program)
    entry_point_name = ir.id

    _SOURCE_CACHE[cache_key] = (source_code, entry_point_name)
    return source_code, entry_point_name


def _load_module(source_code: str, debug: bool) -> types.ModuleType:
    if source_code in _MODULE_CACHE:
        return _MODULE_CACHE[source_code]

    if debug:
        # Write to a real .py so debuggers/tracebacks have file/line info.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8", delete=False
        ) as source_file:
            source_file.write(source_code)
            source_file_name = source_file.name
        print(source_file_name)
        spec = importlib.util.spec_from_file_location("module.name", source_file_name)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = types.ModuleType("roundtrip_module")
        exec(compile(source_code, "<roundtrip>", "exec"), mod.__dict__)

    _MODULE_CACHE[source_code] = mod
    return mod


@dataclasses.dataclass(frozen=True)
class RoundtripArtifact:
    """Source-string artifact for the roundtrip backend.

    The generated Python source is the artifact: picklable, re-execed on
    :meth:`load`. When ``debug`` is true, ``load`` writes a temporary ``.py``
    so debuggers/tracebacks resolve to source lines.
    """

    source_code: str
    entry_point_name: str
    column_axis: common.Dimension | None
    dispatch_backend: next_backend.Backend | None
    debug: bool

    def load(self) -> stages.ExecutableProgram:
        mod = _load_module(self.source_code, self.debug)
        fencil = getattr(mod, self.entry_point_name)
        captured_column_axis = self.column_axis
        dispatch_backend = self.dispatch_backend

        def decorated_fencil(
            *args: Any,
            offset_provider: dict[str, common.Connectivity | common.Dimension],
            out: Any = None,
            column_axis: Optional[
                common.Dimension
            ] = None,  # TODO(tehrengruber): unused, kept for signature compat
            **kwargs: Any,
        ) -> None:
            if out is not None:
                args = (*args, out)
            fencil(
                *args,
                offset_provider=offset_provider,
                backend=dispatch_backend,
                column_axis=captured_column_axis,
                **kwargs,
            )

        return decorated_fencil


@dataclasses.dataclass(frozen=True)
class Roundtrip(workflow.Workflow[definitions.CompilableProgramDef, RoundtripArtifact]):
    debug: Optional[bool] = None
    use_embedded: bool = True
    dispatch_backend: Optional[next_backend.Backend] = None
    transforms: itir_transforms.GTIRTransform = itir_transforms.apply_common_transforms  # type: ignore[assignment] # TODO(havogt): cleanup interface of `apply_common_transforms`

    def __call__(self, inp: definitions.CompilableProgramDef) -> RoundtripArtifact:
        debug = config.DEBUG if self.debug is None else self.debug

        source_code, entry_point_name = _generate_source(
            inp.data,
            offset_provider=inp.args.offset_provider,
            debug=debug,
            use_embedded=self.use_embedded,
            transforms=self.transforms,
        )

        return RoundtripArtifact(
            source_code=source_code,
            entry_point_name=entry_point_name,
            column_axis=inp.args.column_axis,
            dispatch_backend=self.dispatch_backend,
            debug=debug,
        )


# TODO(tehrengruber): introduce factory
default = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=False,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
with_temporaries = next_backend.Backend(
    name="roundtrip_with_temporaries",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=True,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
no_transforms = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(transforms=lambda o, *, offset_provider: o),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)


gtir = next_backend.Backend(
    name="roundtrip_gtir",
    executor=Roundtrip(transforms=itir_transforms.apply_fieldview_transforms),  # type: ignore[arg-type] # don't understand why mypy complains
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.Transforms(
        past_to_itir=past_to_itir.past_to_gtir_factory(),
        foast_to_itir=foast_to_gtir.adapted_foast_to_gtir_factory(cached=True),
        field_view_op_to_prog=foast_to_past.operator_to_program_factory(
            foast_to_itir_step=foast_to_gtir.adapted_foast_to_gtir_factory()
        ),
    ),
)
foast_to_gtir_step = foast_to_gtir.adapted_foast_to_gtir_factory(cached=True)
