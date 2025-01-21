# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Optional

from gt4py.eve import utils as eve_utils
from gt4py.next.ffront import (
    dialect_ast_enums,
    foast_to_gtir,
    program_ast as past,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.past_passes import closure_var_type_deduction, type_deduction
from gt4py.next.ffront.stages import AOT_FOP, AOT_PRG
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


@dataclasses.dataclass(frozen=True)
class ItirShim:
    """
    A wrapper for a FOAST operator definition with `__gt_*__` special methods.

    Can be placed in a PAST program definition's closure variables so the program
    lowering has access to the relevant information.
    """

    definition: AOT_FOP
    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr]

    def __gt_closure_vars__(self) -> Optional[dict[str, Any]]:
        return self.definition.data.closure_vars

    def __gt_type__(self) -> ts.CallableType:
        return self.definition.data.foast_node.type

    def __gt_itir__(self) -> itir.Expr:
        return self.foast_to_itir(self.definition)

    # FIXME[#1582](tehrengruber): remove after refactoring to GTIR
    def __gt_gtir__(self) -> itir.Expr:
        # backend should have self.foast_to_itir set to foast_to_gtir
        return self.foast_to_itir(self.definition)


@dataclasses.dataclass(frozen=True)
class OperatorToProgram(workflow.Workflow[AOT_FOP, AOT_PRG]):
    """
    Generate a PAST program definition from a FOAST operator definition.

    This workflow step must must be given a FOAST -> ITIR lowering step so that it can place
    valid `ItirShim` instances into the closure variables of the generated program.

    Example:
        >>> from gt4py import next as gtx
        >>> from gt4py.next.otf import arguments, toolchain
        >>> IDim = gtx.Dimension("I")

        >>> @gtx.field_operator
        ... def copy(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a

        >>> op_to_prog = OperatorToProgram(foast_to_gtir.adapted_foast_to_gtir_factory())

        >>> compile_time_args = arguments.CompileTimeArgs(
        ...     args=tuple(param.type for param in copy.foast_stage.foast_node.definition.params),
        ...     kwargs={},
        ...     offset_provider={"I", IDim},
        ...     column_axis=None,
        ... )

        >>> copy_program = op_to_prog(toolchain.CompilableProgram(copy.foast_stage, compile_time_args))

        >>> print(copy_program.data.past_node.id)
        __field_operator_copy

        >>> assert copy_program.data.closure_vars["copy"].definition.data is copy.foast_stage
    """

    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr]

    def __call__(self, inp: AOT_FOP) -> AOT_PRG:
        # TODO(tehrengruber): implement mechanism to deduce default values
        #  of arg and kwarg types
        # TODO(tehrengruber): check foast operator has no out argument that clashes
        #  with the out argument of the program we generate here.

        arg_types = inp.args.args
        kwarg_types = inp.args.kwargs

        loc = inp.data.foast_node.location
        # use a new UID generator to allow caching
        param_sym_uids = eve_utils.UIDGenerator()

        type_ = inp.data.foast_node.type
        params_decl: list[past.Symbol] = [
            past.DataSymbol(
                id=param_sym_uids.sequential_id(prefix="__sym"),
                type=arg_type,
                namespace=dialect_ast_enums.Namespace.LOCAL,
                location=loc,
            )
            for arg_type in arg_types
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym: past.Symbol = past.DataSymbol(
            id="out",
            type=type_info.return_type(type_, with_args=list(arg_types), with_kwargs=kwarg_types),
            namespace=dialect_ast_enums.Namespace.LOCAL,
            location=loc,
        )
        out_ref = past.Name(id="out", location=loc)

        if inp.data.foast_node.id in inp.data.closure_vars:
            raise RuntimeError("A closure variable has the same name as the field operator itself.")

        closure_symbols: list[past.Symbol] = [
            past.Symbol(
                id=inp.data.foast_node.id,
                type=ts.DeferredType(constraint=None),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=loc,
            ),
        ]

        fieldop_itir_closure_vars = {inp.data.foast_node.id: ItirShim(inp, self.foast_to_itir)}

        untyped_past_node = past.Program(
            id=f"__field_operator_{inp.data.foast_node.id}",
            type=ts.DeferredType(constraint=ts_ffront.ProgramType),
            params=[*params_decl, out_sym],
            body=[
                past.Call(
                    func=past.Name(id=inp.data.foast_node.id, location=loc),
                    args=params_ref,
                    kwargs={"out": out_ref},
                    location=loc,
                )
            ],
            closure_vars=closure_symbols,
            location=loc,
        )
        untyped_past_node = closure_var_type_deduction.ClosureVarTypeDeduction.apply(
            untyped_past_node, fieldop_itir_closure_vars
        )
        past_node = type_deduction.ProgramTypeDeduction.apply(untyped_past_node)

        return toolchain.CompilableProgram(
            data=ffront_stages.PastProgramDefinition(
                past_node=past_node,
                closure_vars=fieldop_itir_closure_vars,
                grid_type=inp.data.grid_type,
            ),
            args=inp.args,
        )


def operator_to_program_factory(
    foast_to_itir_step: Optional[workflow.Workflow[AOT_FOP, itir.Expr]] = None,
    cached: bool = True,
) -> workflow.Workflow[AOT_FOP, AOT_PRG]:
    """Optionally wrap `OperatorToProgram` in a `CachedStep`."""
    wf: workflow.Workflow[AOT_FOP, AOT_PRG] = OperatorToProgram(
        foast_to_itir_step or foast_to_gtir.adapted_foast_to_gtir_factory()
    )
    if cached:
        wf = workflow.CachedStep(wf, hash_function=ffront_stages.fingerprint_stage)
    return wf
