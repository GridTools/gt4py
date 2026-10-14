# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Optional

from gt4py.next.ffront import (
    dialect_ast_enums,
    foast_to_gtir,
    past_to_itir,
    program_ast as past,
    stages as ffront_stages,
    type_info as ffront_type_info,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.past_passes import closure_var_type_deduction, type_deduction
from gt4py.next.ffront.stages import ConcreteFOASTOperatorDef, ConcretePASTProgramDef
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import workflow
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class ItirShim:
    """
    A wrapper for a FOAST operator definition with `__gt_*__` special methods.

    Can be placed in a PAST program definition's closure variables so the program
    lowering has access to the relevant information.
    """

    definition: ConcreteFOASTOperatorDef
    foast_to_itir: workflow.Workflow[ConcreteFOASTOperatorDef, itir.FunctionDefinition]

    def __gt_closure_vars__(self) -> Optional[dict[str, Any]]:
        return self.definition.data.closure_vars

    def __gt_type__(self) -> ts.CallableType:
        assert isinstance(self.definition.data.foast_node.type, ts.CallableType)
        return self.definition.data.foast_node.type

    def __gt_itir__(self) -> itir.FunctionDefinition:
        return self.foast_to_itir(self.definition)

    # FIXME[#1582](tehrengruber): remove after refactoring to GTIR
    def __gt_gtir__(self) -> itir.FunctionDefinition:
        # backend should have self.foast_to_itir set to foast_to_gtir
        return self.foast_to_itir(self.definition)


@dataclasses.dataclass(frozen=True)
class OperatorToProgram(workflow.Workflow[ConcreteFOASTOperatorDef, ConcretePASTProgramDef]):
    """
    Generate a PAST program definition from a FOAST operator definition.

    This workflow step must must be given a FOAST -> ITIR lowering step so that it can place
    valid `ItirShim` instances into the closure variables of the generated program.

    Example:
        >>> from gt4py import next as gtx
        >>> from gt4py.next.otf import arguments, workflow
        >>> IDim = gtx.Dimension("I")

        >>> @gtx.field_operator
        ... def copy(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a

        >>> op_to_prog = OperatorToProgram(foast_to_gtir.adapted_foast_to_gtir_factory())

        >>> compile_time_args = arguments.CompileTimeArgs(
        ...     args=(
        ...         *(param.type for param in copy.foast_stage.foast_node.definition.params),
        ...         copy.foast_stage.foast_node.definition.type.returns,
        ...     ),
        ...     kwargs={},
        ...     offset_provider={},
        ...     column_axis=None,
        ...     argument_descriptor_contexts={},
        ... )

        >>> copy_program = op_to_prog(
        ...     workflow.ConcreteArtifact(copy.foast_stage, compile_time_args)
        ... )

        >>> print(copy_program.data.past_node.id)
        __field_operator_copy

        >>> assert copy_program.data.closure_vars["copy"].definition.data is copy.foast_stage
    """

    foast_to_itir: workflow.Workflow[ConcreteFOASTOperatorDef, itir.FunctionDefinition]

    def __call__(self, inp: ConcreteFOASTOperatorDef) -> ConcretePASTProgramDef:
        # TODO(tehrengruber): implement mechanism to deduce default values
        #  of arg and kwarg types
        # TODO(tehrengruber): check foast operator has no out argument that clashes
        #  with the out argument of the program we generate here.
        loc = inp.data.foast_node.location
        assert isinstance(inp.data.foast_node.type, ts.CallableType)
        partial_program_type = ffront_type_info.type_in_program_context(inp.data.foast_node.type)
        assert isinstance(partial_program_type, ts_ffront.ProgramType)
        param_types: tuple[tuple[str, ts.TypeSpec], ...] = (
            *(
                (f"__pos_arg{i}", type_)
                for i, type_ in enumerate(partial_program_type.definition.pos_only_args)
            ),
            *partial_program_type.definition.pos_or_kw_args.items(),
            *partial_program_type.definition.kw_only_args.items(),
        )
        assert param_types[-1][0] == "out"

        params_decl: list[past.Symbol] = [
            past.DataSymbol(
                id=name,
                type=type_,  # note: this might be a generic type like VarLenTupleType
                namespace=dialect_ast_enums.Namespace.LOCAL,
                location=loc,
            )
            for name, type_ in param_types
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl[:-1]]
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
            params=params_decl,
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
        # Specialize the parameter types to the concrete argument types of the call before
        # the type deduction: the declared types may be generic — e.g. `DeferredType` for
        # scan operator parameters, which are only promoted to fields at the call — and the
        # body call can not be type checked against them. The specialization happens again
        # later, in `past_to_gtir`, which also covers hand-written programs that are not
        # specialized here. With support for generic types in the program type deduction
        # this step would become obsolete, as the generic variant could be type checked
        # directly.
        untyped_past_node = past_to_itir._specialize_program_params(
            untyped_past_node,
            inp.args.args,
            inp.args.kwargs,
            # argument validation has not run yet on this path; incompatible arguments are
            # rejected right below, by the type deduction of the body call
            check_compatibility=False,
        )
        untyped_past_node = closure_var_type_deduction.ClosureVarTypeDeduction.apply(
            untyped_past_node,
            fieldop_itir_closure_vars,  # type: ignore[arg-type]
        )
        past_node = type_deduction.ProgramTypeDeduction.apply(untyped_past_node)

        return workflow.ConcreteArtifact(
            data=ffront_stages.PASTProgramDef(
                past_node=past_node,
                closure_vars=fieldop_itir_closure_vars,  # type: ignore[arg-type]
                grid_type=inp.data.grid_type,
            ),
            args=inp.args,
        )


def operator_to_program_factory(
    foast_to_itir_step: Optional[
        workflow.Workflow[ConcreteFOASTOperatorDef, itir.FunctionDefinition]
    ] = None,
    cached: bool = True,
) -> workflow.Workflow[ConcreteFOASTOperatorDef, ConcretePASTProgramDef]:
    """Optionally wrap `OperatorToProgram` in a `CachedStep`."""
    wf: workflow.Workflow[ConcreteFOASTOperatorDef, ConcretePASTProgramDef] = OperatorToProgram(
        foast_to_itir_step or foast_to_gtir.adapted_foast_to_gtir_factory()
    )
    if cached:
        wf = workflow.CachedStep.in_memory(
            wf, input_fingerprinter=ffront_stages.semantic_fingerprinter
        )
    return wf
