# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.eve import utils as eve_utils
from gt4py.next.ffront import (
    dialect_ast_enums,
    program_ast as past,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.past_passes import closure_var_type_deduction, type_deduction
from gt4py.next.otf import workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


def foast_to_past(inp: ffront_stages.FoastWithTypes) -> ffront_stages.PastProgramDefinition:
    # TODO(tehrengruber): implement mechanism to deduce default values
    #  of arg and kwarg types
    # TODO(tehrengruber): check foast operator has no out argument that clashes
    #  with the out argument of the program we generate here.

    loc = inp.foast_op_def.foast_node.location
    # use a new UID generator to allow caching
    param_sym_uids = eve_utils.UIDGenerator()

    type_ = inp.foast_op_def.foast_node.type
    params_decl: list[past.Symbol] = [
        past.DataSymbol(
            id=param_sym_uids.sequential_id(prefix="__sym"),
            type=arg_type,
            namespace=dialect_ast_enums.Namespace.LOCAL,
            location=loc,
        )
        for arg_type in inp.arg_types
    ]
    params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
    out_sym: past.Symbol = past.DataSymbol(
        id="out",
        type=type_info.return_type(
            type_, with_args=list(inp.arg_types), with_kwargs=inp.kwarg_types
        ),
        namespace=dialect_ast_enums.Namespace.LOCAL,
        location=loc,
    )
    out_ref = past.Name(id="out", location=loc)

    if inp.foast_op_def.foast_node.id in inp.foast_op_def.closure_vars:
        raise RuntimeError("A closure variable has the same name as the field operator itself.")
    closure_symbols: list[past.Symbol] = [
        past.Symbol(
            id=inp.foast_op_def.foast_node.id,
            type=ts.DeferredType(constraint=None),
            namespace=dialect_ast_enums.Namespace.CLOSURE,
            location=loc,
        ),
    ]

    untyped_past_node = past.Program(
        id=f"__field_operator_{inp.foast_op_def.foast_node.id}",
        type=ts.DeferredType(constraint=ts_ffront.ProgramType),
        params=[*params_decl, out_sym],
        body=[
            past.Call(
                func=past.Name(id=inp.foast_op_def.foast_node.id, location=loc),
                args=params_ref,
                kwargs={"out": out_ref},
                location=loc,
            )
        ],
        closure_vars=closure_symbols,
        location=loc,
    )
    untyped_past_node = closure_var_type_deduction.ClosureVarTypeDeduction.apply(
        untyped_past_node, inp.closure_vars
    )
    past_node = type_deduction.ProgramTypeDeduction.apply(untyped_past_node)

    return ffront_stages.PastProgramDefinition(
        past_node=past_node,
        closure_vars=inp.closure_vars,
        grid_type=inp.foast_op_def.grid_type,
    )


@dataclasses.dataclass(frozen=True)
class FoastToPastClosure(workflow.NamedStepSequence):
    foast_to_past: workflow.Workflow[
        ffront_stages.FoastWithTypes, ffront_stages.PastProgramDefinition
    ]

    def __call__(self, inp: ffront_stages.FoastClosure) -> ffront_stages.PastClosure:
        # TODO(tehrengruber): check all offset providers are given
        # deduce argument types
        arg_types = []
        for arg in inp.args:
            arg_types.append(type_translation.from_value(arg))
        kwarg_types = {}
        for name, arg in inp.kwargs.items():
            kwarg_types[name] = type_translation.from_value(arg)

        past_def = super().__call__(
            ffront_stages.FoastWithTypes(
                foast_op_def=inp.foast_op_def,
                arg_types=tuple(arg_types),
                kwarg_types=kwarg_types,
                closure_vars=inp.closure_vars,
            )
        )

        return ffront_stages.PastClosure(
            past_node=past_def.past_node,
            closure_vars=past_def.closure_vars,
            grid_type=past_def.grid_type,
            args=inp.args,
            kwargs=inp.kwargs,
        )
