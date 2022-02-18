# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from eve import NodeTranslator, SymbolTableTrait
from functional.common import GTTypeError
from functional.ffront import common_types
from functional.ffront import program_ast as past
from functional.iterator import ir as itir


def _size_arg_from_field(field_name: str, dim: int):
    return f"__{field_name}_size_{dim}"


class ProgramLowering(NodeTranslator):
    contexts = (SymbolTableTrait.symtable_merger,)

    def visit_Program(self, node: past.Program, **kwargs):
        symtable = kwargs["symtable"]

        # The ITIR does not support dynamically getting the size of a field. As a workaround we add additional
        #  arguments to the fencil definition containing the size of all fields. The caller of a program is
        #  (e.g. program decorator) is required to pass these arguments.
        size_params = [
            itir.Sym(id=_size_arg_from_field(param.id, dim_idx))
            for param in node.params
            if isinstance(param.type, common_types.FieldType)
            for dim_idx in range(0, len(param.type.dims))
        ]

        closures: list[itir.StencilClosure] = []
        for stmt in node.body:
            if isinstance(stmt, past.Call) and isinstance(
                symtable[stmt.func.id].type.returns, common_types.FieldType
            ):
                closures.append(self._visit_stencil_call(stmt, **kwargs))
            else:
                raise NotImplementedError(
                    "Only calls to functions returning a Field supported currently."
                )

        return itir.FencilDefinition(
            id=node.id,
            params=[itir.Sym(id=inp.id) for inp in node.params] + size_params,
            closures=closures,
        )

    def _visit_stencil_call(self, node: past.Call, **kwargs) -> itir.StencilClosure:
        assert isinstance(node.kwargs["out"].type, common_types.FieldType)

        outputs, domain = self._visit_stencil_call_out_arg(node.kwargs["out"], **kwargs)

        return itir.StencilClosure(
            domain=domain,
            stencil=itir.SymRef(id=node.func.id),
            inputs=[self.visit(arg, **kwargs) for arg in node.args],
            outputs=outputs,
        )

    def _visit_slice_index(self, idx: Union[None, past.Constant], *, dim_size: itir.Expr, **kwargs):
        if idx is None:
            return itir.IntLiteral(value=0)
        elif isinstance(idx, past.Constant):
            if idx.value < 0:
                return itir.FunCall(
                    fun=itir.SymRef("minus"), args=[dim_size, self.visit(idx, **kwargs)]
                )
            else:
                return self.visit(idx, **kwargs)
        raise RuntimeError("Expected `None` or `past.Constant` node.")

    def _visit_stencil_call_out_arg(
        self, node: past.Expr, **kwargs
    ) -> tuple[list[itir.SymRef], itir.FunCall]:
        # as the ITIR does not support slicing a field we have to do a deeper inspection of the PAST to emulate
        #  the behaviour
        if isinstance(node, past.Subscript):
            out_field_name: past.Name = node.value
            if isinstance(node.slice_, past.TupleExpr):
                out_field_slice_: list[past.Slice] = node.slice_.elts
            elif isinstance(node.slice_, past.Slice):
                out_field_slice_: list[past.Slice] = [node.slice_]
            else:
                raise RuntimeError(
                    "Unexpected `out` argument. Must be tuple of slices or slice expression."
                )
            if len(out_field_slice_) != len(node.type.dims):
                raise GTTypeError(
                    f"Too many indices for field {out_field_name}: field is {len(node.type.dims)}"
                    f"-dimensional, but {len(out_field_slice_)} were indexed."
                )
            domain_args = []
            for dim_i, (dim, slice_) in enumerate(zip(node.type.dims, out_field_slice_)):
                # an expression for the size of a dimension
                dim_size = itir.SymRef(id=_size_arg_from_field(out_field_name.id, dim_i))
                # lower bound
                if slice_.lower is None:
                    lower = itir.IntLiteral(value=0)
                elif isinstance(slice_.lower, past.Constant):
                    assert (
                        isinstance(slice_.lower.type, common_types.ScalarType)
                        and slice_.lower.type.kind == common_types.ScalarKind.INT64
                    )
                    if slice_.lower.value < 0:
                        lower = itir.FunCall(
                            fun=itir.SymRef("plus"),
                            args=[dim_size, self.visit(slice_.lower, **kwargs)],
                        )
                    else:
                        lower = self.visit(slice_.lower, **kwargs)
                else:
                    raise AssertionError()
                # upper bound
                if slice_.upper is None:
                    upper = dim_size
                elif isinstance(slice_.upper, past.Constant):
                    assert (
                        isinstance(slice_.upper.type, common_types.ScalarType)
                        and slice_.upper.type.kind == common_types.ScalarKind.INT64
                    )
                    if slice_.upper.value < 0:
                        upper = itir.FunCall(
                            fun=itir.SymRef(id="plus"),
                            args=[dim_size, self.visit(slice_.upper, **kwargs)],
                        )
                    else:
                        upper = self.visit(slice_.upper, **kwargs)
                else:
                    raise AssertionError()

                domain_args.append(
                    itir.FunCall(
                        fun=itir.SymRef(id="named_range"),
                        args=[itir.AxisLiteral(value=dim.value), lower, upper],
                    )
                )

        elif isinstance(node, past.Name):
            out_field_name = node
            domain_args = [
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=dim.value),
                        itir.IntLiteral(value=0),
                        # here we use the artificial size arguments added to the fencil
                        itir.SymRef(id=_size_arg_from_field(out_field_name.id, dim_idx)),
                    ],
                )
                for dim_idx, dim in enumerate(node.type.dims)
            ]
        else:
            raise RuntimeError(
                "Unexpected `out` argument. Must be a `past.Subscript` or `past.Name` node."
            )

        return [self.visit(out_field_name, **kwargs)], itir.FunCall(
            fun=itir.SymRef(id="domain"), args=domain_args
        )

    def visit_Constant(self, node: past.Constant, **kwargs):
        if isinstance(node.type, common_types.ScalarType) and node.type.shape is None:
            match node.type.kind:
                case common_types.ScalarKind.INT32 | common_types.ScalarKind.INT64:
                    return itir.IntLiteral(value=node.value)
                case common_types.ScalarKind.FLOAT32 | common_types.ScalarKind.FLOAT64:
                    return itir.FloatLiteral(value=node.value)
                case common_types.ScalarKind.BOOL:
                    return itir.BoolLiteral(value=node.value)
                case _:
                    raise NotImplementedError(
                        "Scalars of kind {node.type.kind} not supported currently."
                    )

        raise NotImplementedError("Only scalar literals supported currently.")

    def visit_Name(self, node: past.Name, **kwargs):
        return itir.Sym(id=node.id)

    @classmethod
    def apply(cls, node: past.Program) -> itir.FencilDefinition:
        return cls().visit(node)
