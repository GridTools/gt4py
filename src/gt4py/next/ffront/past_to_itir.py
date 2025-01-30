# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Optional, cast

import devtools

from gt4py.eve import NodeTranslator, concepts, traits
from gt4py.next import common, config, errors
from gt4py.next.ffront import (
    fbuiltins,
    gtcallable,
    program_ast as past,
    stages as ffront_stages,
    transform_utils,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.stages import AOT_PRG
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import stages, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


# FIXME[#1582](tehrengruber): This should only depend on the program not the arguments. Remove
#  dependency as soon as column axis can be deduced from ITIR in consumers of the CompilableProgram.
def past_to_gtir(inp: AOT_PRG) -> stages.CompilableProgram:
    """
    Lower a PAST program definition to Iterator IR.

    Example:
        >>> from gt4py import next as gtx
        >>> from gt4py.next.otf import arguments, toolchain
        >>> IDim = gtx.Dimension("I")

        >>> @gtx.field_operator
        ... def copy(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a

        >>> @gtx.program
        ... def copy_program(a: gtx.Field[[IDim], gtx.float32], out: gtx.Field[[IDim], gtx.float32]):
        ...     copy(a, out=out)

        >>> compile_time_args = arguments.CompileTimeArgs(
        ...     args=tuple(param.type for param in copy_program.past_stage.past_node.params),
        ...     kwargs={},
        ...     offset_provider={"I": IDim},
        ...     column_axis=None,
        ... )

        >>> itir_copy = past_to_gtir(
        ...     toolchain.CompilableProgram(copy_program.past_stage, compile_time_args)
        ... )

        >>> print(itir_copy.data.id)
        copy_program

        >>> print(type(itir_copy.data))
        <class 'gt4py.next.iterator.ir.Program'>
    """
    all_closure_vars = transform_utils._get_closure_vars_recursively(inp.data.closure_vars)
    offsets_and_dimensions = transform_utils._filter_closure_vars_by_type(
        all_closure_vars, fbuiltins.FieldOffset, common.Dimension
    )
    grid_type = transform_utils._deduce_grid_type(
        inp.data.grid_type, offsets_and_dimensions.values()
    )

    gt_callables = transform_utils._filter_closure_vars_by_type(
        all_closure_vars, gtcallable.GTCallable
    ).values()

    # FIXME[#1582](tehrengruber): remove after refactoring to GTIR
    # TODO(ricoh): The following calls to .__gt_itir__, which will use whatever
    #  backend is set for each of these field operators (GTCallables). Instead
    #  we should use the current toolchain to lower these to ITIR. This will require
    #  making this step aware of the toolchain it is called by (it can be part of multiple).
    lowered_funcs = []
    for gt_callable in gt_callables:
        lowered_funcs.append(gt_callable.__gt_gtir__())

    itir_program = ProgramLowering.apply(
        inp.data.past_node, function_definitions=lowered_funcs, grid_type=grid_type
    )

    if config.DEBUG or inp.data.debug:
        devtools.debug(itir_program)

    return stages.CompilableProgram(
        data=itir_program,
        args=dataclasses.replace(inp.args, column_axis=_column_axis(all_closure_vars)),
    )


def past_to_gtir_factory(
    cached: bool = True,
) -> workflow.Workflow[AOT_PRG, stages.CompilableProgram]:
    wf = workflow.make_step(past_to_gtir)
    if cached:
        wf = workflow.CachedStep(wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def _column_axis(all_closure_vars: dict[str, Any]) -> Optional[common.Dimension]:
    # construct mapping from column axis to scan operators defined on
    #  that dimension. only one column axis is allowed, but we can use
    #  this mapping to provide good error messages.
    scanops_per_axis: dict[common.Dimension, list[str]] = {}
    for name, gt_callable in transform_utils._filter_closure_vars_by_type(
        all_closure_vars, gtcallable.GTCallable
    ).items():
        if isinstance((type_ := gt_callable.__gt_type__()), ts_ffront.ScanOperatorType):
            scanops_per_axis.setdefault(type_.axis, []).append(name)

    if len(scanops_per_axis.values()) == 0:
        return None

    if len(scanops_per_axis.values()) != 1:
        scanops_per_axis_strs = [
            f"- {dim.value}: {', '.join(scanops)}" for dim, scanops in scanops_per_axis.items()
        ]

        raise TypeError(
            "Only 'ScanOperator's defined on the same axis "
            + "can be used in a 'Program', found:\n"
            + "\n".join(scanops_per_axis_strs)
            + "."
        )

    return iter(scanops_per_axis.keys()).__next__()


def _range_arg_from_field(field_name: str, dim: int) -> str:
    return f"__{field_name}_{dim}_range"


def _flatten_tuple_expr(node: past.Expr) -> list[past.Name | past.Subscript]:
    if isinstance(node, (past.Name, past.Subscript)):
        return [node]
    elif isinstance(node, past.TupleExpr):
        result = []
        for e in node.elts:
            result.extend(_flatten_tuple_expr(e))
        return result
    raise ValueError("Only 'past.Name', 'past.Subscript' or 'past.TupleExpr' thereof are allowed.")


@dataclasses.dataclass
class ProgramLowering(
    traits.PreserveLocationVisitor, traits.VisitorWithSymbolTableTrait, NodeTranslator
):
    """
    Lower Program AST (PAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from gt4py.next.ffront.func_to_past import ProgramParser
    >>> from gt4py.next.iterator import ir
    >>> from gt4py.next import Dimension, Field
    >>>
    >>> float64 = float
    >>> IDim = Dimension("IDim")
    >>>
    >>> def fieldop(inp: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]: ...
    >>> def program(inp: Field[[IDim], "float64"], out: Field[[IDim], "float64"]):
    ...     fieldop(inp, out=out)
    >>>
    >>> parsed = ProgramParser.apply_to_function(program)  # doctest: +SKIP
    >>> fieldop_def = ir.FunctionDefinition(
    ...     id="fieldop",
    ...     params=[ir.Sym(id="inp")],
    ...     expr=ir.FunCall(fun=ir.SymRef(id="deref"), pos_only_args=[ir.SymRef(id="inp")]),
    ... )  # doctest: +SKIP
    >>> lowered = ProgramLowering.apply(
    ...     parsed, [fieldop_def], grid_type=common.GridType.CARTESIAN
    ... )  # doctest: +SKIP
    >>> type(lowered)  # doctest: +SKIP
    <class 'gt4py.next.iterator.ir.Program'>
    >>> lowered.id  # doctest: +SKIP
    SymbolName('program')
    >>> lowered.params  # doctest: +SKIP
    [Sym(id=SymbolName('inp')), Sym(id=SymbolName('out')), Sym(id=SymbolName('__inp_size_0')), Sym(id=SymbolName('__out_size_0'))]
    """

    grid_type: common.GridType

    # TODO(tehrengruber): enable doctests again. For unknown / obscure reasons
    #  the above doctest fails when executed using `pytest --doctest-modules`.

    @classmethod
    def apply(
        cls,
        node: past.Program,
        function_definitions: list[itir.FunctionDefinition],
        grid_type: common.GridType,
    ) -> itir.Program:
        return cls(grid_type=grid_type).visit(node, function_definitions=function_definitions)

    def _gen_size_params_from_program(self, node: past.Program) -> list[itir.Sym]:
        """Generate symbols for each field param and dimension."""
        size_params = []
        for param in node.params:
            fields_dims: list[list[common.Dimension]] = (
                type_info.primitive_constituents(param.type)
                .if_isinstance(ts.FieldType)
                .getattr("dims")
                .filter(lambda dims: len(dims) > 0)
                .to_list()
            )
            if len(fields_dims) > 0:  # otherwise `param` has no constituent which is of `FieldType`
                assert all(field_dims == fields_dims[0] for field_dims in fields_dims)
                index_type = ts.ScalarType(
                    kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper())
                )
                for dim_idx in range(len(fields_dims[0])):
                    size_params.append(
                        itir.Sym(
                            id=_range_arg_from_field(param.id, dim_idx),
                            type=ts.TupleType(types=[index_type, index_type]),
                        )
                    )

        return size_params

    def visit_Program(
        self,
        node: past.Program,
        *,
        function_definitions: list[itir.FunctionDefinition],
        **kwargs: Any,
    ) -> itir.Program:
        # The ITIR does not support dynamically getting the size of a field. As
        #  a workaround we add additional arguments to the fencil definition
        #  containing the size of all fields. The caller of a program is (e.g.
        #  program decorator) is required to pass these arguments.

        params = self.visit(node.params)

        implicit_domain = False
        if any("domain" not in body_entry.kwargs for body_entry in node.body):
            params = params + self._gen_size_params_from_program(node)
            implicit_domain = True

        set_ats = [self._visit_field_operator_call(stmt, **kwargs) for stmt in node.body]
        return itir.Program(
            id=node.id,
            function_definitions=function_definitions,
            params=params,
            declarations=[],
            body=set_ats,
            implicit_domain=implicit_domain,
        )

    def _visit_field_operator_call(self, node: past.Call, **kwargs: Any) -> itir.SetAt:
        assert isinstance(node.kwargs["out"].type, ts.TypeSpec)
        assert type_info.is_type_or_tuple_of_type(node.kwargs["out"].type, ts.FieldType)

        node_kwargs = {**node.kwargs}
        domain = node_kwargs.pop("domain", None)
        output, lowered_domain = self._visit_stencil_call_out_arg(
            node_kwargs.pop("out"), domain, **kwargs
        )

        assert isinstance(node.func.type, (ts_ffront.FieldOperatorType, ts_ffront.ScanOperatorType))

        args, node_kwargs = type_info.canonicalize_arguments(
            node.func.type, node.args, node_kwargs, use_signature_ordering=True
        )

        lowered_args, lowered_kwargs = self.visit(args, **kwargs), self.visit(node_kwargs, **kwargs)

        return itir.SetAt(
            expr=im.call(node.func.id)(*lowered_args, *lowered_kwargs.values()),
            domain=lowered_domain,
            target=output,
        )

    def _visit_slice_bound(
        self,
        slice_bound: Optional[past.Constant],
        default_value: itir.Expr,
        start_idx: itir.Expr,
        stop_idx: itir.Expr,
        **kwargs: Any,
    ) -> itir.Expr:
        if slice_bound is None:
            lowered_bound = default_value
        elif isinstance(slice_bound, past.Constant):
            assert isinstance(slice_bound.type, ts.ScalarType) and type_info.is_integral(
                slice_bound.type
            )
            if slice_bound.value < 0:
                lowered_bound = im.plus(stop_idx, self.visit(slice_bound, **kwargs))
            else:
                lowered_bound = im.plus(start_idx, self.visit(slice_bound, **kwargs))
        else:
            raise AssertionError("Expected 'None' or 'past.Constant'.")
        if slice_bound:
            lowered_bound.location = slice_bound.location
        return lowered_bound

    def _construct_itir_out_arg(self, node: past.Expr) -> itir.Expr:
        if isinstance(node, past.Name):
            return itir.SymRef(id=node.id, location=node.location)
        elif isinstance(node, past.Subscript):
            itir_node = self._construct_itir_out_arg(node.value)
            itir_node.location = node.location
            return itir_node
        elif isinstance(node, past.TupleExpr):
            return itir.FunCall(
                fun=itir.SymRef(id="make_tuple"),
                args=[self._construct_itir_out_arg(el) for el in node.elts],
                location=node.location,
            )
        else:
            raise ValueError(
                "Unexpected 'out' argument. Must be a 'past.Name', 'past.Subscript'"
                " or a 'past.TupleExpr' thereof."
            )

    def _construct_itir_domain_arg(
        self,
        out_field: past.Name,
        node_domain: Optional[past.Expr],
        slices: Optional[list[past.Slice]] = None,
    ) -> itir.FunCall:
        assert isinstance(out_field.type, ts.TypeSpec)
        out_field_types = type_info.primitive_constituents(out_field.type).to_list()
        out_dims = cast(ts.FieldType, out_field_types[0]).dims
        if any(
            not isinstance(out_field_type, ts.FieldType) or out_field_type.dims != out_dims
            for out_field_type in out_field_types
        ):
            raise AssertionError(
                f"Expected constituents of '{out_field.id}' argument to be"
                " fields defined on the same dimensions. This error should be "
                " caught in type deduction already."
            )

        domain_args = []
        domain_args_kind = []
        for dim_i, dim in enumerate(out_dims):
            # an expression for the range of a dimension
            dim_range = itir.SymRef(id=_range_arg_from_field(out_field.id, dim_i))
            dim_start, dim_stop = im.tuple_get(0, dim_range), im.tuple_get(1, dim_range)
            # bounds
            lower: itir.Expr
            upper: itir.Expr
            if node_domain is not None:
                assert isinstance(node_domain, past.Dict)
                lower, upper = self._construct_itir_initialized_domain_arg(dim_i, dim, node_domain)
            else:
                lower = self._visit_slice_bound(
                    slices[dim_i].lower if slices else None,
                    dim_start,
                    dim_start,
                    dim_stop,
                )
                upper = self._visit_slice_bound(
                    slices[dim_i].upper if slices else None,
                    dim_stop,
                    dim_start,
                    dim_stop,
                )

            if dim.kind == common.DimensionKind.LOCAL:
                raise ValueError(f"common.Dimension '{dim.value}' must not be local.")
            domain_args.append(
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[itir.AxisLiteral(value=dim.value, kind=dim.kind), lower, upper],
                )
            )
            domain_args_kind.append(dim.kind)

        if self.grid_type == common.GridType.CARTESIAN:
            domain_builtin = "cartesian_domain"
        elif self.grid_type == common.GridType.UNSTRUCTURED:
            domain_builtin = "unstructured_domain"
        else:
            raise AssertionError()

        return itir.FunCall(
            fun=itir.SymRef(id=domain_builtin),
            args=domain_args,
            location=(node_domain or out_field).location,
        )

    def _construct_itir_initialized_domain_arg(
        self, dim_i: int, dim: common.Dimension, node_domain: past.Dict
    ) -> list[itir.FunCall]:
        assert len(node_domain.values_[dim_i].elts) == 2
        keys_dims_types = cast(ts.DimensionType, node_domain.keys_[dim_i].type).dim
        if keys_dims_types != dim:
            raise ValueError(
                "common.Dimensions in out field and field domain are not equivalent:"
                f"expected '{dim}', got '{keys_dims_types}'."
            )

        return [self.visit(bound) for bound in node_domain.values_[dim_i].elts]

    @staticmethod
    def _compute_field_slice(node: past.Subscript) -> list[past.Slice]:
        out_field_name: past.Name = node.value
        out_field_slice_: list[past.Slice]
        if isinstance(node.slice_, past.TupleExpr) and all(
            isinstance(el, past.Slice) for el in node.slice_.elts
        ):
            out_field_slice_ = cast(list[past.Slice], node.slice_.elts)  # type ensured by if
        elif isinstance(node.slice_, past.Slice):
            out_field_slice_ = [node.slice_]
        else:
            raise AssertionError(
                "Unexpected 'out' argument, must be tuple of slices or slice expression."
            )
        node_dims = cast(ts.FieldType, node.type).dims
        assert isinstance(node_dims, list)
        if isinstance(node.type, ts.FieldType) and len(out_field_slice_) != len(node_dims):
            raise errors.DSLError(
                node.location,
                f"Too many indices for field '{out_field_name}': field is {len(node_dims)}"
                f"-dimensional, but {len(out_field_slice_)} were indexed.",
            )
        return out_field_slice_

    def _visit_stencil_call_out_arg(
        self, out_arg: past.Expr, domain_arg: Optional[past.Expr], **kwargs: Any
    ) -> tuple[itir.Expr, itir.FunCall]:
        if isinstance(out_arg, past.Subscript):
            # as the ITIR does not support slicing a field we have to do a deeper
            # inspection of the PAST to emulate the behaviour
            out_field_name: past.Name = out_arg.value
            return (
                self._construct_itir_out_arg(out_field_name),
                self._construct_itir_domain_arg(
                    out_field_name, domain_arg, self._compute_field_slice(out_arg)
                ),
            )
        elif isinstance(out_arg, past.Name):
            return (
                self._construct_itir_out_arg(out_arg),
                self._construct_itir_domain_arg(out_arg, domain_arg),
            )
        elif isinstance(out_arg, past.TupleExpr):
            flattened = _flatten_tuple_expr(out_arg)

            first_field = flattened[0]
            assert all(
                self.visit(field.type).dims == self.visit(first_field.type).dims
                for field in flattened
            ), "Incompatible fields in tuple: all fields must have the same dimensions."

            field_slice = None
            if isinstance(first_field, past.Subscript):
                assert all(
                    isinstance(field, past.Subscript) for field in flattened
                ), "Incompatible field in tuple: either all fields or no field must be sliced."
                assert all(
                    concepts.eq_nonlocated(
                        first_field.slice_,
                        field.slice_,  # type: ignore[union-attr] # mypy cannot deduce type
                    )
                    for field in flattened
                ), "Incompatible field in tuple: all fields must be sliced in the same way."
                field_slice = self._compute_field_slice(first_field)
                first_field = first_field.value

            return (
                self._construct_itir_out_arg(out_arg),
                self._construct_itir_domain_arg(first_field, domain_arg, field_slice),
            )
        else:
            raise AssertionError(
                "Unexpected 'out' argument. Must be a 'past.Subscript', 'past.Name' or 'past.TupleExpr' node."
            )

    def visit_Constant(self, node: past.Constant, **kwargs: Any) -> itir.Literal:
        if isinstance(node.type, ts.ScalarType) and node.type.shape is None:
            match node.type.kind:
                case ts.ScalarKind.STRING:
                    raise NotImplementedError(
                        f"Scalars of kind '{node.type.kind}' not supported currently."
                    )
            typename = node.type.kind.name.lower()
            return im.literal(str(node.value), typename)

        raise NotImplementedError("Only scalar literals supported currently.")

    def visit_Name(self, node: past.Name, **kwargs: Any) -> itir.SymRef:
        return itir.SymRef(id=node.id)

    def visit_Symbol(self, node: past.Symbol, **kwargs: Any) -> itir.Sym:
        return itir.Sym(id=node.id, type=node.type)

    def visit_BinOp(self, node: past.BinOp, **kwargs: Any) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id=node.op.value),
            args=[self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)],
        )

    def visit_Call(self, node: past.Call, **kwargs: Any) -> itir.FunCall:
        if node.func.id in ["maximum", "minimum"]:
            assert len(node.args) == 2
            return itir.FunCall(
                fun=itir.SymRef(id=node.func.id),
                args=[self.visit(node.args[0]), self.visit(node.args[1])],
            )
        else:
            raise NotImplementedError("Only 'minimum', and 'maximum' builtins supported currently.")
