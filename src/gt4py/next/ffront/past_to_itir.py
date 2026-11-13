# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, cast

import devtools

from gt4py.eve import NodeTranslator, traits
from gt4py.next import common, config, errors, utils
from gt4py.next.ffront import (
    fbuiltins,
    gtcallable,
    program_ast as past,
    stages as ffront_stages,
    transform_utils,
    type_info as ffront_ti,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.stages import ConcretePASTProgramDef
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im, misc as ir_misc
from gt4py.next.iterator.transforms import remap_symbols, replace_get_domain_range_with_constants
from gt4py.next.otf import arguments, stages, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


# FIXME[#1582](tehrengruber): This should only depend on the program not the arguments. Remove
#  dependency as soon as column axis can be deduced from ITIR in consumers of the CompilableProgram.
def past_to_gtir(inp: ConcretePASTProgramDef) -> stages.CompilableProgramDef:
    """
    Lower a PAST program definition to Iterator IR.

    Example:
        >>> from gt4py import next as gtx
        >>> from gt4py.next.otf import arguments, workflow
        >>> IDim = gtx.Dimension("I")

        >>> @gtx.field_operator
        ... def copy(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a

        >>> @gtx.program
        ... def copy_program(
        ...     a: gtx.Field[[IDim], gtx.float32], out: gtx.Field[[IDim], gtx.float32]
        ... ):
        ...     copy(a, out=out)

        >>> compile_time_args = arguments.CompileTimeArgs(
        ...     args=tuple(param.type for param in copy_program.past_stage.past_node.params),
        ...     kwargs={},
        ...     offset_provider={},
        ...     column_axis=None,
        ...     argument_descriptor_contexts={},
        ... )

        >>> itir_copy = past_to_gtir(
        ...     workflow.ConcreteArtifact(copy_program.past_stage, compile_time_args)
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

    past_node = _specialize_program_params(inp.data.past_node, inp.args.args, inp.args.kwargs)

    itir_program = ProgramLowering.apply(
        past_node, function_definitions=lowered_funcs, grid_type=grid_type
    )

    # TODO(tehrengruber): Put this in a dedicated transformation step.
    if arguments.StaticArg in inp.args.argument_descriptor_contexts:
        static_arg_descriptors = inp.args.argument_descriptor_contexts[arguments.StaticArg]
        if not all(
            isinstance(arg_descriptor, arguments.StaticArg)
            or all(el is None for el in utils.flatten_nested_tuple(arg_descriptor))  # type: ignore[arg-type]
            for arg_descriptor in static_arg_descriptors.values()
        ):
            raise NotImplementedError("Only top-level arguments can be static.")
        static_args = {
            name: im.literal_from_tuple_value(descr.value)  # type: ignore[union-attr]  # type checked above
            for name, descr in static_arg_descriptors.items()
            if not any(el is None for el in utils.flatten_nested_tuple(descr))  # type: ignore[arg-type]
        }
        body = remap_symbols.RemapSymbolRefs().visit(itir_program.body, symbol_map=static_args)
        itir_program = itir.Program(
            id=itir_program.id,
            function_definitions=itir_program.function_definitions,
            params=itir_program.params,
            declarations=itir_program.declarations,
            body=body,
        )

    # TODO(tehrengruber): Put this in a dedicated transformation step.
    if context := inp.args.argument_descriptor_contexts.get(arguments.FieldDomainDescriptor, None):
        field_domains = {
            param: utils.tree_map(lambda x: x.domain if x is not None else x)(v)
            for param, v in context.items()
        }
        itir_program = (
            replace_get_domain_range_with_constants.ReplaceGetDomainRangeWithConstants.apply(
                itir_program, sizes=field_domains
            )
        )

    # Translate NamedCollectionTypes to TupleTypes in compile-time args
    args = tuple(ffront_ti.named_collections_to_tuple_types(arg) for arg in inp.args.args)
    kwargs: dict[str, ts.TypeSpec] = {
        k: ffront_ti.named_collections_to_tuple_types(v) for k, v in inp.args.kwargs.items()
    }
    compile_time_args = dataclasses.replace(
        inp.args, args=args, kwargs=kwargs, column_axis=_column_axis(all_closure_vars)
    )

    if config.DEBUG or inp.data.debug:
        devtools.debug(itir_program)

    return stages.CompilableProgramDef(data=itir_program, args=compile_time_args)


def past_to_gtir_factory(
    cached: bool = True,
) -> workflow.Workflow[ConcretePASTProgramDef, stages.CompilableProgramDef]:
    wf = workflow.make_step(past_to_gtir)
    if cached:
        wf = workflow.CachedStep.in_memory(
            wf, input_fingerprinter=ffront_stages.semantic_fingerprinter
        )
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
        scanops_per_axis_str = "\n".join(
            f"- {dim.value}: {', '.join(scanops)}" for dim, scanops in scanops_per_axis.items()
        )

        raise TypeError(
            "Only 'ScanOperator's defined on the same axis "
            f"can be used in a 'Program', found:\n{scanops_per_axis_str}\n"
        )

    return iter(scanops_per_axis.keys()).__next__()


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


def _unwrap_tuple_expr(
    expr: past.Expr, path: tuple[int | type_info.VarLenTupleElementKey, ...]
) -> tuple[past.Expr, Sequence[int | type_info.VarLenTupleElementKey]]:
    """Unwrap (nested) TupleExpr by following the given path as long as possible.

    If a non-tuple expression is encountered, the current expression and the remaining path are
    returned.
    """
    path_remainder: Sequence[int | type_info.VarLenTupleElementKey] = path
    while isinstance(expr, past.TupleExpr):
        idx, *path_remainder = path_remainder
        # a `TupleExpr` always has a fixed-length tuple type, so the path element is an index
        assert isinstance(idx, int)
        expr = expr.elts[idx]

    return expr, path_remainder


def _specialize_program_params(
    past_node: past.Program,
    arg_types: Sequence[ts.TypeSpec],
    kwarg_types: dict[str, ts.TypeSpec],
    check_compatibility: bool = True,
) -> past.Program:
    """
    Replace the program parameter types by the concrete argument types of the call.

    In particular, parameters declared with a generic type, e.g. a variable-length tuple,
    only have a concrete type at the call. The arguments are expected to be canonicalized
    against the signature already (see `transform_program_args`).
    """
    param_types: dict[str, ts.TypeSpec] = {
        **{str(param.id): arg_type for param, arg_type in zip(past_node.params, arg_types)},
        **kwarg_types,
    }
    new_params = []
    for param in past_node.params:
        param_type = param_types[str(param.id)]
        # incompatible argument types have already been rejected in `transform_program_args`
        assert not check_compatibility or type_info.is_concretizable(
            param.type, to_type=param_type
        ), (
            f"Argument of type '{param_type}' is not compatible with parameter "
            f"'{param.id}' of type '{param.type}'."
        )
        new_params.append(param.copy(update={"type": param_type}))
    return past_node.copy(update={"params": new_params})


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

        set_ats = [self._visit_field_operator_call(stmt, **kwargs) for stmt in node.body]
        return itir.Program(
            id=node.id,
            function_definitions=function_definitions,
            params=params,
            declarations=[],
            body=set_ats,
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

        args, node_kwargs = type_info.canonicalize_arguments(node.func.type, node.args, node_kwargs)

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

    def _construct_itir_out_arg(self, node: past.Expr, allow_slicing: bool) -> itir.Expr:
        if isinstance(node, past.Name):
            return itir.SymRef(id=node.id, location=node.location)
        elif isinstance(node, past.Subscript):
            assert allow_slicing
            itir_node = self._construct_itir_out_arg(node.value, allow_slicing)
            itir_node.location = node.location
            return itir_node
        elif isinstance(node, past.TupleExpr):
            return itir.FunCall(
                fun=itir.SymRef(id="make_tuple"),
                args=[self._construct_itir_out_arg(el, allow_slicing) for el in node.elts],
                location=node.location,
            )
        else:
            raise ValueError(
                "Unexpected 'out' argument. Must be a 'past.Name', 'past.Subscript'"
                " or a 'past.TupleExpr' thereof."
            )

    def _construct_itir_domain_arg(
        self,
        out_expr: itir.Expr,
        out_type: ts.FieldType,
        slices: Optional[list[past.Slice]] = None,
    ) -> itir.FunCall:
        ranges: dict[common.Dimension, tuple[itir.Expr, itir.Expr]] = {}
        for dim_i, dim in enumerate(out_type.dims):
            if dim.kind == common.DimensionKind.LOCAL:
                raise ValueError(f"common.Dimension '{dim.value}' must not be local.")
            # an expression for the range of a dimension
            dim_range = im.call("get_domain_range")(out_expr, im.axis_literal(dim))

            dim_start, dim_stop = im.tuple_get(0, dim_range), im.tuple_get(1, dim_range)
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
            ranges[dim] = (lower, upper)

        domain_expr = im.domain(self.grid_type, ranges)
        domain_expr.location = out_expr.location
        return domain_expr

    def _lower_domain_arg(self, domain_arg: past.Expr) -> itir.Expr:
        if isinstance(domain_arg, past.TupleExpr):
            return im.make_tuple(*(self._lower_domain_arg(el) for el in domain_arg.elts))
        assert isinstance(domain_arg, past.Dict)
        ranges = {
            cast(ts.DimensionType, dim_name.type).dim: (
                self.visit(bounds.elts[0]),
                self.visit(bounds.elts[1]),
            )
            for dim_name, bounds in zip(domain_arg.keys_, domain_arg.values_, strict=True)
        }
        domain_expr = im.domain(self.grid_type, ranges)
        domain_expr.location = domain_arg.location
        return domain_expr

    def _split_field_and_slice(
        self, field: past.Name | past.Subscript
    ) -> tuple[past.Name, list[past.Slice] | None]:
        if isinstance(field, past.Subscript):
            return field.value, _compute_field_slice(field)
        else:
            assert isinstance(field, past.Name)
            return field, None

    def _visit_stencil_call_out_arg(
        self, out_arg: past.Expr, domain_arg: Optional[past.Expr], **kwargs: Any
    ) -> tuple[itir.Expr, itir.Expr]:
        assert isinstance(out_arg, (past.Subscript, past.Name, past.TupleExpr)), (
            "Unexpected 'out' argument. Must be a 'past.Subscript', 'past.Name' or 'past.TupleExpr' node."
        )
        assert out_arg.type is not None

        if domain_arg is not None:
            # Mutual exclusivity of a `domain` argument and slices is ensured in the type
            # deduction, so the target is just the plain lowered `out` argument. Note that
            # a single domain may apply to an entire tuple subtree of the target (see
            # `itir.SetAt`).
            return self._construct_itir_out_arg(
                out_arg, allow_slicing=False
            ), self._lower_domain_arg(domain_arg)
        else:

            @ir_misc.tree_map_type_into_expr(with_path_arg=True)
            def impl(
                out_type: ts.FieldType,
                current: itir.Expr,
                path: tuple[int | type_info.VarLenTupleElementKey, ...],
            ) -> itir.Expr:
                out_field, path_remainder = _unwrap_tuple_expr(out_arg, path)

                # A subscript (domain slice) can only appear on a single field, i.e. on an
                # expression that does not decompose into further tuple elements.
                assert isinstance(out_field, past.Name) or (
                    isinstance(out_field, past.Subscript) and len(path_remainder) == 0
                )
                _, slice_info = self._split_field_and_slice(out_field)

                return self._construct_itir_domain_arg(current, out_type, slice_info)

            lowered_out = self._construct_itir_out_arg(out_arg, allow_slicing=True)
            lowered_domain = impl(out_arg.type, lowered_out)

            return lowered_out, lowered_domain

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
        return itir.SymRef(id=node.id, location=node.location)

    def visit_Symbol(self, node: past.Symbol, **kwargs: Any) -> itir.Sym:
        if isinstance(node.type, ts.COLLECTION_TYPE_SPECS):
            new_symbol_type: ts.TypeSpec = ffront_ti.named_collections_to_tuple_types(node.type)
        else:
            new_symbol_type = node.type
        return itir.Sym(id=node.id, type=new_symbol_type)

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
