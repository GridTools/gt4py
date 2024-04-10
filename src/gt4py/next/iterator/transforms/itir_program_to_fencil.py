import copy
import typing
from typing import TypeGuard, Any

import dataclasses

import numpy as np

from gt4py import eve
from gt4py.eve import concepts
from gt4py.eve.utils import UIDGenerator
from gt4py.next.common import NeighborTable, Dimension, Connectivity, promote_dims
from gt4py.next.ffront import lowering_utils
from gt4py.next.ffront.lowering_utils import primitive_constituents, \
    apply_to_structual_primitive_constituents
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_let, is_call_to
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, domain_intersection, \
    SymbolicRange, FencilWithTemporaries, Temporary, domain_union, convert_type, _group_offsets, \
    _max_domain_sizes_by_location_type
from gt4py.next.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts
from gt4py.next.iterator.type_inference import infer_all, infer


def tuple_expr_to_list(node: itir.Node):
    if is_call_to(node, "make_tuple"):
        return node.args
    return node

class SimplifyGetDomain(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        if is_call_to(node, "get_domain") and is_call_to(node.args[0], "apply_stencil"):
            _, _, domain = node.args[0].args
            return domain
        return self.generic_visit(node)

@dataclasses.dataclass
class PropagateAndSimplifyDomain(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    offset_provider: typing.Any

    @classmethod
    def apply(cls, node: itir.Program, offset_provider):
        return cls(offset_provider=offset_provider).visit(node, domains=node.domains)

    def visit_FunCall(self, node: itir.Program, **kwargs):
        domains: dict[str, typing.Any] = kwargs["domains"]

        # TODO: document why only once
        if isinstance(node.fun, itir.Lambda):
            new_kwargs = {k: v for k, v in kwargs.items() if k != "domains"}
            args = self.generic_visit(node.args, **kwargs)
            fun = self.generic_visit(node.fun, domains={
                **domains,
                **{param.id: arg.annex.domain for param, arg in zip(node.fun.params, args, strict=True) if
                   hasattr(arg.annex, "domain")}
            }, **new_kwargs)
            node = itir.FunCall(fun=fun, args=args)
            if hasattr(fun.expr.annex, "domain"):
                node.annex.domain = fun.expr.annex.domain
            return node

        node = self.generic_visit(node, **kwargs)

        if is_call_to(node, "apply_stencil"):
            _, _, domain = node.args
            node.annex.domain = domain

        if is_call_to(node, "broadcast_to_common_domain"):
            args = self.visit(node.args, domains=domains)
            domains = [arg.annex.domain for arg in args]
            if len(domains) == 1:
                return self.visit(domains[0], **kwargs)
            promoted_dim = promote_dims(*([Dimension(value=dim) for dim in SymbolicDomain.from_expr(domain).ranges.keys()] for domain in domains))
            return self.visit(
                im.call("intersect")(*(im.call("broadcast_domain")(domain, im.make_tuple(*(itir.AxisLiteral(value=dim.value) for dim in promoted_dim))) for domain in domains)),
                **kwargs)

        if is_call_to(node, "broadcast_domain"):
            domain_expr, axes_expr = node.args
            self.visit(domain_expr, domains=domains)
            domain = SymbolicDomain.from_expr(domain_expr)
            axes = tuple_expr_to_list(axes_expr)
            ranges = [*domain.ranges.items()]
            new_ranges = {}
            for axis in axes:
                if len(ranges) > 0 and axis.value == ranges[0][0]:
                    new_ranges[axis.value] = ranges[0][1]
                    ranges.pop(0)
                else:
                    new_ranges[axis.value] = SymbolicRange(
                        im.literal("neginf", "int32"),
                        im.literal("inf", "int32")
                    )
            assert len(ranges) == 0
            return self.visit(SymbolicDomain(domain.grid_type, new_ranges).as_expr(), **kwargs)

        if is_call_to(node, "strip_domain_axis"):
            domain = SymbolicDomain.from_expr(node.args[0])
            domain.ranges.pop(node.args[1].value)
            return self.visit(domain.as_expr(), **kwargs)

        if is_call_to(node, "intersect"):
            domains = [SymbolicDomain.from_expr(arg) for arg in node.args]
            return self.visit(domain_intersection(domains).as_expr(), **kwargs)

        if is_call_to(node, "inverse_translate_domain"):
            domain_expr, offset_name, offset = node.args
            domain = SymbolicDomain.from_expr(domain_expr)
            offset_provider = self.offset_provider[offset_name.value]
            ranges = {}
            for axis, range_ in SymbolicDomain.from_expr(domain_expr).ranges.items():
                if isinstance(offset_provider, NeighborTable) and axis == offset_provider.neighbor_axis.value:
                    ranges[offset_provider.origin_axis.value] = SymbolicRange(
                        np.min(offset_provider.table),
                        np.max(offset_provider.table)
                    )
                elif isinstance(offset_provider, Dimension):
                    ranges[axis] = range_.translate(offset)
                else:
                    ranges[axis] = range_
            return self.visit(SymbolicDomain(domain.grid_type, ranges).as_expr(), **kwargs)

        if is_call_to(node, "get_range_from_domain"):
            raise NotImplementedError()

        # TODO: rethink this
        if is_call_to(node, "tuple_get") and hasattr(node.args[1].annex, "domain"):
            node.annex.domain = self.visit(node.args[1].annex.domain, **kwargs)

        if is_call_to(node, "make_tuple") and any(hasattr(arg.annex, "domain") for arg in node.args):
            assert all(hasattr(arg.annex, "domain") for arg in node.args)
            # TODO: intersect?
            node.annex.domain = self.visit(node.args[0].annex.domain, **kwargs)

        # here we actually replace the domain
        if is_call_to(node, "get_domain") and hasattr(node.args[0].annex, "domain"):
            return self.visit(node.args[0].annex.domain, **kwargs)

        if is_call_to(node, "if_"):
            condition, true_value, false_value = node.args
            if hasattr(true_value.annex, "domain") and hasattr(false_value.annex, "domain"):
                node = im.if_(condition, true_value, false_value)
                node.annex.domain = self.visit(im.call("intersect")(true_value.annex.domain, false_value.annex.domain), **kwargs)
                return node

        return node

    def visit_SymRef(self, node: itir.SymRef, *, domains, **kwargs):
        if node.id in domains:
            node.annex.domain = domains[node.id]
        # TODO: maybe fail if not domain builtin as everything else should be a field? how about tuple
        return node


def is_apply_stencil_with_let_arg(node: itir.FunCall):
    if is_call_to(node, "apply_stencil"):
        _, args_expr, _ = node.args
        args = tuple_expr_to_list(args_expr)
        return any(is_let(arg) for arg in args)
    return False

def is_apply_stencil_with_let_args(node: itir.FunCall):
    if is_call_to(node, "apply_stencil"):
        _, args_expr, _ = node.args
        return is_let(args_expr)
    return False

pliasi_uids = UIDGenerator(prefix="__pliasi")
class PropagateLetInApplyStencilInput(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        if is_apply_stencil_with_let_args(node):  # apply_stencil(stencil, let(...), domain)
            stencil, args_expr, domain = node.args
            used_symbols = set(collect_symbol_refs(domain)) | set(collect_symbol_refs(stencil))
            assert not any(bound_var.id in used_symbols for bound_var in args_expr.fun.params)
            return self.visit(im.let(*zip(args_expr.fun.params, args_expr.args))(
                im.apply_stencil(stencil, args_expr.fun.expr, domain)
            ))
        elif is_apply_stencil_with_let_arg(node):  # apply_stencil(stencil, {let(...), ...}, domain)
            let_args: dict[str, itir.Expr] = {}
            new_args = []

            stencil, args_expr, domain = node.args
            for arg in tuple_expr_to_list(args_expr):
                if is_let(arg):
                    symbol_map: dict[str, str] = {}
                    for let_param, let_arg in zip(arg.fun.params, arg.args, strict=True):
                        new_param = pliasi_uids.sequential_id()
                        symbol_map[let_param.id] = new_param
                        let_args[new_param] = let_arg
                    new_arg = inline_lambdas.inline_lambda(im.let(
                        *zip(arg.fun.params, symbol_map.values(), strict=True)
                    )(arg.fun.expr), symbol_map.keys())
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)
            new_node = im.let(
                *let_args.items()
            )(im.apply_stencil(stencil, im.make_tuple(*new_args), domain))
            return self.generic_visit(new_node)  # revisit again in case there are more let args
        return node


def inline_predicate(node: itir.FunCall):
    if is_call_to(node, "apply_stencil"):
        stencil, _, domain_expr = node.args

        if is_call_to(stencil, "scan"):
            return False

        domain = SymbolicDomain.from_expr(domain_expr)
        for range_ in domain.ranges.values():
            if isinstance(range_.start, itir.Literal) and range_.start.value in ["inf", "neginf"]:
                return True
            if isinstance(range_.stop, itir.Literal) and range_.stop.value in ["inf", "neginf"]:
                return True
    assert hasattr(node.annex, "recorded_shifts")
    return len(node.annex.recorded_shifts) < 2


def promote_stencil_to_lambda(node: itir.SymRef | itir.Lambda):
    if node == im.ref("deref"):
        return im.lambda_("it")(im.deref("it"))
    assert isinstance(node, itir.Lambda)
    return node

def inline_into_stencil(stencil: itir.Lambda | itir.SymRef, arg_idx: int, arg: itir.Expr, uid_gen: UIDGenerator) -> tuple[itir.Lambda, itir.Expr]:
    if is_call_to(stencil, "scan"):
        breakpoint()
    stencil = promote_stencil_to_lambda(stencil)
    inlined_stencil, inlined_args_expr, _ = arg.args

    new_arg = im.sym(uid_gen.sequential_id())

    lifted_stencil = im.lift(inlined_stencil)(*[im.tuple_get(i, new_arg.id) for i in range(len(inlined_stencil.params))])
    new_stencil_body = im.let(stencil.params[arg_idx], lifted_stencil)(stencil.expr)

    new_stencil_params = [param if i!=arg_idx else new_arg.id for i, param in enumerate(stencil.params)]
    new_stencil = im.lambda_(*new_stencil_params)(new_stencil_body)

    return new_stencil, inlined_args_expr

class FieldViewToLocalView(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        # do not visit children in apply stencil as the make_tuple of its inputs should stay as is
        #  e.g. apply_stencil(stencil, {...}, domain)
        # here the {...} should not be transformed. Additionally visiting stencils does not make
        # sense they are local already
        # TODO: this breaks with let statements  let inputs={} apply_stencil(stencil, inputs, domain)
        if is_call_to(node, "apply_stencil"):
            stencil, args, _ = node.args
            new_args = [self.visit(arg) for arg in tuple_expr_to_list(args)]

            return im.lift(stencil)(*new_args)

        node = self.generic_visit(node)

        if is_call_to(node, "tuple_get"):
            index, tuple_expr = node.args
            assert isinstance(index, itir.Literal)
            return im.lift(im.lambda_("it")(im.tuple_get(index, im.deref("it"))))(tuple_expr)

        if is_call_to(node, "make_tuple"):
            if is_call_to(node, "make_tuple"):
                # convert into iterator of tuples
                els = tuple_expr_to_list(node.args)
                return im.lift(im.lambda_(
                    *(f"it{i}" for i in range(len(els)))
                )(im.make_tuple(*(im.deref(f"it{i}") for i in range(len(els))))))(*els)
            return node

        if is_call_to(node, "if_"):  # params used in condition need to be deref for scalar if
            condition, true_branch, false_branch = node.args
            derefed_condition_args = {sym: im.deref(sym) for sym in collect_symbol_refs(condition)}
            condition = im.let(*derefed_condition_args.items())(condition)

            return im.if_(
                condition,
                true_branch,
                false_branch
            )

        return node


def canonicalize_apply_stencil(node: itir.FunCall, uid_gen):
    stencil, args, domain = node.args[0], tuple_expr_to_list(node.args[1]), node.args[2]

    if is_call_to(stencil, "scan"):
        definition = stencil.args[0]
        stencil_params = definition.params[1:]
        stencil_body = definition.expr
    else:
        stencil = promote_stencil_to_lambda(stencil)
        stencil_params = stencil.params
        stencil_body = stencil.expr

    new_params = []
    new_args = []

    def _impl(arg):
        new_name = uid_gen.sequential_id()
        new_params.append(im.sym(new_name))
        new_args.append(arg)
        replaced_arg = im.ref(new_name)
        replaced_arg.annex.recorded_shifts = arg.annex.recorded_shifts
        return replaced_arg

    did_canonicalize = False
    new_stencil_body = stencil_body
    for param, arg in zip(stencil_params, args, strict=True):
        if is_call_to(arg, "make_tuple"):
            did_canonicalize = True
            new_stencil_body = im.let(
                param,
                apply_to_structual_primitive_constituents(_impl, arg)
            )(new_stencil_body)
        elif is_call_to(arg, "if_"):  # scalar if is not supported so inline
            # TODO: canonicalize again
            did_canonicalize = True
            condition, true_branch, false_branch = arg.args
            true_branch_sym = uid_gen.sequential_id()
            false_branch_sym = uid_gen.sequential_id()
            outer_map = {uid_gen.sequential_id(): sym for sym in collect_symbol_refs(condition)}
            inner_map = {sym: im.deref(gen_sym) for gen_sym, sym in outer_map.items()}
            new_stencil_body = im.let(
                param, im.if_(im.let(*inner_map.items())(condition), true_branch_sym, false_branch_sym)
            )(new_stencil_body)
            new_params = [*new_params, *outer_map.keys(), true_branch_sym, false_branch_sym]
            new_args = [*new_args, *outer_map.values(), true_branch, false_branch]
        elif is_let(arg):
            raise ValueError("Not let statements allowed in arguments to apply stencil. Propagate first.")
        else:  # keep as is
            new_params.append(param)
            new_args.append(arg)

    if is_call_to(stencil, "scan"):
        definition = stencil.args[0]
        new_stencil = im.call("scan")(
            im.lambda_(definition.params[0], *new_params)(new_stencil_body),
            *stencil.args[1:])
    else:
        new_stencil = im.lambda_(*new_params)(new_stencil_body)

    new_apply_stencil = im.apply_stencil(
        new_stencil,
        im.make_tuple(*new_args),
        domain
    )
    new_apply_stencil.annex.recorded_shifts = node.annex.recorded_shifts

    # TODO: canonicalize again if needed for nested tuples

    return did_canonicalize, new_apply_stencil

def apply_stencil_inliner(node: itir.FunCall, uid_gen):
    stencil, args, domain = node.args[0], tuple_expr_to_list(node.args[1]), node.args[2]

    if is_call_to(stencil, "scan"):
        definition = stencil.args[0]
        new_stencil_body = definition.expr
        stencil_params = definition.params[1:]
    else:
        stencil = promote_stencil_to_lambda(stencil)
        new_stencil_body = stencil.expr
        stencil_params = stencil.params

    did_inline = False
    new_params = []
    new_args = []

    for param, arg in zip(stencil_params, args, strict=True):
        if not isinstance(arg, itir.SymRef) and inline_predicate(arg):
            did_inline = True

            lifted_stencil = FieldViewToLocalView().visit(arg)

            used_symbols = list(set(collect_symbol_refs(lifted_stencil)) - set(
                itir.CIR_BUILTINS))  # TODO: collect_symbol_refs uses the regular builtins of the IR

            # avoid collisions, todo: only if needed
            inner_stencil_args = [im.ref(uid_gen.sequential_id()) for _ in
                                  range(len(used_symbols))]
            lifted_stencil = im.let(*zip(used_symbols, inner_stencil_args))(lifted_stencil)

            new_stencil_body = im.let(param, lifted_stencil)(new_stencil_body)

            # TODO: not so nice, just to get the recorded shifts
            used_symbols_refs = {
                sym.id: sym for sym in arg.pre_walk_values().if_isinstance(itir.SymRef).filter(
                    lambda sym: sym.id in used_symbols)
            }

            new_params = [*new_params, *[im.sym(arg.id) for arg in inner_stencil_args]]
            new_args = [*new_args, *[used_symbols_refs[s] for s in used_symbols]]
        else:
            new_params.append(param)  # take param as is
            new_args.append(arg)

    if is_call_to(stencil, "scan"):
        definition = stencil.args[0]
        new_stencil = im.call("scan")(
            im.lambda_(definition.params[0], *new_params)(new_stencil_body),
            *stencil.args[1:]
        )
    else:
        new_stencil = im.lambda_(*new_params)(new_stencil_body)
    new_apply_stencil = im.apply_stencil(new_stencil, im.make_tuple(*new_args), domain)
    if not hasattr(node.annex, "recorded_shifts"):
        breakpoint()
    new_apply_stencil.annex.recorded_shifts = node.annex.recorded_shifts  # todo

    return did_inline, new_apply_stencil

@dataclasses.dataclass
class ReplaceUnspecifiedGridType(eve.NodeTranslator):
    domain_builtin: typing.Literal["unstructured_domain", "cartesian_domain"]

    @classmethod
    def apply(cls, node: itir.Node):
        domain_constructions = node.pre_walk_values().if_isinstance(itir.SymRef).filter(lambda sym: sym.id in ["unstructured_domain", "cartesian_domain"]).map(lambda sym: sym.id).to_set()
        assert len(domain_constructions) == 1
        return cls(domain_builtin=list(domain_constructions)[0]).visit(node)

    def visit_SymRef(self, node: itir.SymRef):
        if node.id == "domain":
            return im.ref(self.domain_builtin)
        return node

def fused_apply_stencil_calls(node: itir.FunCall):
    fused_apply_stencil_uidgen = UIDGenerator(prefix="__fused_appl_stncl")  # all symbols are local
    assert is_call_to(node, "make_tuple")
    new_params = []
    new_args = []
    domain = None
    recorded_shifts = set()

    def handle_apply_stencil_call(node: itir.FunCall):
        nonlocal new_params, new_args, domain, recorded_shifts
        assert is_call_to(node, "apply_stencil")
        stencil, args_expr, inner_domain = node.args

        if not domain:
            domain = inner_domain
        else:
            assert domain == inner_domain

        recorded_shifts = {*recorded_shifts, *node.annex.recorded_shifts}

        # fuse stencil
        new_inner_args = {}
        for arg in tuple_expr_to_list(args_expr):
            new_inner_param = fused_apply_stencil_uidgen.sequential_id()
            new_inner_args[new_inner_param] = arg
        new_params = [*new_params, *new_inner_args.keys()]
        new_args = [*new_args, *new_inner_args.values()]
        return im.call(stencil)(*new_inner_args.keys())

    new_stencil_body = apply_to_structual_primitive_constituents(handle_apply_stencil_call, node)
    new_stencil = im.lambda_(*new_params)(new_stencil_body)
    new_apply_stencil_call = im.apply_stencil(new_stencil, im.make_tuple(*new_args), domain)
    new_apply_stencil_call.annex.recorded_shifts = recorded_shifts
    return new_apply_stencil_call


class HackScalars(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: itir.Program):
        assert isinstance(node, itir.Program)
        return cls().visit(node)

    def visit_Program(self, node: itir.Program):
        return self.generic_visit(node, context={param.id: None for param in node.params})

    def visit_FunCall(self, node: itir.Program, **kwargs):
        context: dict[str, typing.Any] = kwargs["context"]

        # TODO: document why only once
        if isinstance(node.fun, itir.Lambda):  # handle let statements
            new_kwargs = {k: v for k, v in kwargs.items() if k != "context"}
            args = self.generic_visit(node.args, **kwargs)
            fun = self.generic_visit(node.fun, context={
                **context,
                **{param.id: arg for param, arg in zip(node.fun.params, args, strict=True)}
            }, **new_kwargs)
            node = itir.FunCall(fun=fun, args=args)
            if hasattr(fun.expr.annex, "domain"):
                node.annex.domain = fun.expr.annex.domain
            return node

        if is_call_to(node, "apply_stencil"):  # don't look into stencil
            stencil, args, domain = node.args
            def impl(el: itir.Expr):
                # all literal arguments need to be transformed into a field
                if isinstance(el, itir.Literal):
                    return im.apply_stencil(im.lambda_()(el), im.make_tuple(), im.call("domain")())
                return self.visit(el, **kwargs)
            new_args = apply_to_structual_primitive_constituents(impl, args)
            new_domain = self.visit(domain, **kwargs)  # here we explicitly don't inherit apply_stencil_arg

            return im.apply_stencil(stencil, new_args, new_domain)

        if is_call_to(node, ["domain", "cartesian_domain", "unstructured_domain"]):
            return node

        if is_call_to(node, "scan"):
            return node

        if is_call_to(node, "get_domain"):
            if isinstance(node.args[0], itir.Literal):
                # all literal arguments just have a point like domain
                return im.call("domain")()

        if is_call_to(node, "if_"):
            return im.if_(node.args[0], self.visit(node.args[1], **kwargs), self.visit(node.args[2], **kwargs))

        node = self.generic_visit(node, **kwargs)

        if is_call_to(node, itir.ARITHMETIC_BUILTINS):
            used_symbols = collect_symbol_refs(node)
            node = im.apply_stencil(im.lambda_(*used_symbols)(
                im.let(*zip(used_symbols, map(im.deref, used_symbols)))(node)
            ), im.make_tuple(*used_symbols), im.call("domain")())

        return node

def program_to_fencil(program: itir.Program, offset_provider):
    program = InlineFundefs().visit(program)
    program = PruneUnreferencedFundefs().visit(program)

    # inline lambdas so that all aliases disappear, e.g let a=b f(a) -> f(b)
    # also inline literals
    program = InlineLambdas.apply(
        program,
        opcount_preserving=True,
        force_inline_lambda_args=False,
    )

    program = CollapseTuple.apply(program, use_global_type_inference=False)
    program = HackScalars.apply(program)
    program = ReplaceUnspecifiedGridType.apply(program)
    program = SimplifyGetDomain().visit(program)
    program = PropagateAndSimplifyDomain.apply(program, offset_provider)
    program = ConstantFolding.apply(program)
    program = PropagateLetInApplyStencilInput().visit(program)
    program = CollapseTuple.apply(program, use_global_type_inference=False)

    # inlining
    TraceShifts.apply(program, inputs_only=False, save_to_annex=True)

    tmp_names = UIDGenerator(prefix="__tmp")
    inlined_names = UIDGenerator(prefix="__inlined")

    tmps: list[itir.Sym] = []
    stmt_stack = [*program.stmts]
    new_stmts = []
    while stmt_stack:
        stmt = stmt_stack.pop(0)
        assert isinstance(stmt, itir.Assign)

        # it is much easier to work with apply stencil calls that don't have nested make_tuple calls
        if is_call_to(stmt.expr, "apply_stencil"):
            did_canonicalize, new_apply_stencil_call = canonicalize_apply_stencil(stmt.expr, inlined_names)
            if did_canonicalize:
                new_stmt = itir.Assign(
                    target=stmt.target,
                    expr=new_apply_stencil_call
                )
                stmt_stack = [new_stmt, *stmt_stack]
                continue

        if is_call_to(stmt.expr, "apply_stencil"):
            stencil, args, domain = stmt.expr.args[0], tuple_expr_to_list(stmt.expr.args[1]), stmt.expr.args[2]
            if any(not isinstance(el, itir.SymRef) and not inline_predicate(el) for arg in args for el in primitive_constituents(arg)):
                let_args = {}

                def extract_temporary(arg):
                    if not isinstance(arg, itir.SymRef):
                        tmp_name = tmp_names.sequential_id()
                        let_args[tmp_name] = arg
                        return tmp_name
                    else:
                        return arg
                new_args = [lowering_utils.apply_to_structual_primitive_constituents(extract_temporary, arg) for arg in args]

                new_apply_stencil = im.apply_stencil(stencil, im.make_tuple(*new_args), domain)
                new_apply_stencil.annex.recorded_shifts = stmt.expr.annex.recorded_shifts
                new_assign_expr = im.let(*let_args.items())(new_apply_stencil)
                new_assign_expr.annex.recorded_shifts = new_apply_stencil.annex.recorded_shifts
                new_stmt = itir.Assign(
                    target=stmt.target,
                    expr=new_assign_expr
                )
                stmt_stack = [new_stmt, *stmt_stack]
                continue

            # inlining
            did_inline, new_apply_stencil_call = apply_stencil_inliner(stmt.expr, inlined_names)
            # todo: copy recorded shifts?
            if did_inline:
                new_stmt = itir.Assign(
                    target=stmt.target,
                    expr=new_apply_stencil_call
                )
                stmt_stack = [new_stmt, *stmt_stack]
                continue

            # all elements of input argument to apply stencil should be SymRefs now.
            assert all(isinstance(arg, itir.SymRef) for arg in tuple_expr_to_list(stmt.expr.args[1]))

            new_stmts.append(stmt)
        elif is_let(stmt.expr):
            new_stmt_stack = []
            for param, arg in zip(stmt.expr.fun.params, stmt.expr.args, strict=True):
                # todo: inlining
                tmps.append(param.id)
                tmp_assign = itir.Assign(target=im.ref(param.id), expr=arg)  # TODO: collision
                new_stmt_stack.append(tmp_assign)
            new_stmt = itir.Assign(target=stmt.target, expr=stmt.expr.fun.expr)
            stmt_stack = [*new_stmt_stack, new_stmt, *stmt_stack]
        elif is_call_to(stmt.expr, "make_tuple"):
            # TODO: there are cases were we unnecessarily create apply stencil calls, inline
            #  funcall again first
            new_expr = fused_apply_stencil_calls(stmt.expr)
            new_stmt = itir.Assign(target=stmt.target, expr=new_expr)
            stmt_stack = [new_stmt, *stmt_stack]
        else:
            assert False

    program = itir.Program(
        id=program.id,
        params=program.params,
        tmps=[im.sym(tmp) for tmp in tmps],
        function_definitions=program.function_definitions,
        stmts=new_stmts,
    )

    # we should be able to avoid this
    TraceShifts.apply(program, inputs_only=False, save_to_annex=True)

    symbolic_sizes = None # TODO
    horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)

    # propagate domain backwards
    all_consumed_domains = {}
    for stmt in reversed(program.stmts):
        stencil, args_expr, stmt_domain_expr = stmt.expr.args
        stmt_domain = SymbolicDomain.from_expr(stmt_domain_expr)

        if stmt.target.id in all_consumed_domains:
            stmt_domain = all_consumed_domains[stmt.target.id]

        stencil_params = stencil.args[0].params[1:] if is_call_to(stencil, "scan") else stencil.params
        for param, arg in zip(stencil_params, tuple_expr_to_list(args_expr)):
            if not hasattr(arg.annex, "recorded_shifts"):
                breakpoint()

            if not isinstance(arg, itir.SymRef):
                breakpoint()

            # the domains consumed only by this arg
            consumed_domains: list[SymbolicDomain] = (
                [all_consumed_domains[arg.id]] if arg.id in all_consumed_domains else []
            )

            for shift_chain in param.annex.recorded_shifts:
                consumed_domain = copy.deepcopy(stmt_domain)
                for offset_name, offset in _group_offsets(shift_chain):
                    if isinstance(offset_provider[offset_name], Dimension):
                        # cartesian shift
                        dim = offset_provider[offset_name].value
                        consumed_domain.ranges[dim] = consumed_domain.ranges[dim].translate(offset)
                    elif isinstance(offset_provider[offset_name], Connectivity):
                        # unstructured shift
                        nbt_provider = offset_provider[offset_name]
                        old_axis = nbt_provider.origin_axis.value
                        new_axis = nbt_provider.neighbor_axis.value

                        assert old_axis in consumed_domain.ranges.keys()
                        assert new_axis not in consumed_domain.ranges or old_axis == new_axis

                        if symbolic_sizes is None:
                            new_range = SymbolicRange(
                                im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                                im.literal(
                                    str(horizontal_sizes[new_axis]), itir.INTEGER_INDEX_BUILTIN
                                ),
                            )
                        else:
                            new_range = SymbolicRange(
                                im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                                im.ref(symbolic_sizes[new_axis]),
                            )
                        # TODO(tehrengruber): Revisit. Somehow the order matters so preserve it.
                        consumed_domain.ranges = dict(
                            (axis, range_) if axis != old_axis else (new_axis, new_range)
                            for axis, range_ in consumed_domain.ranges.items()
                        )
                        # TODO(tehrengruber): Revisit. Somehow the order matters so preserve it.
                        consumed_domain.ranges = dict(
                            (axis, range_) if axis != old_axis else (new_axis, new_range)
                            for axis, range_ in consumed_domain.ranges.items()
                        )
                    else:
                        raise NotImplementedError()
                consumed_domains.append(consumed_domain)

            # compute the bounds of all consumed domains
            if consumed_domains:
                all_consumed_domains[arg.id] = domain_union(consumed_domains)

    for stmt in program.stmts:
        if stmt.target.id in all_consumed_domains:
            stmt.expr.args[2] = all_consumed_domains[stmt.target.id].as_expr()

    # get temporary domains
    tmp_domains = {}
    for stmt in program.stmts:
        if stmt.target.id in tmps:
            assert is_call_to(stmt.expr, "apply_stencil")
            _, _, domain = stmt.expr.args
            tmp_domains[stmt.target.id] = domain

    # run type inference to get temporary types
    program_params = {param.id: param for param in program.params}
    var_dtypes = {}
    for stmt in program.stmts:
        assert is_call_to(stmt.expr, "apply_stencil")
        stencil, args_expr, domain = stmt.expr.args
        args = tuple_expr_to_list(args_expr)
        stencil_params = stencil.args[0].params[1:] if is_call_to(stencil,
                                                                  "scan") else stencil.params
        for param, arg in zip(stencil_params, args, strict=True):
            if arg.id in program_params:
                param.kind = program_params[arg.id].kind
                param.dtype = program_params[arg.id].dtype
            else:
                param.raw_dtype = var_dtypes[arg.id]
        var_dtypes[stmt.target.id] = infer(stencil).ret.dtype

    # just to get things nicely looking
    program = ConstantFolding.apply(program)
    program = InlineLambdas.apply(
        program,
        opcount_preserving=True,
        force_inline_lambda_args=False,
    )
    program = InlineLifts.apply(program)

    # translate into legacy stencil
    closures = []
    for stmt in program.stmts:
        assert(is_call_to(stmt.expr, "apply_stencil"))
        stencil, args, domain = stmt.expr.args[0], tuple_expr_to_list(stmt.expr.args[1]), stmt.expr.args[2]

        closures.append(itir.StencilClosure(
            domain=domain,
            stencil=stencil,
            output=stmt.target,
            inputs=args
        ))

    fencil = itir.FencilDefinition(
        id=program.id,
        function_definitions=[],
        params=program.params+[im.sym(tmp) for tmp in tmps],
        closures=closures
    )

    fencil_with_temporaries = FencilWithTemporaries(
        fencil=fencil,
        params=program.params,
        tmps=[Temporary(id=tmp, domain=tmp_domains[tmp], dtype=convert_type(var_dtypes[tmp])) for tmp in tmps]
    )

    return fencil_with_temporaries