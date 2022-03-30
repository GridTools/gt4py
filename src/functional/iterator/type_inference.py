import functools
from dataclasses import dataclass, field, fields

from eve import NodeTranslator
from functional.iterator import ir


datatype = functools.partial(dataclass, frozen=True, slots=True)


class VarMixin:
    _counter = -1

    @classmethod
    def fresh(cls, *args):
        VarMixin._counter += 1
        return cls(VarMixin._counter, *args)


@datatype
class DType:
    ...


@datatype
class Var(DType, VarMixin):
    idx: int


@datatype
class Tuple(DType):
    elems: tuple[DType, ...]


@datatype
class PartialTupleVar(DType, VarMixin):
    idx: int
    elems: tuple[tuple[int, DType], ...]


@datatype
class PrefixTupleVar(DType, VarMixin):
    idx: int
    prefix: DType
    others: DType


@datatype
class Fun(DType):
    args: tuple[DType, ...]
    ret: DType


@datatype
class Val(DType):
    kind: DType = field(default_factory=Var.fresh)
    dtype: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


@datatype
class ValTupleVar(DType, VarMixin):
    idx: int
    kind: DType = field(default_factory=Var.fresh)
    dtypes: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


@datatype
class ValVar(DType, VarMixin):
    idx: int
    kind: DType = field(default_factory=Var.fresh)
    dtype: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


@datatype
class FunVar(DType, VarMixin):
    idx: int
    args: DType
    ret: DType


@datatype
class Column(DType):
    ...


@datatype
class Scalar(DType):
    ...


@datatype
class Primitive(DType):
    name: str


@datatype
class Value(DType):
    ...


@datatype
class Iterator(DType):
    ...


BOOL_DTYPE = Primitive("bool")  # type: ignore [call-arg]
INT_DTYPE = Primitive("int")  # type: ignore [call-arg]
FLOAT_DTYPE = Primitive("float")  # type: ignore [call-arg]


class TypeInferrer(NodeTranslator):
    def visit_SymRef(self, node, *, constraints, symtypes):
        if node.id in symtypes:
            return symtypes[node.id]
        if node.id == "deref":
            dtype = Var.fresh()
            size = Var.fresh()
            return Fun(Tuple((Val(Iterator(), dtype, size),)), Val(Value(), dtype, size))
            return Fun(
                Tuple((ValVar.fresh(Iterator(), dtype, size),)), ValVar.fresh(Value(), dtype, size)
            )
        if node.id in ("plus", "minus", "multiplies", "divides"):
            v = ValVar.fresh(Value())
            v = Val(Value())
            return Fun(Tuple((v, v)), v)
        if node.id in ("eq", "less", "greater"):
            v = Val(Value())
            ret = Val(Value(), BOOL_DTYPE, v.size)
            return Fun(Tuple((v, v)), ret)
        if node.id == "not_":
            v = Val(Value(), BOOL_DTYPE)
            return Fun(Tuple((v,)), v)
        if node.id in ("and_", "or_"):
            v = Val(Value(), BOOL_DTYPE)
            return Fun(Tuple((v, v)), v)
        if node.id == "if_":
            v = Val(Value())
            c = Val(Value(), BOOL_DTYPE, v.size)
            return Fun(Tuple((c, v, v)), v)
        if node.id == "lift":
            args = ValTupleVar(Iterator())
            dtype = Var.fresh()
            size = Var.fresh()
            stencil_ret = ValVar.fresh(Value(), dtype, size)
            lifted_ret = ValVar.fresh(Iterator(), dtype, size)
            stencil_ret = Val(Value(), dtype, size)
            lifted_ret = Val(Iterator(), dtype, size)
            return Fun(Tuple((Fun(args, stencil_ret),)), Fun(args, lifted_ret))
            return Fun(Tuple((FunVar.fresh(args, stencil_ret),)), FunVar.fresh(args, lifted_ret))
        if node.id == "reduce":
            dtypes = Var.fresh()
            size = Var.fresh()
            acc = Val(Value(), Var.fresh(), size)
            f_args = PrefixTupleVar.fresh(acc, ValTupleVar.fresh(Value(), dtypes, size))
            ret_args = ValTupleVar.fresh(Iterator(), dtypes, size)
            f = Fun(f_args, acc)
            ret = Fun(ret_args, acc)
            return Fun(Tuple((f, acc)), ret)
        if node.id == "scan":
            dtypes = Var.fresh()
            fwd = Val(Value(), BOOL_DTYPE, Scalar())
            acc = Val(Value(), Var.fresh(), Scalar())
            f_args = PrefixTupleVar.fresh(acc, ValTupleVar.fresh(Value(), dtypes, Scalar()))
            ret_args = ValTupleVar.fresh(Iterator(), dtypes, Column())
            f = Fun(f_args, acc)
            ret = Fun(ret_args, Val(Value(), acc.dtype, Column()))
            return Fun(Tuple((f, fwd, acc)), ret)

        assert node.id not in ir.BUILTINS
        return Var.fresh()

    def visit_BoolLiteral(self, node, *, constraints, symtypes):
        return Val(Value(), BOOL_DTYPE)
        return ValVar.fresh(Value(), BOOL_DTYPE)

    def visit_IntLiteral(self, node, *, constraints, symtypes):
        return Val(Value(), INT_DTYPE)
        return ValVar.fresh(Value(), INT_DTYPE)

    def visit_FloatLiteral(self, node, *, constraints, symtypes):
        return Val(Value(), FLOAT_DTYPE)
        return ValVar.fresh(Value(), FLOAT_DTYPE)

    def visit_Lambda(self, node, *, constraints, symtypes):
        ptypes = {p.id: Var.fresh() for p in node.params}
        ret = self.visit(node.expr, constraints=constraints, symtypes=symtypes | ptypes)
        return Fun(Tuple(tuple(ptypes[p.id] for p in node.params)), ret)

    def visit_FunCall(self, node, *, constraints, symtypes):
        if isinstance(node.fun, ir.SymRef):
            if node.fun.id == "make_tuple":
                argtypes = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                kind = Var.fresh()
                size = Var.fresh()
                dtype = Tuple(tuple(Var.fresh() for _ in argtypes))
                for d, a in zip(dtype.elems, argtypes):
                    constraints.add((Val(kind, d, size), a))
                return Val(kind, dtype, size)
            if node.fun.id == "tuple_get":
                if len(node.args) != 2:
                    raise TypeError("tuple_get requires exactly two arguments")
                if not isinstance(node.args[0], ir.IntLiteral):
                    raise TypeError("The first argument to tuple_get must be a literal int")
                idx = node.args[0].value
                tup = self.visit(node.args[1], constraints=constraints, symtypes=symtypes)
                kind = Var.fresh()
                elem = Var.fresh()
                size = Var.fresh()
                val = Val(kind, PartialTupleVar.fresh(((idx, elem),)), size)
                constraints.add((tup, val))
                return Val(kind, elem, size)
            if node.fun.id == "shift":
                # note: we just ignore the offsets
                it = Val(Iterator())
                return Fun(Tuple((it,)), it)

        fun = self.visit(node.fun, constraints=constraints, symtypes=symtypes)
        args = Tuple(tuple(self.visit(node.args, constraints=constraints, symtypes=symtypes)))
        ret = Var.fresh()
        constraints.add((fun, Fun(args, ret)))
        return ret


def rename(s, t):
    def r(x):
        if x == s:
            return t
        if isinstance(x, ValVar):
            return ValVar(x.idx, r(x.kind), r(x.dtype), r(x.size))
        if isinstance(x, (Var, Column, Scalar, Primitive, Value, Iterator)):
            return x
        if isinstance(x, Val):
            return Val(r(x.kind), r(x.dtype), r(x.size))
        if isinstance(x, Fun):
            return Fun(r(x.args), r(x.ret))
        if isinstance(x, FunVar):
            return FunVar(x.idx, r(x.args), r(x.ret))
        if isinstance(x, Tuple):
            return Tuple(tuple(r(e) for e in x.elems))
        if isinstance(x, PartialTupleVar):
            return PartialTupleVar(x.idx, tuple((i, r(e)) for i, e in x.elems))
        if isinstance(x, PrefixTupleVar):
            prefix = r(x.prefix)
            others = r(x.others)
            if isinstance(others, Tuple):
                return Tuple((prefix,) + others.elems)
            return PrefixTupleVar(x.idx, r(x.prefix), r(x.others))
        if isinstance(x, ValTupleVar):
            kind = r(x.kind)
            dtypes = r(x.dtypes)
            size = r(x.size)
            if isinstance(dtypes, Tuple):
                return Tuple(tuple(Val(kind, d, size) for d in dtypes.elems))
            return ValTupleVar(x.idx, kind, dtypes, size)
        if isinstance(x, tuple):
            return tuple(r(xi) for xi in x)
        if isinstance(x, set):
            return {r(xi) for xi in x}
        raise AssertionError()

    return r


def fv(x):
    if isinstance(x, Var):
        return {x}
    if isinstance(x, ValVar):
        return {x} | fv(x.kind) | fv(x.dtype) | fv(x.size)
    if isinstance(x, FunVar):
        return {x} | fv(x.args) | fv(x.ret)
    if isinstance(x, ValTupleVar):
        return {x} | fv(x.kind) | fv(x.dtypes) | fv(x.size)
    if isinstance(x, Val):
        return fv(x.kind) | fv(x.dtype) | fv(x.size)
    if isinstance(x, Fun):
        return fv(x.args) | fv(x.ret)
    if isinstance(x, Tuple):
        return set().union(*(fv(elem) for elem in x.elems))
    if isinstance(x, (Column, Scalar, Primitive, Value, Iterator)):
        return set()
    raise AssertionError()


def unify(dtype, constraints):
    while constraints:
        s, t = constraints.pop()
        if s == t:
            continue
        elif isinstance(s, Var) and s not in fv(t):
            r = rename(s, t)
            dtype = r(dtype)
            constraints = r(constraints)
        elif isinstance(t, Var) and t not in fv(s):
            r = rename(t, s)
            dtype = r(dtype)
            constraints = r(constraints)
        elif isinstance(s, Fun) and isinstance(t, Fun):
            constraints.add((s.args, t.args))
            constraints.add((s.ret, t.ret))
        elif isinstance(s, Val) and isinstance(t, Val):
            constraints.add((s.kind, t.kind))
            constraints.add((s.dtype, t.dtype))
            constraints.add((s.size, t.size))
        elif isinstance(s, ValVar) and isinstance(t, ValVar):
            assert s not in fv(t) and t not in fv(s)
            r = rename(s, t)
            dtype = r(dtype)
            constraints = r(constraints)
            constraints.add((s.kind, t.kind))
            constraints.add((s.dtype, t.dtype))
            constraints.add((s.size, t.size))
        elif isinstance(s, Tuple) and isinstance(t, Tuple):
            if len(s.elems) != len(t.elems):
                raise TypeError(f"Can not satisfy constraint {s} = {t}")
            for c in zip(s.elems, t.elems):
                constraints.add(c)
        elif isinstance(s, FunVar) and isinstance(t, Fun):
            assert s not in fv(t)
            r = rename(s, t)
            dtype = r(dtype)
            constraints = r(constraints)
            constraints.add((s.args, t.args))
            constraints.add((s.ret, t.ret))
        elif isinstance(s, PartialTupleVar) and isinstance(t, Tuple):
            for i, x in s.elems:
                constraints.add((x, t.elems[i]))
        elif isinstance(s, Tuple) and isinstance(t, PartialTupleVar):
            for i, x in t.elems:
                constraints.add((x, s.elems[i]))
        elif isinstance(s, PrefixTupleVar) and isinstance(t, Tuple):
            assert s not in fv(t)
            r = rename(s, t)
            dtype = r(dtype)
            constraints = r(constraints)
            constraints.add((s.prefix, t.elems[0]))
            constraints.add((s.others, Tuple(t.elems[1:])))
        elif isinstance(s, ValTupleVar) and isinstance(t, Tuple):
            assert s not in fv(t)
            r = rename(s, t)
            dtype = r(dtype)
            constraints = r(constraints)
            s_expanded = Tuple(tuple(Val(s.kind, Var.fresh(), s.size) for _ in t.elems))
            constraints.add((s.dtypes, Tuple(tuple(e.dtype for e in s_expanded.elems))))
            constraints.add((s_expanded, t))
        else:
            raise AssertionError()

    return dtype


def reindex_vars(dtype):
    index_map = dict()

    def r(x):
        if isinstance(x, DType):
            values = {f.name: r(getattr(x, f.name)) for f in fields(x)}
            if isinstance(x, VarMixin):
                values["idx"] = index_map.setdefault(x.idx, len(index_map))
            return type(x)(**values)
        if isinstance(x, tuple):
            return tuple(r(xi) for xi in x)
        assert isinstance(x, (str, int))
        return x

    return r(dtype)


def infer(expr, symtypes=None):
    if symtypes is None:
        symtypes = dict()
    constraints = set()
    dtype = TypeInferrer().visit(expr, constraints=constraints, symtypes=symtypes)
    unified = unify(dtype, constraints)
    return reindex_vars(unified)
