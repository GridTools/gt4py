import functools
from dataclasses import dataclass, field, fields

from eve import NodeTranslator
from functional.iterator import ir


datatype = functools.partial(dataclass, frozen=True, slots=True)


class VarMixin:
    _counter = -1

    @staticmethod
    def fresh_index():
        VarMixin._counter += 1
        return VarMixin._counter

    @classmethod
    def fresh(cls, *args):
        return cls(VarMixin.fresh_index(), *args)


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
class PrefixTuple(DType):
    prefix: DType
    others: DType


@datatype
class Fun(DType):
    args: DType = field(default_factory=Var.fresh)
    ret: DType = field(default_factory=Var.fresh)


@datatype
class Val(DType):
    kind: DType = field(default_factory=Var.fresh)
    dtype: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


@datatype
class ValTuple(DType):
    kind: DType = field(default_factory=Var.fresh)
    dtypes: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


@datatype
class UniformValTupleVar(DType, VarMixin):
    idx: int
    kind: DType = field(default_factory=Var.fresh)
    dtype: DType = field(default_factory=Var.fresh)
    size: DType = field(default_factory=Var.fresh)


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


@datatype
class Closure(DType):
    output: DType
    inputs: DType


@datatype
class FunDef(DType):
    name: str
    fun: DType


@datatype
class Fencil(DType):
    name: str
    fundefs: DType
    params: DType


@datatype
class LetPolymorphic(DType):
    dtype: DType


def children(dtype):
    for f in fields(dtype):
        yield f.name, getattr(dtype, f.name)


def freshen(dtype):
    index_map = dict()

    def r(x):
        if isinstance(x, DType):
            values = {k: r(v) for k, v in children(x)}
            if isinstance(x, VarMixin):
                values["idx"] = index_map.setdefault(x.idx, VarMixin.fresh_index())
            return type(x)(**values)
        if isinstance(x, tuple):
            return tuple(r(xi) for xi in x)
        assert isinstance(x, (str, int))
        return x

    res = r(dtype)
    return res


BOOL_DTYPE = Primitive("bool")  # type: ignore [call-arg]
INT_DTYPE = Primitive("int")  # type: ignore [call-arg]
FLOAT_DTYPE = Primitive("float")  # type: ignore [call-arg]
AXIS_DTYPE = Primitive("axis")  # type: ignore [call-arg]
NAMED_RANGE_DTYPE = Primitive("named_range")  # type: ignore [call-arg]
DOMAIN_DTYPE = Primitive("domain")  # type: ignore [call-arg]


class TypeInferrer(NodeTranslator):
    def visit_SymRef(self, node, *, constraints, symtypes):
        if node.id in symtypes:
            res = symtypes[node.id]
            if isinstance(res, LetPolymorphic):
                return freshen(res.dtype)
            return res
        if node.id == "deref":
            dtype = Var.fresh()
            size = Var.fresh()
            return Fun(Tuple((Val(Iterator(), dtype, size),)), Val(Value(), dtype, size))
        if node.id == "can_deref":
            dtype = Var.fresh()
            size = Var.fresh()
            return Fun(Tuple((Val(Iterator(), dtype, size),)), Val(Value(), BOOL_DTYPE, size))
        if node.id in ("plus", "minus", "multiplies", "divides"):
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
            size = Var.fresh()
            args = ValTuple(Iterator(), Var.fresh(), size)
            dtype = Var.fresh()
            stencil_ret = Val(Value(), dtype, size)
            lifted_ret = Val(Iterator(), dtype, size)
            return Fun(Tuple((Fun(args, stencil_ret),)), Fun(args, lifted_ret))
        if node.id == "reduce":
            dtypes = Var.fresh()
            size = Var.fresh()
            acc = Val(Value(), Var.fresh(), size)
            f_args = PrefixTuple(acc, ValTuple(Value(), dtypes, size))
            ret_args = ValTuple(Iterator(), dtypes, size)
            f = Fun(f_args, acc)
            ret = Fun(ret_args, acc)
            return Fun(Tuple((f, acc)), ret)
        if node.id == "scan":
            dtypes = Var.fresh()
            fwd = Val(Value(), BOOL_DTYPE, Scalar())
            acc = Val(Value(), Var.fresh(), Scalar())
            f_args = PrefixTuple(acc, ValTuple(Iterator(), dtypes, Scalar()))
            ret_args = ValTuple(Iterator(), dtypes, Column())
            f = Fun(f_args, acc)
            ret = Fun(ret_args, Val(Value(), acc.dtype, Column()))
            return Fun(Tuple((f, fwd, acc)), ret)
        if node.id == "domain":
            args = UniformValTupleVar.fresh(Value(), NAMED_RANGE_DTYPE, Scalar())
            ret = Val(Value(), DOMAIN_DTYPE, Scalar())
            return Fun(args, ret)
        if node.id == "named_range":
            args = Tuple(
                (
                    Val(Value(), AXIS_DTYPE, Scalar()),
                    Val(Value(), INT_DTYPE, Scalar()),
                    Val(Value(), INT_DTYPE, Scalar()),
                )
            )
            ret = Val(Value(), NAMED_RANGE_DTYPE, Scalar())
            return Fun(args, ret)

        assert node.id not in ir.BUILTINS
        return Var.fresh()

    def visit_Literal(self, node, *, constraints, symtypes):
        return Val(Value(), Primitive(node.type))

    def visit_AxisLiteral(self, node, *, constraints, symtypes):
        return Val(Value(), AXIS_DTYPE, Scalar())

    def visit_OffsetLiteral(self, node, *, constraints, symtypes):
        return Var.fresh()

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
                if not isinstance(node.args[0], ir.Literal) or node.args[0].type != "int":
                    raise TypeError("The first argument to tuple_get must be a literal int")
                idx = int(node.args[0].value)
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

    def visit_FunctionDefinition(self, node, *, constraints, symtypes):
        if node.id in symtypes:
            raise TypeError(f"Multiple definitions of symbol {node.id}")

        fun = self.visit(
            ir.Lambda(params=node.params, expr=node.expr),
            constraints=constraints,
            symtypes=symtypes,
        )
        constraints.add((fun, Fun()))
        return FunDef(node.id, fun)

    def visit_StencilClosure(self, node, *, constraints, symtypes):
        domain = self.visit(node.domain, constraints=constraints, symtypes=symtypes)
        stencil = self.visit(node.stencil, constraints=constraints, symtypes=symtypes)
        output = self.visit(node.output, constraints=constraints, symtypes=symtypes)
        inputs = Tuple(tuple(self.visit(node.inputs, constraints=constraints, symtypes=symtypes)))
        output_dtype = Var.fresh()
        constraints.add((domain, Val(Value(), Primitive("domain"), Scalar())))
        constraints.add((output, Val(Iterator(), output_dtype, Column())))
        constraints.add((stencil, Fun(inputs, Val(Value(), output_dtype, Column()))))
        return Closure(output, inputs)

    def visit_FencilDefinition(self, node, *, constraints, symtypes):
        def funtypes():
            ftypes = []
            fmap = dict()
            # TODO: order by dependencies? Or is this done before?
            for f in node.function_definitions:
                c = constraints.copy()
                f = self.visit(f, constraints=c, symtypes=symtypes | fmap)
                f = unify(f, c)
                ftypes.append(f)
                fmap[f.name] = LetPolymorphic(f.fun)
            return Tuple(tuple(ftypes)), fmap

        params = {p.id: Var.fresh() for p in node.params}
        self.visit(
            node.closures, constraints=constraints, symtypes=symtypes | funtypes()[1] | params
        )
        return Fencil(node.id, funtypes()[0], Tuple(tuple(params[p.id] for p in node.params)))


def rename(s, t):
    def r(x):
        if x == s:
            return t
        if isinstance(x, DType):
            res = type(x)(**{k: r(v) for k, v in children(x)})
            if isinstance(res, ValTuple) and isinstance(res.dtypes, Tuple):
                # convert a ValTuple to a Tuple if it is fully resolved
                return Tuple(tuple(Val(res.kind, d, res.size) for d in res.dtypes.elems))
            if isinstance(res, PrefixTuple) and isinstance(res.others, Tuple):
                # convert a PrefixTuple to a Tuple if it is fully resolved
                return Tuple((res.prefix,) + res.others.elems)
            return res
        if isinstance(x, (tuple, set)):
            return type(x)(r(v) for v in x)
        assert isinstance(x, (str, int))
        return x

    return r


def free_variables(x):
    if isinstance(x, DType):
        res = set().union(*(free_variables(v) for _, v in children(x)))
        if isinstance(x, VarMixin):
            res.add(x)
        return res
    if isinstance(x, tuple):
        return set().union(*(free_variables(v) for v in x))
    assert isinstance(x, (str, int))
    return set()


def handle_constraint(constraint, dtype, constraints):  # noqa: C901
    s, t = constraint
    if s == t:
        return dtype, constraints

    if isinstance(s, Var):
        assert s not in free_variables(t)
        r = rename(s, t)
        dtype = r(dtype)
        constraints = r(constraints)
        return dtype, constraints

    if isinstance(s, Fun) and isinstance(t, Fun):
        constraints.add((s.args, t.args))
        constraints.add((s.ret, t.ret))
        return dtype, constraints

    if isinstance(s, Val) and isinstance(t, Val):
        constraints.add((s.kind, t.kind))
        constraints.add((s.dtype, t.dtype))
        constraints.add((s.size, t.size))
        return dtype, constraints

    if isinstance(s, Tuple) and isinstance(t, Tuple):
        if len(s.elems) != len(t.elems):
            raise TypeError(f"Can not satisfy constraint {s} = {t}")
        for c in zip(s.elems, t.elems):
            constraints.add(c)
        return dtype, constraints

    if isinstance(s, PartialTupleVar) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        for i, x in s.elems:
            constraints.add((x, t.elems[i]))
        return dtype, constraints

    if isinstance(s, PartialTupleVar) and isinstance(t, PartialTupleVar):
        assert s not in free_variables(t) and t not in free_variables(s)
        se = dict(s.elems)
        te = dict(t.elems)
        for i in set(se) & set(te):
            constraints.add((se[i], te[i]))
        combined = PartialTupleVar.fresh(tuple((se | te).items()))
        r = rename(s, combined)
        dtype = r(dtype)
        constraints = r(constraints)
        r = rename(t, combined)
        dtype = r(dtype)
        constraints = r(constraints)
        return dtype, constraints

    if isinstance(s, PrefixTuple) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        constraints.add((s.prefix, t.elems[0]))
        constraints.add((s.others, Tuple(t.elems[1:])))
        return dtype, constraints

    if isinstance(s, PrefixTuple) and isinstance(t, PrefixTuple):
        assert s not in free_variables(t) and t not in free_variables(s)
        constraints.add((s.prefix, t.prefix))
        constraints.add((s.others, t.others))
        return dtype, constraints

    if isinstance(s, ValTuple) and isinstance(t, Tuple):
        s_expanded = Tuple(tuple(Val(s.kind, Var.fresh(), s.size) for _ in t.elems))
        constraints.add((s.dtypes, Tuple(tuple(e.dtype for e in s_expanded.elems))))
        constraints.add((s_expanded, t))
        return dtype, constraints

    if isinstance(s, ValTuple) and isinstance(t, ValTuple):
        assert s not in free_variables(t) and t not in free_variables(s)
        constraints.add((s.kind, t.kind))
        constraints.add((s.dtypes, t.dtypes))
        constraints.add((s.size, t.size))
        return dtype, constraints

    if isinstance(s, UniformValTupleVar) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        r = rename(s, t)
        dtype = r(dtype)
        constraints = r(constraints)
        elem_dtype = Val(s.kind, s.dtype, s.size)
        for e in t.elems:
            constraints.add((e, elem_dtype))
        return dtype, constraints

    if isinstance(s, UniformValTupleVar) and isinstance(t, UniformValTupleVar):
        constraints.add((s.kind, t.kind))
        constraints.add((s.dtype, t.dtype))
        constraints.add((s.size, t.size))
        return dtype, constraints


def unify(dtype, constraints):
    while constraints:
        c = constraints.pop()
        r = handle_constraint(c, dtype, constraints)
        if not r:
            r = handle_constraint(c[::-1], dtype, constraints)

        if not r:
            raise TypeError(f"Can not satisfy constraint: {c[0]} ≡ {c[1]}")
        dtype, constraints = r

    return dtype


def reindex_vars(dtype):
    index_map = dict()

    def r(x):
        if isinstance(x, DType):
            values = {k: r(v) for k, v in children(x)}
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


def pretty_str(x):  # noqa: C901
    def subscript(i):
        return "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(i))

    def superscript(i):
        return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(i))

    def fmt_size(size):
        if size == Column():
            return "ᶜ"
        if size == Scalar():
            return "ˢ"
        assert isinstance(size, Var)
        return superscript(size.idx)

    def fmt_dtype(kind, dtype_str):
        if kind == Value():
            return dtype_str
        if kind == Iterator():
            return "It[" + dtype_str + "]"
        assert isinstance(kind, Var)
        return "ItOrVal" + subscript(kind.idx) + "[" + dtype_str + "]"

    if isinstance(x, Tuple):
        return "(" + ", ".join(pretty_str(e) for e in x.elems) + ")"
    if isinstance(x, PartialTupleVar):
        s = ""
        if x.elems:
            e = dict(x.elems)
            for i in range(max(e) + 1):
                s += (pretty_str(e[i]) if i in e else "…") + ", "
        return "(" + s + "…)" + subscript(x.idx)
    if isinstance(x, PrefixTuple):
        return pretty_str(x.prefix) + ":" + pretty_str(x.others)
    if isinstance(x, Fun):
        return pretty_str(x.args) + " → " + pretty_str(x.ret)
    if isinstance(x, Val):
        return fmt_dtype(x.kind, pretty_str(x.dtype) + fmt_size(x.size))
    if isinstance(x, Primitive):
        return x.name
    if isinstance(x, FunDef):
        return x.name + " :: " + pretty_str(x.fun)
    if isinstance(x, Closure):
        return pretty_str(x.inputs) + " ⇒ " + pretty_str(x.output)
    if isinstance(x, Fencil):
        return (
            "{"
            + "".join(pretty_str(f) + ", " for f in x.fundefs.elems)
            + x.name
            + pretty_str(x.params)
            + "}"
        )
    if isinstance(x, ValTuple):
        assert isinstance(x.dtypes, Var)
        return "(" + fmt_dtype(x.kind, "T" + fmt_size(x.size)) + ", …)" + subscript(x.dtypes.idx)
    if isinstance(x, VarMixin):
        return "T" + subscript(x.idx)
    raise AssertionError()
