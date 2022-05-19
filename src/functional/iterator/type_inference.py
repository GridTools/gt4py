import eve
from eve.utils import noninstantiable
from functional.iterator import ir


class VarMixin:
    _counter = -1

    @staticmethod
    def fresh_index():
        VarMixin._counter += 1
        return VarMixin._counter

    @classmethod
    def fresh(cls, **kwargs):
        return cls(idx=VarMixin.fresh_index(), **kwargs)


@noninstantiable
class DType(eve.FrozenNode):
    ...


class Var(DType, VarMixin):
    idx: int


class Tuple(DType):
    elems: tuple[DType, ...]


class PartialTupleVar(DType, VarMixin):
    idx: int
    elems: tuple[tuple[int, DType], ...]


class PrefixTuple(DType):
    prefix: DType
    others: DType


class Fun(DType):
    args: DType = eve.field(default_factory=Var.fresh)
    ret: DType = eve.field(default_factory=Var.fresh)


class Val(DType):
    kind: DType = eve.field(default_factory=Var.fresh)
    dtype: DType = eve.field(default_factory=Var.fresh)
    size: DType = eve.field(default_factory=Var.fresh)


class ValTuple(DType):
    kind: DType = eve.field(default_factory=Var.fresh)
    dtypes: DType = eve.field(default_factory=Var.fresh)
    size: DType = eve.field(default_factory=Var.fresh)


class UniformValTupleVar(DType, VarMixin):
    idx: int
    kind: DType = eve.field(default_factory=Var.fresh)
    dtype: DType = eve.field(default_factory=Var.fresh)
    size: DType = eve.field(default_factory=Var.fresh)


class Column(DType):
    ...


class Scalar(DType):
    ...


class Primitive(DType):
    name: str


class Value(DType):
    ...


class Iterator(DType):
    ...


class Closure(DType):
    output: DType
    inputs: DType


class FunDef(DType):
    name: str
    fun: DType


class Fencil(DType):
    name: str
    fundefs: DType
    params: DType


class LetPolymorphic(DType):
    dtype: DType


def freshen(dtype):
    def indexer(index_map):
        return VarMixin.fresh_index()

    index_map = dict()
    return _ReindexVars(indexer).visit(dtype, index_map=index_map)


BOOL_DTYPE = Primitive(name="bool")  # type: ignore [call-arg]
INT_DTYPE = Primitive(name="int")  # type: ignore [call-arg]
FLOAT_DTYPE = Primitive(name="float")  # type: ignore [call-arg]
AXIS_DTYPE = Primitive(name="axis")  # type: ignore [call-arg]
NAMED_RANGE_DTYPE = Primitive(name="named_range")  # type: ignore [call-arg]
DOMAIN_DTYPE = Primitive(name="domain")  # type: ignore [call-arg]


class TypeInferrer(eve.NodeTranslator):
    def visit_SymRef(self, node, *, constraints, symtypes):
        if node.id in symtypes:
            res = symtypes[node.id]
            if isinstance(res, LetPolymorphic):
                return freshen(res.dtype)
            return res
        if node.id == "deref":
            dtype = Var.fresh()
            size = Var.fresh()
            return Fun(
                args=Tuple(elems=(Val(kind=Iterator(), dtype=dtype, size=size),)),
                ret=Val(kind=Value(), dtype=dtype, size=size),
            )
        if node.id == "can_deref":
            dtype = Var.fresh()
            size = Var.fresh()
            return Fun(
                args=Tuple(elems=(Val(kind=Iterator(), dtype=dtype, size=size),)),
                ret=Val(kind=Value(), dtype=BOOL_DTYPE, size=size),
            )
        if node.id in ("plus", "minus", "multiplies", "divides"):
            v = Val(kind=Value())
            return Fun(args=Tuple(elems=(v, v)), ret=v)
        if node.id in ("eq", "less", "greater"):
            v = Val(kind=Value())
            ret = Val(kind=Value(), dtype=BOOL_DTYPE, size=v.size)
            return Fun(args=Tuple(elems=(v, v)), ret=ret)
        if node.id == "not_":
            v = Val(kind=Value(), dtype=BOOL_DTYPE)
            return Fun(args=Tuple(elems=(v,)), ret=v)
        if node.id in ("and_", "or_"):
            v = Val(kind=Value(), dtype=BOOL_DTYPE)
            return Fun(args=Tuple(elems=(v, v)), ret=v)
        if node.id == "if_":
            v = Val(kind=Value())
            c = Val(kind=Value(), dtype=BOOL_DTYPE, size=v.size)
            return Fun(args=Tuple(elems=(c, v, v)), ret=v)
        if node.id == "lift":
            size = Var.fresh()
            args = ValTuple(kind=Iterator(), dtypes=Var.fresh(), size=size)
            dtype = Var.fresh()
            stencil_ret = Val(kind=Value(), dtype=dtype, size=size)
            lifted_ret = Val(kind=Iterator(), dtype=dtype, size=size)
            return Fun(
                args=Tuple(elems=(Fun(args=args, ret=stencil_ret),)),
                ret=Fun(args=args, ret=lifted_ret),
            )
        if node.id == "reduce":
            dtypes = Var.fresh()
            size = Var.fresh()
            acc = Val(kind=Value(), dtype=Var.fresh(), size=size)
            f_args = PrefixTuple(
                prefix=acc, others=ValTuple(kind=Value(), dtypes=dtypes, size=size)
            )
            ret_args = ValTuple(kind=Iterator(), dtypes=dtypes, size=size)
            f = Fun(args=f_args, ret=acc)
            ret = Fun(args=ret_args, ret=acc)
            return Fun(args=Tuple(elems=(f, acc)), ret=ret)
        if node.id == "scan":
            dtypes = Var.fresh()
            fwd = Val(kind=Value(), dtype=BOOL_DTYPE, size=Scalar())
            acc = Val(kind=Value(), dtype=Var.fresh(), size=Scalar())
            f_args = PrefixTuple(
                prefix=acc, others=ValTuple(kind=Iterator(), dtypes=dtypes, size=Scalar())
            )
            ret_args = ValTuple(kind=Iterator(), dtypes=dtypes, size=Column())
            f = Fun(args=f_args, ret=acc)
            ret = Fun(args=ret_args, ret=Val(kind=Value(), dtype=acc.dtype, size=Column()))
            return Fun(args=Tuple(elems=(f, fwd, acc)), ret=ret)
        if node.id == "domain":
            args = UniformValTupleVar.fresh(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar())
            ret = Val(kind=Value(), dtype=DOMAIN_DTYPE, size=Scalar())
            return Fun(args=args, ret=ret)
        if node.id == "named_range":
            args = Tuple(
                elems=(
                    Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar()),
                    Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
                    Val(kind=Value(), dtype=INT_DTYPE, size=Scalar()),
                )
            )
            ret = Val(kind=Value(), dtype=NAMED_RANGE_DTYPE, size=Scalar())
            return Fun(args=args, ret=ret)

        assert node.id not in ir.BUILTINS
        return Var.fresh()

    def visit_Literal(self, node, *, constraints, symtypes):
        return Val(kind=Value(), dtype=Primitive(name=node.type))

    def visit_AxisLiteral(self, node, *, constraints, symtypes):
        return Val(kind=Value(), dtype=AXIS_DTYPE, size=Scalar())

    def visit_OffsetLiteral(self, node, *, constraints, symtypes):
        return Var.fresh()

    def visit_Lambda(self, node, *, constraints, symtypes):
        ptypes = {p.id: Var.fresh() for p in node.params}
        ret = self.visit(node.expr, constraints=constraints, symtypes=symtypes | ptypes)
        return Fun(args=Tuple(elems=tuple(ptypes[p.id] for p in node.params)), ret=ret)

    def visit_FunCall(self, node, *, constraints, symtypes):
        if isinstance(node.fun, ir.SymRef):
            if node.fun.id == "make_tuple":
                argtypes = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                kind = Var.fresh()
                size = Var.fresh()
                dtype = Tuple(elems=tuple(Var.fresh() for _ in argtypes))
                for d, a in zip(dtype.elems, argtypes):
                    constraints.add((Val(kind=kind, dtype=d, size=size), a))
                return Val(kind=kind, dtype=dtype, size=size)
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
                val = Val(kind=kind, dtype=PartialTupleVar.fresh(elems=((idx, elem),)), size=size)
                constraints.add((tup, val))
                return Val(kind=kind, dtype=elem, size=size)
            if node.fun.id == "shift":
                # note: we just ignore the offsets
                it = Val(kind=Iterator())
                return Fun(args=Tuple(elems=(it,)), ret=it)

        fun = self.visit(node.fun, constraints=constraints, symtypes=symtypes)
        args = Tuple(elems=tuple(self.visit(node.args, constraints=constraints, symtypes=symtypes)))
        ret = Var.fresh()
        constraints.add((fun, Fun(args=args, ret=ret)))
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
        return FunDef(name=node.id, fun=fun)

    def visit_StencilClosure(self, node, *, constraints, symtypes):
        domain = self.visit(node.domain, constraints=constraints, symtypes=symtypes)
        stencil = self.visit(node.stencil, constraints=constraints, symtypes=symtypes)
        output = self.visit(node.output, constraints=constraints, symtypes=symtypes)
        inputs = Tuple(
            elems=tuple(self.visit(node.inputs, constraints=constraints, symtypes=symtypes))
        )
        output_dtype = Var.fresh()
        constraints.add((domain, Val(kind=Value(), dtype=Primitive(name="domain"), size=Scalar())))
        constraints.add((output, Val(kind=Iterator(), dtype=output_dtype, size=Column())))
        constraints.add(
            (stencil, Fun(args=inputs, ret=Val(kind=Value(), dtype=output_dtype, size=Column())))
        )
        return Closure(output=output, inputs=inputs)

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
                fmap[f.name] = LetPolymorphic(dtype=f.fun)
            return Tuple(elems=tuple(ftypes)), fmap

        params = {p.id: Var.fresh() for p in node.params}
        self.visit(
            node.closures, constraints=constraints, symtypes=symtypes | funtypes()[1] | params
        )
        return Fencil(
            name=node.id,
            fundefs=funtypes()[0],
            params=Tuple(elems=tuple(params[p.id] for p in node.params)),
        )


class _Rename(eve.NodeTranslator):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def __call__(self, node):
        return self.visit(node)

    def generic_visit(self, node):
        if node == self.src:
            return self.dst
        return super().generic_visit(node)

    def visit_ValTuple(self, node):
        node = self.generic_visit(node)
        assert isinstance(node, ValTuple)
        if isinstance(node.dtypes, Tuple):
            return Tuple(
                elems=tuple(Val(kind=node.kind, dtype=d, size=node.size) for d in node.dtypes.elems)
            )
        return node

    def visit_PrefixTuple(self, node):
        node = self.generic_visit(node)
        assert isinstance(node, PrefixTuple)
        if isinstance(node.others, Tuple):
            return Tuple(elems=(node.prefix,) + node.others.elems)
        return node


rename = _Rename


class _FreeVariables(eve.NodeVisitor):
    def visit_DType(self, node, *, free_variables):
        self.generic_visit(node, free_variables=free_variables)
        if isinstance(node, VarMixin):
            free_variables.add(node)


def free_variables(x):
    fv = set()
    _FreeVariables().visit(x, free_variables=fv)
    return fv


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
        combined = PartialTupleVar.fresh(elems=tuple((se | te).items()))
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
        constraints.add((s.others, Tuple(elems=t.elems[1:])))
        return dtype, constraints

    if isinstance(s, PrefixTuple) and isinstance(t, PrefixTuple):
        assert s not in free_variables(t) and t not in free_variables(s)
        constraints.add((s.prefix, t.prefix))
        constraints.add((s.others, t.others))
        return dtype, constraints

    if isinstance(s, ValTuple) and isinstance(t, Tuple):
        s_expanded = Tuple(
            elems=tuple(Val(kind=s.kind, dtype=Var.fresh(), size=s.size) for _ in t.elems)
        )
        constraints.add((s.dtypes, Tuple(elems=tuple(e.dtype for e in s_expanded.elems))))
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
        elem_dtype = Val(kind=s.kind, dtype=s.dtype, size=s.size)
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


class _ReindexVars(eve.NodeTranslator):
    def __init__(self, indexer):
        super().__init__()
        self.indexer = indexer

    def visit_DType(self, node, *, index_map):
        node = self.generic_visit(node, index_map=index_map)
        if isinstance(node, VarMixin):
            new_index = index_map.setdefault(node.idx, self.indexer(index_map))
            new_values = {
                k: (new_index if k == "idx" else v) for k, v in node.iter_children_items()
            }
            return type(node)(**new_values)
        return node


def reindex_vars(dtype):
    def indexer(index_map):
        return len(index_map)

    index_map = dict()
    return _ReindexVars(indexer).visit(dtype, index_map=index_map)


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
