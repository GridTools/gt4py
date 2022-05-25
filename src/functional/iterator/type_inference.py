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
class DType(eve.Node, unsafe_hash=True):  # type: ignore[call-arg]
    def __str__(self) -> str:
        return pformat(self)


class Var(DType, VarMixin):
    idx: int


class Tuple(DType):
    elems: tuple[DType, ...]


class PartialTupleVar(DType, VarMixin):
    idx: int
    elem_indices: tuple[int, ...]
    elem_values: tuple[DType, ...]


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
    fun: Fun


class Fencil(DType):
    name: str
    fundefs: tuple[DType, ...]
    params: tuple[DType, ...]


class LetPolymorphic(DType):
    dtype: DType


class Box(eve.Node, unsafe_hash=True):  # type: ignore[call-arg]
    value: DType

    def __str__(self):
        return "«" + str(self.value) + "»"


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
                val = Val(
                    kind=kind,
                    dtype=PartialTupleVar.fresh(elem_indices=(idx,), elem_values=(elem,)),
                    size=size,
                )
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
        ftypes = []
        fmap = dict()
        for f in node.function_definitions:
            c = set()
            f = self.visit(f, constraints=c, symtypes=symtypes | fmap)
            f = unify(f, c)
            ftypes.append(f)
            fmap[f.name] = LetPolymorphic(dtype=f.fun)

        params = {p.id: Var.fresh() for p in node.params}
        self.visit(node.closures, constraints=constraints, symtypes=symtypes | fmap | params)
        return Fencil(
            name=node.id,
            fundefs=tuple(ftypes),
            params=tuple(params[p.id] for p in node.params),
        )


class _Rename(eve.ReusingNodeTranslator):
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


class _Dedup(eve.ReusingNodeTranslator):
    def visit(self, node, *, memo):
        node = super().visit(node, memo=memo)
        return memo.setdefault(node, node)


class Renamer:
    def __init__(self):
        self.parents = dict()

    def register(self, dtype):
        def collect_parents(node):
            for field, child in node.iter_children_items():
                if isinstance(child, DType):
                    self.parents.setdefault(child, []).append((node, field, None))
                    collect_parents(child)
                elif isinstance(child, tuple):
                    for i, c in enumerate(child):
                        if isinstance(c, DType):
                            self.parents.setdefault(c, []).append((node, field, i))
                            collect_parents(c)
                else:
                    assert isinstance(child, (int, str))

        collect_parents(dtype)

    def rename(self, src, dst):
        nodes = self.parents.pop(src, None)
        if not nodes:
            return

        dst_parents = self.parents.setdefault(dst, [])
        follow_ups = []
        for node, field, index in nodes:
            if isinstance(node, ValTuple) and field == "dtypes" and isinstance(dst, Tuple):
                tup = Tuple(
                    elems=tuple(Val(kind=node.kind, dtype=d, size=node.size) for d in dst.elems)
                )
                follow_ups.append((node, tup))
            elif isinstance(node, PrefixTuple) and field == "others" and isinstance(dst, Tuple):
                tup = Tuple(elems=(node.prefix,) + dst.elems)
                follow_ups.append((node, tup))
            else:
                popped = self.parents.pop(node, None)
                if index is None:
                    setattr(node, field, dst)
                else:
                    field_list = list(getattr(node, field))
                    field_list[index] = dst
                    setattr(node, field, tuple(field_list))

                dst_parents.append((node, field, index))
                if popped:
                    self.parents[node] = popped

        for s, d in follow_ups:
            self.register(d)
            self.rename(s, d)


def handle_constraint(constraint, constraints, renamer):  # noqa: C901
    s, t = constraint
    s = s.value
    t = t.value
    if s == t:
        return True

    def rename(x, y):
        renamer.register(x)
        renamer.register(y)
        renamer.rename(x, y)

    def add_constraint(x, y):
        x = Box(value=x)
        y = Box(value=y)
        renamer.register(x)
        renamer.register(y)
        constraints.append((x, y))

    if isinstance(s, Var):
        assert s not in free_variables(t)
        rename(s, t)
        return True

    if isinstance(s, Fun) and isinstance(t, Fun):
        add_constraint(s.args, t.args)
        add_constraint(s.ret, t.ret)
        return True

    if isinstance(s, Val) and isinstance(t, Val):
        add_constraint(s.kind, t.kind)
        add_constraint(s.dtype, t.dtype)
        add_constraint(s.size, t.size)
        return True

    if isinstance(s, Tuple) and isinstance(t, Tuple):
        if len(s.elems) != len(t.elems):
            raise TypeError(f"Can not satisfy constraint {s} = {t}")
        for lhs, rhs in zip(s.elems, t.elems):
            add_constraint(lhs, rhs)
        return True

    if isinstance(s, PartialTupleVar) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        for i, x in zip(s.elem_indices, s.elem_values):
            add_constraint(x, t.elems[i])
        return True

    if isinstance(s, PartialTupleVar) and isinstance(t, PartialTupleVar):
        assert s not in free_variables(t) and t not in free_variables(s)
        se = dict(s.elems)
        te = dict(t.elems)
        for i in set(se) & set(te):
            add_constraint(se[i], te[i])
        elems = se | te
        combined = PartialTupleVar.fresh(
            elem_indices=tuple(elems.keys()), elem_values=tuple(elems.values())
        )
        rename(s, combined)
        rename(t, combined)
        return True

    if isinstance(s, PrefixTuple) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        add_constraint(s.prefix, t.elems[0])
        add_constraint(s.others, Tuple(elems=t.elems[1:]))
        return True

    if isinstance(s, PrefixTuple) and isinstance(t, PrefixTuple):
        assert s not in free_variables(t) and t not in free_variables(s)
        add_constraint(s.prefix, t.prefix)
        add_constraint(s.others, t.others)
        return True

    if isinstance(s, ValTuple) and isinstance(t, Tuple):
        s_expanded = Tuple(
            elems=tuple(Val(kind=s.kind, dtype=Var.fresh(), size=s.size) for _ in t.elems)
        )
        add_constraint(s.dtypes, Tuple(elems=tuple(e.dtype for e in s_expanded.elems)))
        add_constraint(s_expanded, t)
        return True

    if isinstance(s, ValTuple) and isinstance(t, ValTuple):
        assert s not in free_variables(t) and t not in free_variables(s)
        add_constraint(s.kind, t.kind)
        add_constraint(s.dtypes, t.dtypes)
        add_constraint(s.size, t.size)
        return True

    if isinstance(s, UniformValTupleVar) and isinstance(t, Tuple):
        assert s not in free_variables(t)
        rename(s, t)
        elem_dtype = Val(kind=s.kind, dtype=s.dtype, size=s.size)
        for e in t.elems:
            add_constraint(e, elem_dtype)
        return True

    if isinstance(s, UniformValTupleVar) and isinstance(t, UniformValTupleVar):
        add_constraint(s.kind, t.kind)
        add_constraint(s.dtype, t.dtype)
        add_constraint(s.size, t.size)
        return True

    return False


def unify(dtype, constraints):
    memo = dict()
    dtype = _Dedup().visit(dtype, memo=memo)
    constraints = {_Dedup().visit(c, memo=memo) for c in constraints}
    del memo

    renamer = Renamer()
    dtype = Box(value=dtype)
    renamer.register(dtype)
    constraints = [(Box(value=s), Box(value=t)) for s, t in constraints]
    for s, t in constraints:
        renamer.register(s)
        renamer.register(t)

    while constraints:
        c = constraints.pop()
        handled = handle_constraint(c, constraints, renamer)
        if not handled:
            handled = handle_constraint(c[::-1], constraints, renamer)

        if not handled:
            raise TypeError(f"Can not satisfy constraint: {c[0].value} ≡ {c[1].value}")

    return dtype.value


class _ReindexVars(eve.ReusingNodeTranslator):
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


class PrettyPrinter(eve.ReusingNodeTranslator):
    @staticmethod
    def _subscript(i):
        return "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(i))

    @staticmethod
    def _superscript(i):
        return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(i))

    def _fmt_size(self, size):
        if size == Column():
            return "ᶜ"
        if size == Scalar():
            return "ˢ"
        assert isinstance(size, Var)
        return self._superscript(size.idx)

    def _fmt_dtype(self, kind, dtype_str):
        if kind == Value():
            return dtype_str
        if kind == Iterator():
            return "It[" + dtype_str + "]"
        assert isinstance(kind, Var)
        return "ItOrVal" + self._subscript(kind.idx) + "[" + dtype_str + "]"

    def visit_Tuple(self, node):
        return "(" + ", ".join(self.visit(e) for e in node.elems) + ")"

    def visit_PartialTupleVar(self, node):
        s = ""
        if node.elem_indices:
            e = dict(zip(node.elem_indices, node.elem_values))
            for i in range(max(e) + 1):
                s += (self.visit(e[i]) if i in e else "_") + ", "
        return "(" + s + "…)" + self._subscript(node.idx)

    def visit_PrefixTuple(self, node):
        return self.visit(node.prefix) + ":" + self.visit(node.others)

    def visit_Fun(self, node):
        return self.visit(node.args) + " → " + self.visit(node.ret)

    def visit_Val(self, node):
        return self._fmt_dtype(node.kind, self.visit(node.dtype) + self._fmt_size(node.size))

    def visit_Primitive(self, node):
        return node.name

    def visit_FunDef(self, node):
        return node.name + " :: " + self.visit(node.fun)

    def visit_Closure(self, node):
        return self.visit(node.inputs) + " ⇒ " + self.visit(node.output)

    def visit_Fencil(self, node):
        return (
            "{"
            + "".join(self.visit(f) + ", " for f in node.fundefs)
            + node.name
            + "("
            + ", ".join(self.visit(p) for p in node.params)
            + ")}"
        )

    def visit_ValTuple(self, node):
        if not isinstance(node.dtypes, Var):
            return self.visit_DType(node)
        return (
            "("
            + self._fmt_dtype(node.kind, "T" + self._fmt_size(node.size))
            + ", …)"
            + self._subscript(node.dtypes.idx)
        )

    def visit_UniformValTupleVar(self, node):
        return (
            "("
            + self.visit(Val(kind=node.kind, dtype=node.dtype, size=node.size))
            + " × n"
            + self._subscript(node.idx)
            + ")"
        )

    def visit_Var(self, node):
        return "T" + self._subscript(node.idx)

    def visit_DType(self, node):
        return (
            node.__class__.__name__
            + "("
            + ", ".join(f"{k}={v}" for k, v in node.iter_children_items())
            + ")"
        )


pformat = PrettyPrinter().visit


def pprint(x: DType):
    print(pformat(x))
