import functools
from dataclasses import dataclass

from eve import NodeTranslator
from functional.iterator import ir


datatype = functools.partial(dataclass, frozen=True, slots=True)


@datatype
class DType:
    ...


@datatype
class Primitive(DType):
    name: str


@datatype
class TypeVar(DType):
    idx: int

    _counter = -1

    @classmethod
    def fresh(cls):
        cls._counter += 1
        return cls(cls._counter)


@datatype
class Tuple(DType):
    elements: tuple[DType, ...]


@datatype
class PartialTupleVar(DType):
    idx: int
    elements: tuple[tuple[int, DType], ...]

    _counter = -1

    @classmethod
    def fresh(cls, *args):
        cls._counter += 1
        return cls(cls._counter, args)


def rename(s, t):
    def impl(x):
        if x == s:
            return t
        if isinstance(x, (TypeVar, Primitive)):
            return x
        if isinstance(x, Tuple):
            return Tuple(tuple(impl(v) for v in x.elements))
        if isinstance(x, PartialTupleVar):
            return PartialTupleVar(x.idx, tuple((i, impl(v)) for i, v in x.elements))
        if isinstance(x, tuple):
            return tuple(impl(v) for v in x)
        if isinstance(x, dict):
            return {k: impl(v) for k, v in x.items()}
        if isinstance(x, set):
            return {impl(v) for v in x}
        raise AssertionError(f"Unexpected value {x}")

    return impl


def unify(dtypes, constraints):
    while constraints:
        s, t = constraints.pop()
        if s == t:
            continue
        if isinstance(s, TypeVar) or isinstance(t, TypeVar):
            r = rename(s, t) if isinstance(s, TypeVar) else rename(t, s)
            constraints = r(constraints)
            dtypes = r(dtypes)
        elif isinstance(s, Tuple) and isinstance(t, PartialTupleVar):
            for i, x in t.elements:
                constraints.add((s.elements[i], x))
        elif isinstance(s, PartialTupleVar) and isinstance(t, Tuple):
            for i, x in s.elements:
                constraints.add((t.elements[i], x))
        elif isinstance(s, PartialTupleVar) and isinstance(t, PartialTupleVar):
            se = dict(s.elements)
            te = dict(t.elements)
            for i in set(se) & set(te):
                constraints.add((se[i], te[i]))
            combined = PartialTupleVar.fresh(*(se | te).items())
            r = rename(s, combined)
            constraints = r(constraints)
            dtypes = r(dtypes)
            r = rename(t, combined)
            constraints = r(constraints)
            dtypes = r(dtypes)
        else:
            raise TypeError(f"Type unification failed, can not resolve {s} = {t}")

    vmap = dict()

    def compress(v):
        if isinstance(v, TypeVar):
            return vmap.setdefault(v.idx, len(vmap))
        if isinstance(v, Primitive):
            return v.name
        if isinstance(v, Tuple):
            return tuple(compress(vi) for vi in v.elements)
        if isinstance(v, PartialTupleVar):
            e = dict(v.elements)
            return tuple(compress(e.get(i, TypeVar.fresh())) for i in range(max(e) + 1))
        return v

    return {k: compress(v) for k, v in dtypes.items()}


BOOL_TYPE = Primitive("bool")  # type: ignore [call-arg]
INT_TYPE = Primitive("int")  # type: ignore [call-arg]
FLOAT_TYPE = Primitive("float")  # type: ignore [call-arg]


class CommonTypeDeduction(NodeTranslator):
    def visit_SymRef(self, node, *, constraints, symtypes):
        if node.id in ir.BUILTINS:
            raise TypeError("Can not deduce unapplied builtin type")
        return symtypes[node.id]

    def visit_BoolLiteral(self, node, **kwargs):
        return BOOL_TYPE

    def visit_IntLiteral(self, node, **kwargs):
        return INT_TYPE

    def visit_FloatLiteral(self, node, **kwargs):
        return FLOAT_TYPE

    def visit_FunCall(self, node, *, constraints, symtypes):  # noqa: C901
        if isinstance(node.fun, ir.Lambda):
            argtypes = {
                p.id: v
                for p, v in zip(
                    node.fun.params,
                    self.visit(node.args, constraints=constraints, symtypes=symtypes),
                )
            }

            return self.visit(node.fun.expr, constraints=constraints, symtypes=symtypes | argtypes)
        elif isinstance(node.fun, ir.SymRef):
            if node.fun.id == "deref":
                return self.visit(node.args[0], constraints=constraints, symtypes=symtypes)
            elif node.fun.id in ("plus", "minus", "multiplies", "divides"):
                x, y = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                constraints.add((x, y))
                return x
            elif node.fun.id in ("eq", "greater", "less"):
                x, y = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                constraints.add((x, y))
                return BOOL_TYPE
            elif node.fun.id in ("and_", "or_"):
                x, y = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                constraints.add((x, BOOL_TYPE))
                constraints.add((y, BOOL_TYPE))
                return BOOL_TYPE
            elif node.fun.id == "not_":
                x = self.visit(node.args[0], constraints=constraints, symtypes=symtypes)
                constraints.add((x, BOOL_TYPE))
                return BOOL_TYPE
            elif node.fun.id == "if_":
                c, x, y = self.visit(node.args, constraints=constraints, symtypes=symtypes)
                constraints.add((c, BOOL_TYPE))
                constraints.add((x, y))
                return x
            elif node.fun.id == "make_tuple":
                return Tuple(
                    tuple(self.visit(node.args, constraints=constraints, symtypes=symtypes))
                )
            elif node.fun.id == "tuple_get":
                if not isinstance(node.args[0], ir.IntLiteral):
                    raise TypeError("Tuple indices must be literal")
                tup = self.visit(node.args[1], constraints=constraints, symtypes=symtypes)
                res = TypeVar.fresh()
                ptup = PartialTupleVar.fresh((node.args[0].value, res))
                constraints.add((tup, ptup))
                return res
        elif isinstance(node.fun, ir.FunCall) and isinstance(node.fun.fun, ir.SymRef):
            if node.fun.fun.id == "shift":
                return self.visit(node.args[0], constraints=constraints, symtypes=symtypes)
            elif node.fun.fun.id == "scan":
                mock_call = ir.FunCall(fun=node.fun.args[0], args=[node.fun.args[2]] + node.args)
                return self.visit(mock_call, constraints=constraints, symtypes=symtypes)
            elif node.fun.fun.id == "lift":
                mock_call = ir.FunCall(fun=node.fun.args[0], args=node.args)
                return self.visit(mock_call, constraints=constraints, symtypes=symtypes)

        raise AssertionError(f"Unhandled function call: {node}")

    def visit_StencilClosure(self, node, *, constraints, symtypes):
        mock_call = ir.FunCall(fun=node.stencil, args=node.inputs)
        out = self.visit(mock_call, constraints=constraints, symtypes=symtypes)
        constraints.add((symtypes[node.output.id], out))

    def visit_FencilDefinition(self, node):
        symtypes = {param.id: TypeVar.fresh() for param in node.params}
        constraints = set()
        self.visit(node.closures, constraints=constraints, symtypes=symtypes)
        symtypes = unify(symtypes, constraints)
        return tuple(symtypes[param.id] for param in node.params)

    def visit_Program(self, node):
        if node.function_definitions:
            raise TypeError("Can only handle inlined functions")
        return {f.id: self.visit(f) for f in node.fencil_definitions}

    @classmethod
    def apply(cls, node):
        return cls().visit(node)
