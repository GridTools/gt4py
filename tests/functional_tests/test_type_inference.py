from functional import type_inference as ti


def test_renamer():
    class Foo(ti.Type):
        bar: ti.Type
        baz: ti.Type

    class Bar(ti.Type):
        ...

    r = ti._Renamer()
    actual = [
        (
            ti._Box(value=Foo(bar=ti.TypeVar(idx=0), baz=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=0)),
        )
    ]
    src = ti.TypeVar(idx=0)
    dst = ti.TypeVar(idx=1)
    for s, t in actual:
        r.register(s)
        r.register(t)
    r.register(src)
    r.register(dst)
    r.rename(src, dst)
    expected = [
        (
            ti._Box(value=Foo(bar=ti.TypeVar(idx=1), baz=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=1)),
        )
    ]
    assert actual == expected


def test_custom_type_inference():
    class Fun(ti.Type):
        arg: ti.Type
        ret: ti.Type

    class Basic(ti.Type):
        name: str

    v = [ti.TypeVar(idx=i) for i in range(5)]
    constraints = {
        (v[0], Fun(arg=v[1], ret=v[2])),
        (Fun(arg=v[0], ret=v[3]), v[4]),
        (Basic(name="int"), v[1]),
        (v[1], v[2]),
    }
    dtype = v[4]

    expected = Fun(arg=Fun(arg=Basic(name="int"), ret=Basic(name="int")), ret=ti.TypeVar(idx=0))

    actual = ti.reindex_vars(ti.unify(dtype, constraints))
    assert actual == expected
