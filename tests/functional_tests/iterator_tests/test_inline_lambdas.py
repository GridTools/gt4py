import functional.iterator.ir as itir
from functional.iterator.transforms.inline_lambdas import InlineLambdas


def test_simple():
    literal = itir.Literal(value="1", type="int")

    result = InlineLambdas.apply(itir.FunCall(fun=itir.Lambda(params=[], expr=literal), args=[]))
    assert result == literal
