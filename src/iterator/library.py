from iterator.builtins import reduce


def sum_(fun=None):
    if fun is None:
        return reduce(lambda a, b: a + b, 0)
    else:
        return reduce(lambda first, a, b: first + fun(a, b), 0)  # TODO tracing for *args


def dot(a, b):
    return reduce(lambda acc, a_n, c_n: acc + a_n * c_n, 0)(a, b)
