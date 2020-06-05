## using-dsl: gtscript
import numpy


@lazy_stencil()
def axpy(x: Field[numpy.float64], y: Field[numpy.float64], *, alpha: numpy.float64 = 1.0):
    # """Treat each vertical column of x, y as a vector and apply axpy to all in parallel."""
    with computation(PARALLEL), interval(...):
        y = x * alpha + y


@lazy_stencil(backend="numpy")
def dot(x: Field[numpy.float64], y: Field[numpy.float64], dot: Field[numpy.float64]):
    # """Treat each vertical column of x, y as a vector and store dot products in the last vertical layer of dot."""
    with computation(PARALLEL), interval(...):
        dot = x * y
    with computation(FORWARD), interval(1, None):
        dot = dot + dot[0, 0, -1]
