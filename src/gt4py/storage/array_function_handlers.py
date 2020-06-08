import numpy as np

REGISTRY = dict()


def register(handler):
    REGISTRY[handler.__name__] = handler
    return handler


def reduction_handler(op_ufunc, *, a, axis, out, **kwargs):
    dtype = np.bool
    if out is not None:
        dtype = out.dtype
        out = (out,)
    reduce_kwargs = dict(axis=axis, initial=op_ufunc.identity, where=True, dtype=dtype)
    if "keepdims" in kwargs:
        reduce_kwargs["keepdims"] = kwargs["keepdims"]

    return op_ufunc.reduce(a, **reduce_kwargs)


@register
def all(types, a, axis=None, out=None, **kwargs):
    return reduction_handler(np.logical_and, a=a, axis=axis, out=out, **kwargs)


@register
def any(types, a, axis=None, out=None, **kwargs):
    return reduction_handler(np.logical_or, a=a, axis=axis, out=out, **kwargs)


@register
def transpose(types, a, axes=None):
    if axes is not None and tuple(axes) == tuple(range(a.ndim)):
        return a[...]
    raise NotImplementedError
