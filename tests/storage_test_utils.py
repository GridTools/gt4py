import numpy as np


try:
    import dace
except ImportError:
    dace = None


class ArrayWrapper:
    def __init__(self, array, **kwargs):
        self.array = array

    @property
    def __array_interface__(self):
        return self.array.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __descriptor__(self):
        return dace.data.create_datadescriptor(self.array)


class NdarraySubclassWrapper(np.ndarray, ArrayWrapper):
    """DaCe currently breaks for non-ndarrays at call time (works fine for compilation)."""

    def __new__(cls, array, **kwargs):
        return array.view(cls)

    def __init__(self, array, **kwargs):
        # skip ndarray init (already done.)
        super(np.ndarray, self).__init__(array=array, **kwargs)


class DimensionsWrapper(ArrayWrapper):
    def __init__(self, dimensions, **kwargs):
        self.__gt_dims__ = dimensions
        super().__init__(**kwargs)


class OriginWrapper(ArrayWrapper):
    def __init__(self, *, origin, **kwargs):
        self.__gt_origin__ = origin
        super().__init__(**kwargs)

    def __descriptor__(self):
        res = super().__descriptor__()
        res.__gt_origin__ = self.__gt_origin__
        return res


class NdarraySubclassOriginWrapper(NdarraySubclassWrapper, OriginWrapper):
    pass
