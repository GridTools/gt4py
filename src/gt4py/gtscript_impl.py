import collections
import types

import numpy as np


class _FieldDescriptor:
    def __init__(self, dtype, axes):
        if isinstance(dtype, str):
            self.dtype = dtype
        else:
            if dtype not in _VALID_DATA_TYPES:
                raise ValueError("Invalid data type descriptor")
            self.dtype = np.dtype(dtype)
        self.axes = axes if isinstance(axes, collections.abc.Collection) else [axes]

    def __repr__(self):
        return f"_FieldDescriptor(dtype={repr(self.dtype)}, axes={repr(self.axes)})"

    def __str__(self):
        return f"Field<{str(self.dtype)}, [{', '.join(str(ax) for ax in self.axes)}]>"


class _FieldDescriptorMaker:
    def __getitem__(self, dtype_and_axes):
        from gt4py.gtscript import IJK

        if isinstance(dtype_and_axes, collections.abc.Collection) and not isinstance(
            dtype_and_axes, str
        ):
            dtype, axes = dtype_and_axes
        else:
            dtype, axes = [dtype_and_axes, IJK]
        return _FieldDescriptor(dtype, axes)


_VALID_DATA_TYPES = (bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64)


class _Axis:
    def __init__(self, name: str):
        assert name
        self.name = name

    def __repr__(self):
        return f"_Axis(name={self.name})"

    def __str__(self):
        return self.name


class _SequenceDescriptor:
    def __init__(self, dtype, length):
        self.dtype = dtype
        self.length = length


class _SequenceDescriptorMaker:
    def __getitem__(self, dtype, length=None):
        return dtype, length


class _ComputationContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def _set_arg_dtypes(definition, dtypes):
    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    for arg, value in annotations.items():
        if isinstance(value, _FieldDescriptor) and isinstance(value.dtype, str):
            if value.dtype in dtypes:
                annotations[arg] = _FieldDescriptor(dtypes[value.dtype], value.axes)
            else:
                raise ValueError(f"Missing '{value.dtype}' dtype definition for arg '{arg}'")
        elif isinstance(value, str):
            if value in dtypes:
                annotations[arg] = dtypes[value]
            else:
                raise ValueError(f"Missing '{value}' dtype definition for arg '{arg}'")

    return definition
