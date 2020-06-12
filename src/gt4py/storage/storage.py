# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import collections
import numbers

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np

from . import utils as storage_utils
from .array_function_handlers import REGISTRY as array_function_registry
from .default_parameters import get_default_parameters
from gt4py import backend as gt_backend
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils


def empty(
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    gpu=False,
    *,
    managed=False,
):

    return storage(
        None,
        shape=shape,
        dtype=dtype,
        default_parameters=default_parameters,
        default_origin=default_origin,
        axes=axes,
        gpu=gpu,
        copy=False,
        managed=managed,
    )


def ones(
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    gpu=False,
    *,
    managed=False,
):
    return storage(
        data=1,
        shape=shape,
        dtype=dtype,
        default_parameters=default_parameters,
        default_origin=default_origin,
        axes=axes,
        gpu=gpu,
        copy=True,
        managed=managed,
    )


def zeros(
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    gpu=False,
    *,
    managed=False,
):
    return storage(
        data=0,
        shape=shape,
        dtype=dtype,
        default_parameters=default_parameters,
        default_origin=default_origin,
        axes=axes,
        gpu=gpu,
        copy=True,
        managed=managed,
    )


def full(
    value,
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    gpu=False,
    *,
    managed=False,
):
    return storage(
        data=value,
        shape=shape,
        dtype=dtype,
        default_parameters=default_parameters,
        default_origin=default_origin,
        axes=axes,
        gpu=gpu,
        copy=True,
        managed=managed,
    )


def asstorage(
    data,
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    *,
    managed=None,
    device_data=None,
):
    return storage(
        data,
        default_parameters=default_parameters,
        default_origin=default_origin,
        shape=shape,
        dtype=dtype,
        axes=axes,
        managed=managed,
        copy=False,
        device_data=device_data,
    )


def storage(
    data,
    *,
    default_parameters=None,
    default_origin=None,
    shape=None,
    dtype=None,
    axes=None,
    alignment=None,
    gpu=None,
    layout_map=None,
    copy=True,
    device_data=None,
    managed=None,
):
    return Storage(
        data,
        default_parameters=default_parameters,
        default_origin=default_origin,
        shape=shape,
        dtype=dtype,
        axes=axes,
        alignment=alignment,
        gpu=gpu,
        layout_map=layout_map,
        copy=copy,
        device_data=device_data,
        managed=managed,
    )


class Storage(abc.ABC, np.lib.mixins.NDArrayOperatorsMixin):

    SUPPORTED_UFUNCS = {
        # math operations
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.logaddexp,
        np.logaddexp2,
        np.true_divide,
        np.floor_divide,
        np.negative,
        np.positive,
        np.power,
        np.remainder,
        np.mod,
        np.fmod,
        np.divmod,
        np.absolute,
        np.fabs,
        np.rint,
        np.sign,
        np.heaviside,
        np.conj,  # note: complex types not supported
        np.conjugate,  # note: complex types not supported
        np.exp,
        np.exp2,
        np.log,
        np.log2,
        np.log10,
        np.expm1,
        np.log1p,
        np.sqrt,
        np.square,
        np.cbrt,
        np.reciprocal,
        np.gcd,
        np.lcm,
        # trigonometric functions
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.arctan2,
        np.hypot,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
        # bit-twiddling functions
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.invert,
        np.left_shift,
        np.right_shift,
        # comparison functions
        np.greater,
        np.greater_equal,
        np.less,
        np.less_equal,
        np.not_equal,
        np.equal,
        np.logical_and,
        np.logical_or,
        np.logical_xor,
        np.logical_not,
        np.maximum,
        np.minimum,
        np.fmax,
        np.fmin,
        # floating functions
        np.isfinite,
        np.isinf,
        np.isnan,
        # np.isnat,  # times not supported
        np.fabs,
        np.signbit,
        np.copysign,
        np.nextafter,
        np.spacing,
        np.modf,
        np.ldexp,
        np.frexp,
        np.fmod,
        np.floor,
        np.ceil,
        np.trunc,
    }

    def __new__(
        cls,
        data,
        *,
        default_parameters=None,
        default_origin=None,
        shape=None,
        dtype=None,
        axes=None,
        alignment=None,
        gpu=None,
        layout_map=None,
        copy=True,
        device_data=None,
        managed=None,
    ):
        # 1) for each parameter assert that type and value is valid
        if data is not None:
            if not isinstance(data, Storage):
                try:
                    np.asarray(data)
                except:
                    try:
                        cp.asarray(data)
                    except:
                        raise TypeError("'data' not understood as array")

        if shape is not None:
            if not gt_utils.is_iterable_of(shape, numbers.Integral):
                raise TypeError("'shape' must be an iterable of integers")
            elif not all(map(lambda x: x >= 0, shape)):
                raise ValueError("shape contains negative values")
            shape = tuple(int(s) for s in shape)
        if default_parameters is not None:
            if not isinstance(default_parameters, str):
                raise TypeError("'default_parameters' must be a string")
            elif not default_parameters in ["F", "C"] + gt_backend.REGISTRY.names:
                raise ValueError(
                    f"'default_parameters' must be in {['F','C']+gt_backend.REGISTRY.names}"
                )
        if dtype is not None:
            if isinstance(dtype, gt_ir.DataType):
                dtype = dtype.dtype
            else:
                try:
                    dtype = np.dtype(dtype)
                except:
                    raise TypeError("'dtype' not understood ")

        if axes is not None:
            if not gt_utils.is_iterable_of(
                axes, iterable_class=collections.abc.Sequence, item_class=str
            ):
                raise TypeError("'axes' must be a sequence of unique characters in 'IJK'")
            axes = "".join(axes)
            if (
                not len(axes) <= 3
                or not all([c in "IJK" for c in axes])
                or not len(set(axes)) == len(axes)
                or not sorted(axes) == list(axes)
            ):
                raise ValueError("'axes' must be a sub-sequence of 'IJK'")
            mask = tuple(c in axes for c in "IJK")
        else:
            mask = None

        if alignment is not None:
            if not isinstance(alignment, int):
                raise TypeError("'alignment' must be an integer")
            if not alignment > 0:
                raise ValueError("'alignment' must be positive")

        if gpu is not None:
            if not isinstance(gpu, bool):
                raise TypeError("'gpu' must be a boolean")

        if layout_map is not None:
            if not gt_utils.is_iterable_of(layout_map, int) and not callable(layout_map):
                if not callable(layout_map):
                    raise TypeError(
                        f"'layout_map' must either be an iterable of integers"
                        " or a callable returning such an iterable when given 'ndim'"
                    )

        if not isinstance(copy, bool):
            raise TypeError("'copy' must be a boolean")

        if device_data is not None:
            if not isinstance(device_data, Storage):
                try:
                    cp.asarray(device_data)
                except:
                    raise TypeError("'device_data' not understood as gpu array")

        if managed is not None:
            if not isinstance(managed, str) and managed != False:
                raise TypeError("'managed_memory' must be a string or 'False'")
            elif not managed in ["cuda", "gt4py", False]:
                raise ValueError('\'managed_memory\' must be in ["cuda", "gt4py", False]')

        # 2a) if data is storage, use those parameters
        if isinstance(data, Storage) or isinstance(device_data, Storage):
            template = data if isinstance(data, Storage) else device_data
            if gpu is None:
                gpu = template.gpu
            if alignment is None:
                alignment = template.alignment
            if layout_map is None:
                layout_map = template.layout_map
            if default_origin is None:
                default_origin = template.default_origin
        # 2b) if data is given, infer some more params
        if data is not None:
            if dtype is None:
                dtype = np.asarray(data).dtype
        # 2b) fill in default parameters.
        if default_parameters is not None:
            default_parameters = get_default_parameters(default_parameters)
            if gpu is None:
                gpu = default_parameters.gpu
            if alignment is None:
                alignment = default_parameters.alignment
            if layout_map is None:
                layout_map = default_parameters.layout_map

        # 3) determine data and/or device_data buffers
        if data is not None or device_data is not None:
            if gpu:
                if managed is not None:
                    pass
                else:
                    pass
            else:
                pass

        # 4) fill in missing parameters from given data/device_data
        if shape is None:
            if data is not None:
                shape = data.shape
            elif device_data is not None:
                shape = device_data.shape
        if mask is None:
            if isinstance(data, Storage):
                mask = data.mask
            elif isinstance(device_data, Storage):
                mask = data.mask

        if isinstance(data, Storage):
            ndim = data.ndim
        elif isinstance(device_data, Storage):
            ndim = device_data.ndim
        elif mask is not None:
            ndim = len(mask)
        elif shape is not None:
            ndim = len(shape)
        else:
            raise TypeError("not enough information to determine the number of dimensions")

        if layout_map is None:
            if isinstance(data, Storage):
                layout_map = data._layout_map
            elif isinstance(device_data, Storage):
                layout_map = device_data._layout_map

        elif not gt_utils.is_iterable_of(layout_map, item_class=int):
            assert callable(layout_map)
            layout_map = layout_map(ndim)
            if not gt_utils.is_iterable_of(layout_map, item_class=int):
                raise TypeError(
                    f"'layout_map' did not return an iterable of integers for ndim={ndim}"
                )

        if layout_map is not None and (
            not all(item >= 0 for item in layout_map)
            or not all(item < len(layout_map) for item in layout_map)
            or not len(set(layout_map)) == len(layout_map)
        ):
            raise ValueError(
                f"elements of layout map must be a permutation of (0, ..., len(layout_map))"
            )

        # 5a) assert consistency of parameters

        if mask is not None:
            if len(shape) == len(mask):
                shape = tuple(s for s, m in zip(shape, mask) if m)
        # 5b) if not copy: assert consistency with given buffer

        # 6) assert info is provided for all required parameters

        # 7) fill in missing parameters where a default is available
        assert isinstance(ndim, int) and ndim >= 0

        if layout_map is None:
            layout_map = tuple(range(ndim))
        if alignment is None:
            alignment = 1
        if mask is None:
            mask = (True,) * ndim

        #
        # if not copy:
        #     managed_memory = storage_utils.is_cuda_managed(data)
        #     if kwargs.get("managed_memory", managed_memory) != managed_memory:
        #         raise ValueError(
        #             f"tried to construct storage from existing array with 'managed_memory' set to"
        #             f"'{kwargs.get('managed_memory', managed_memory)}', but the provided data is"
        #             + ("" if managed_memory else " not")
        #             + " cuda managed memory"
        #         )
        # else:
        #     managed_memory = kwargs.get("managed_memory", None)
        #
        # if device is not None:
        #     if not isinstance(device, str):
        #         raise TypeError("device must be a string")
        #     if device not in ["cpu", "gpu"]:
        #         raise ValueError("device must be 'cpu' or 'gpu'")
        #     device = device
        # else:
        #     device = parameters.device
        #
        # if alignment is not None:
        #     if not isinstance(alignment, int):
        #         raise TypeError("expected 'alignment' to be of type 'int'")
        #     if not alignment > 0:
        #         raise ValueError("'alignment' must be positive")
        #     alignment = alignment
        # else:
        #     alignment = parameters.alignment
        #
        # if layout_map is not None:
        #     if callable(layout_map):
        #         try:
        #             layout_map = layout_map(ndim)
        #         except:
        #             layout_map = None
        #     if not gt_utils.is_iterable_of(layout_map, item_class=bool):
        #         raise TypeError(
        #             "'layout_map' must be a sequence of length ndim containing unique integers"
        #             "or a callable returning such an iterable when given ndim."
        #         )
        #     layout_map = storage_utils.normalize_layout_map(layout_map, ndim)
        # elif default_parameters is None and isinstance(data, Storage):
        #     layout_map = data.layout_map
        # else:
        #     layout_map = parameters.layout_map(ndim)
        #
        # if default_origin is None:
        #     default_origin = [0] * ndim
        # default_origin = storage_utils.normalize_default_origin(default_origin, mask)
        #
        # #######################################################################
        # if device == "gpu" and managed_memory is None:
        #     raise ValueError(
        #         "'managed_memory' must be a boolean if 'device' is 'gpu' and 'copy' is 'True'"
        #     )
        #
        # if cp is not None and isinstance(data, cp.ndarray):
        #     if isinstance(storage, GPUStorage) or isinstance(storage, ExplicitlySyncedGPUStorage):
        #         tmp = storage_utils.gpu_view(storage)
        #         tmp[...] = data
        #     else:
        #         storage[...] = cp.asnumpy(data)
        # else:
        #     storage[...] = data

        # if device is None:
        #     if default_parameters is None:
        #         raise ValueError("neither 'default_parameters' nor 'device' specified")
        #     device = gt_backend.from_name(default_parameters).storage_info["device"]

        # 7) determine storage type
        if gpu:
            if managed == "cuda":
                storage_t = CudaManagedGPUStorage
                kwargs = {}
            elif managed == "gt4py":
                storage_t = SoftwareManagedGPUStorage
                kwargs = {}
            else:
                storage_t = GPUStorage
                kwargs = {}
        else:
            storage_t = CPUStorage
            kwargs = {}

        assert gt_utils.is_iterable_of(shape, item_class=int, iterable_class=tuple)
        assert gt_utils.is_iterable_of(mask, item_class=bool, iterable_class=tuple)

        return storage_t.__new__(
            storage_t,
            array=data,
            shape=shape,
            dtype=dtype,
            default_origin=default_origin,
            mask=mask,
            layout_map=layout_map,
            alignment=alignment,
            copy=copy,
            **kwargs,
        )

    # def __init__(
    #     self,
    #     *,
    #     shape,
    #     dtype,
    #     data=None,
    #     default_origin=None,
    #     mask=None,
    #     default_parameters=None,
    #     alignment=None,
    #     device=None,
    #     layout_map=None,
    #     copy=True,
    # ):
    #     # """
    #     #
    #     # Parameters
    #     # ----------
    #     #
    #     # shape: tuple of ints
    #     #     the shape of the storage
    #     #
    #     # dtype: data type compatible with numpy dtypes
    #     #     supported are the floating point and integer dtypes of numpy
    #     #
    #     # default_parameters: string, backend identifier
    #     #     Currently possible: debug, numpy, gtx86, gtmc, gtcuda
    #     #     specifies according to which backend alignment, layout_map and device should be chosen if not otherwise
    #     #     specified.
    #     # default_origin: tuple of ints
    #     #     determines the point to which the storage memory address is aligned.
    #     #     for performance, this should be the coordinates of the most common origin
    #     #     at call time.
    #     #     when calling a stencil and no origin is specified, the default_origin is used.
    #     #
    #     # mask: list of booleans
    #     #     False entries indicate that the corresponding dimension is masked, i.e. the storage
    #     #     has reduced dimension and reading and writing from offsets along this axis acces the same element.
    #     # """
    #
    #     self._alignment = alignment
    #     self._layout_map = layout_map
    #     self._mask = mask
    #     self._default_origin = default_origin

    def all(*args, **kwargs):
        return np.all(*args, **kwargs)

    def any(*args, **kwargs):
        return np.any(*args, **kwargs)

    def min(*args, **kwargs):
        return np.min(*args, **kwargs)

    def max(*args, **kwargs):
        return np.max(*args, **kwargs)

    def transpose(*args, **kwargs):
        return np.transpose(*args, **kwargs)

    @property
    def mask(self):
        return tuple(self._mask)

    @property
    def axes(self):
        return "".join([c for m, c in zip(self._mask, "IJK") if m])

    @property
    def ndim(self):
        return sum(self._mask)

    @property
    def default_origin(self):
        return self._default_origin

    @property
    def alignment(self):
        return self._alignment

    @property
    def layout_map(self):
        return self._layout_map

    @property
    @abc.abstractmethod
    def shape(self):
        pass

    @property
    def _expanded_shape(self):
        return storage_utils.expand_shape(self.shape, self._mask)

    @property
    @abc.abstractmethod
    def strides(self):
        pass

    @property
    @abc.abstractmethod
    def dtype(self):
        pass

    @property
    def gpu(self):
        return False

    @property
    @abc.abstractmethod
    def _ptr(self):
        pass

    @abc.abstractmethod
    def __array__(self):
        pass

    @property
    @abc.abstractmethod
    def __array_interface__(self):
        pass

    def copy(self):
        res = storage(self, copy=True)
        return res

    def __deepcopy__(self, memo={}):
        res = storage(self, copy=True)
        return res

    @abc.abstractmethod
    def __array_function__(self, func, types, args, kwargs):
        pass

    @abc.abstractmethod
    def _forward_ufunc(
        self,
        ufunc,
        method,
        inputs,
        broadcastable_input_shapes,
        outputs,
        broadcastable_output_shape,
        kwargs,
    ):
        pass

    @abc.abstractmethod
    def _forward_setitem(self):
        pass

    @abc.abstractmethod
    def _forward_getitem(self):
        pass

    def _ufunc_out_types(self, ufunc, method, inputs, dtype, kwargs):
        kwargs = dict(kwargs)
        if "axis" in kwargs:
            kwargs.pop("axis")
        # outputs = kwargs.pop("out")
        # if not isinstance(outputs, tuple):
        #     outputs = (outputs,)

        input_dtypes = [
            np.dtype(inp.dtype) if hasattr(inp, "dtype") else np.dtype(type(inp)) for inp in inputs
        ]
        # output_dtypes = [np.dtype(outp.dtype) for outp in outputs]
        inputs = [t.type() for t in input_dtypes]
        # outputs = [t.type() for t in output_dtypes]
        # kwargs["out"] = tuple(outputs)
        try:
            out = np.ufunc.__dict__[method](ufunc, *inputs, **kwargs)
        except:
            # go on, s.t. the actual forwarding call raises
            return []

        out = out if isinstance(out, tuple) else (out,)
        return [o.dtype for o in out]

    def _calculate_broadcast_shape_and_mask(self, storages, expanded_array_shapes):
        broadcast_shape = [1] * 3
        broadcast_mask = [False] * 3
        for stor in storages:
            for i, (s, m) in enumerate(zip(stor._expanded_shape, stor.mask)):
                if m:
                    if broadcast_mask[i] and not broadcast_shape[i] == s:
                        raise ValueError
                    broadcast_shape[i] = s
                    broadcast_mask[i] = True
        for shape in expanded_array_shapes:
            for i, s in enumerate(shape):
                if s != 1:
                    if broadcast_mask[i] and not (broadcast_shape[i] == s):
                        raise ValueError
                    broadcast_shape[i] = s
                    broadcast_mask[i] = True
        return broadcast_shape, broadcast_mask

    def _broadcast_info(self, inputs, outputs):
        inputs = [
            inp if isinstance(inp, Storage) else storage_utils.asarray(inp) for inp in inputs
        ]
        expanded_input_shapes = [
            inp._expanded_shape if isinstance(inp, Storage) else inp.shape for inp in inputs
        ]  # only at the end of the function is it guaranteed that all shapes in this list are expanded

        input_storages = [inp for inp in inputs if isinstance(inp, Storage)]
        output_storages = [outp for outp in outputs if isinstance(outp, Storage)]
        storages = input_storages + output_storages

        ndim = 3

        expanded_input_array_shapes = [
            inp.shape for inp in inputs if (not isinstance(inp, Storage) and inp.ndim == ndim)
        ]
        expanded_output_array_shapes = [
            outp.shape for outp in outputs if (not isinstance(outp, Storage) and outp.ndim == ndim)
        ]

        # check if input arrays w/ only unmasked defined match input and
        for i, s in enumerate(expanded_input_shapes):
            if not len(s) == ndim:
                assert len(input_storages) <= 1
                if len(input_storages) == 1:
                    input_mask = input_storages[0].mask
                    if sum(input_mask) == len(s):
                        res = tuple(
                            s[sum(input_mask[: j + 1]) - 1] if m else 1
                            for j, m in enumerate(input_mask)
                        )
                        expanded_input_shapes[i] = res
                        expanded_input_array_shapes.append(res)

        broadcast_shape, broadcast_mask = self._calculate_broadcast_shape_and_mask(
            storages, expanded_input_array_shapes + expanded_output_array_shapes
        )

        # check if arrays w/ only unmasked defined match joint shape/mask
        # conflict: raise
        # match: expand
        # check if input arrays w/ only unmasked defined match input and
        for i, s in enumerate(expanded_input_shapes):
            if not len(s) == ndim:
                if sum(broadcast_mask) == len(s):
                    expanded_shape = storage_utils.expand_shape(s, broadcast_mask)
                    expanded_input_shapes[i] = expanded_shape
                elif s == ():
                    expanded_input_shapes[i] = (1,) * ndim
                else:
                    raise ValueError
        return expanded_input_shapes, broadcast_shape, broadcast_mask

    def _ufunc_broadcast_info(self, ufunc, method, inputs, kwargs, *, outputs):

        bcast_input_shapes, bcast_shape, bcast_mask = self._broadcast_info(
            inputs, outputs if outputs is not None else []
        )
        if method == "reduce":
            if len(bcast_shape) > 0:
                axis = kwargs.get("axis", (0,))
                if axis is None:
                    axis = tuple(range(len(bcast_shape)))
                for a in axis:
                    if a < len(bcast_shape):  # otherwise, ufunc forwarding will raise eventually
                        bcast_shape[a] = 1
                        bcast_mask[a] = False
        return bcast_input_shapes, bcast_shape, bcast_mask

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        kwargs = dict(kwargs)
        outputs = kwargs.get("out", None)
        dtype = kwargs.get("dtype", None)

        if outputs is not None:
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for outp in outputs:
                outp_is_buffer = True
                try:
                    tmp_asarray = np.asarray(outp)
                    assert (
                        gt_ir.nodes.DataType.from_dtype(tmp_asarray.dtype)
                        != gt_ir.nodes.DataType.INVALID
                    ), "output data type not supported"
                except AssertionError:
                    raise
                except Exception:
                    outp_is_buffer = False
                if not (outp_is_buffer or hasattr(outp, "__cuda_array_interface__")):
                    return NotImplemented

        if method not in ["__call__", "reduce"]:
            return NotImplemented
        if ufunc not in self.SUPPORTED_UFUNCS:
            return NotImplemented

        if method == "reduce":
            if ufunc.nin != 2:
                raise ValueError(
                    "reduce only supported for binary functions"
                )  # Numpy error string
            if ufunc.nout != 1:
                raise ValueError(
                    "reduce only supported for functions returning a single value"
                )  # Numpy error string
        else:
            if ufunc.nin != len(inputs):
                raise ValueError("invalid number of arguments")  # Numpy error string
            if outputs is not None and ufunc.nout != len(outputs):
                raise ValueError(
                    f"The 'out' tuple must have exactly {ufunc.nout} entries: one per ufunc output"
                )  # Numpy error string

        tmp_inputs = []
        for inp in inputs:
            inp_is_buffer = True
            try:
                tmp_asarray = np.asarray(inp)
                assert (
                    gt_ir.nodes.DataType.from_dtype(tmp_asarray.dtype)
                    != gt_ir.nodes.DataType.INVALID
                ), "input data type not supported"
            except AssertionError:
                raise
            except:
                inp_is_buffer = False
            if not (inp_is_buffer or hasattr(inp, "__cuda_array_interface__")):
                return NotImplemented

            expanded_input_shapes, expanded_output_shape, output_mask = self._ufunc_broadcast_info(
                ufunc, method, inputs, kwargs, outputs=outputs
            )

        if outputs is None:
            out_types = self._ufunc_out_types(ufunc, method, inputs, dtype, kwargs)
            # out_mask, out_shape, expanded_output_shape = self._ufunc_broadcast_info(
            #     ufunc, method, inputs, expanded_input_shapes, out
            # )
            out_shape = tuple(s for s, m in zip(expanded_output_shape, output_mask) if m)

            # determine the default origin as the dimension-wise minimum of all minima
            input_storages = [inp for inp in inputs if isinstance(inp, Storage)]

            ndim = 3

            out_origin = [0] * ndim
            for inp in inputs:
                if isinstance(inp, Storage):
                    expanded_input_origin = storage_utils.expand_shape(
                        inp.default_origin, inp.mask
                    )
                    for i, m in enumerate(inp.mask):
                        if m:
                            out_origin[i] = max(expanded_input_origin[i], out_origin[i])
            out_origin = tuple([o for i, o in enumerate(out_origin) if output_mask[i]])

            out_alignment = max(
                inp._alignment if isinstance(inp, Storage) else 1 for inp in inputs
            )

            out_layout_map = next(inp for inp in inputs if isinstance(inp, Storage))._layout_map

            # allocate
            storage_t = type(self)
            outputs = tuple(
                storage_t(
                    shape=out_shape,
                    dtype=t,
                    mask=output_mask,
                    alignment=out_alignment,
                    layout_map=out_layout_map,
                    default_origin=out_origin,
                )
                for t in out_types
            )

        return self._forward_ufunc(
            ufunc, method, inputs, expanded_input_shapes, outputs, expanded_output_shape, kwargs
        )

    def __setitem__(self, key, value):
        target = self.__getitem__(key)

        expanded_input_shapes, broadcast_shape, broadcast_mask = self._broadcast_info(
            inputs=[value], outputs=[target]
        )
        assert len(expanded_input_shapes) == 1
        target._forward_setitem(value, expanded_input_shapes[0], broadcast_shape)

    def __getitem__(self, key):
        if not isinstance(key, numbers.Integral) and not gt_utils.is_iterable_of(
            iterable_class=tuple, item_class=(slice, numbers.Integral)
        ):
            raise TypeError("Index type not supported.")
        if not isinstance(key, tuple):
            key = (key,)
        res = storage(self, copy=False)
        if key is Ellipsis:
            return res
        res._forward_getitem(key)

        key = key + (slice(None, None),) * (self.ndim - len(key))
        tmp_mask = list(res._mask)
        for i, m in enumerate(self.mask):
            if m:
                tmp_mask[i] = isinstance(key[sum(self.mask[:i])], slice)
        res._mask = tuple(tmp_mask)
        return res

    def __iconcat__(self, other):
        raise NotImplementedError("Concatenation of storages is not supported")


class CPUStorage(Storage):
    def __new__(
        cls,
        *,
        shape,
        dtype,
        array=None,
        default_parameters=None,
        default_origin=None,
        mask=None,
        alignment=None,
        layout_map=None,
        copy=False,
        **kwargs,
    ):
        self = super(Storage, cls).__new__(
            CPUStorage,
            # shape=shape,
            # dtype=dtype,
            # default_origin=default_origin,
            # default_parameters=default_parameters,
            # mask=mask,
            # alignment=alignment,
            # layout_map=layout_map,
        )
        self._alignment = alignment
        self._layout_map = layout_map
        self._mask = mask
        self._default_origin = default_origin

        reduced_layout_map = [None] * self.ndim
        for ctr, idx in enumerate(
            np.argsort(list(mp for mp, msk in zip(self._layout_map, self.mask) if msk))
        ):
            reduced_layout_map[idx] = ctr

        if array is None or copy:
            _, self._array = storage_utils.allocate_cpu(
                default_origin,
                shape,
                reduced_layout_map,
                dtype,
                self._alignment * np.dtype(dtype).itemsize,
            )
        else:
            self._array = np.asarray(array)

        if copy:
            self._array[...] = array
        return self

    __array_priority__ = 10

    @property
    def shape(self):
        return self._array.shape

    @property
    def strides(self):
        return self._array.strides

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def _ptr(self):
        return self._array.ctypes.data

    def __array__(self):
        return self._array.__array__()

    @property
    def __array_interface__(self):
        return self._array.__array_interface__

    def __deepcopy__(self, memo={}):
        res = storage(self, copy=True)
        return res

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ not in array_function_registry:
            raise NotImplementedError
        else:
            return array_function_registry[func.__name__](types, *args, **kwargs)

    def _forward_ufunc(
        self,
        ufunc,
        method,
        inputs,
        broadcastable_input_shapes,
        outputs,
        broadcastable_output_shape,
        kwargs,
    ):

        inp = tuple(x._array if isinstance(x, CPUStorage) else x for x in inputs)
        inp = tuple(
            x.reshape(s) if isinstance(x, np.ndarray) else x
            for x, s in zip(inp, broadcastable_input_shapes)
        )
        outp = tuple(o._array if isinstance(o, CPUStorage) else o for o in outputs)
        kwargs["out"] = tuple(x.reshape(broadcastable_output_shape) for x in outp)
        if "axis" in kwargs:
            axis = kwargs["axis"]
            if axis is None:
                axis_shape = ()
            else:
                axis_shape = tuple(
                    b for i, b in enumerate(broadcastable_output_shape) if i not in kwargs["axis"]
                )
            kwargs["out"] = tuple(x.reshape(axis_shape) for x in kwargs["out"])
        if len(kwargs["out"]) == 0:
            kwargs.pop("out")
        # copy of kwargs needed since ufunc removes tuple for single element "out" tuple
        result = getattr(ufunc, method)(*inp, **dict(kwargs))
        assert all(
            r is o
            for r, o in zip(result if isinstance(result, tuple) else (result,), kwargs["out"])
        ), "output is new array"
        return outputs[0] if ufunc.nout == 1 else outputs

    def _forward_setitem(self, value, expanded_input_shape, expanded_output_shape):
        value = np.array(value, copy=False).reshape(expanded_input_shape)
        target_array = self._array.reshape(expanded_output_shape)
        target_array.__setitem__(Ellipsis, value)

    def _forward_getitem(self, key):
        self._array = self._array[key]
