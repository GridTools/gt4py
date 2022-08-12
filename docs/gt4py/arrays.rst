===============================
Allocation and Array Interfaces
===============================

GT4Py does not provide its own data container class, but supports established python standards for exposing
N-dimensional buffers. There is a minimalistic interface allowing to specify the correspondence of buffer dimensions
to the semantic dimensions assumed in the stencil. This correspondence does not necessarily need to be specified since
the stencils specify a default ordering.

GT4Py provides utilities to allocate buffers that have optimal layout and alignment for a given backend.

In this document, we describe the interfaces for
* supported buffer interfaces
* exposing dimension labels and the behavior for default values
* performance-optimal allocation

----------
Interfaces
----------

Stencil Calls
-------------

Supported Buffer Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user is free to choose a buffer interface (or multiple) as long as it is supported by ``numpy.asarray`` in case a
CPU backend is chosen or ``cupy.asarray`` for a GPU backend respectively. If multiple buffer interfaces are implementedm
the provided information needs to agree otherwise the behaviour is undefined. Similarly the backend is also free to
choose what buffer interface to use in order to retrieve the required information (e.g. pointer, strides, etc.) In
particular, we support the following interfaces to expose a buffer:

* `__array_interface__ <https://omz-software.com/pythonista/numpy/reference/arrays.interface.html>`_ (CPU backends)
* `__cuda_array_interface__ <https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html>`_ (GPU backends)
* `python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_ (CPU backends)

Internally, gt4py uses the utilities `gt4py.utils.as_numpy` and `gt4py.utils.as_cupy` to retrieve the buffers. GT4Py
developers are advised to always use those utilities as to guarantee support across gt4py as the supported interfaces
are extended.

Dimension Mapping
^^^^^^^^^^^^^^^^^

The user can optionally implement a ``__gt_dims__`` attribute in the object implementing any of the supported buffer
interfaces. As a fallback if neither is specified the dimensions given in the annotations (by means of
``gtscript.Field``) are assumed. The returned object should be a tuple of strings labeling the dimensions in index
order.

Valid dimension strings are ``"I"``, ``"J"``, ``"K"`` as well as decimal string representations of integer
numbers to denote data dimensions.

Developers are advised to use the utility ``gt4py.utils.get_dims(storage, annotation)``,
which implements this lookup.

Note: Support for xarray can be added manually by the user by means of the mechanism described
`here <https://xarray.pydata.org/en/stable/internals/extending-xarray.html>`_.

Default Origin
^^^^^^^^^^^^^^

A buffer object can optionally implement the ``__gt_origin__`` attribute which is used as the origin value unless
overwritten by the ``origin`` keyword argument to the stencil call.



Allocation
----------

For the performance-optimal allocation and initialization of arrays to be used in GT4Py, we provide the following set of
functions which closely resemble their NumPy counterparts (meaning of the common parameters is explained below).

The return type is either a ``numpy.ndarray`` or a `cupy.ndarray`, for CPU and GPU backends, respectively.

:code:`empty(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> ndarray`
    Allocate an array with uninitialized (undefined) values.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.

:code:`zeros(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate an array with values initialized to 0.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        For common keyword-only arguments, please see below.

:code:`ones(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate an array with values initialized to 1.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.


:code:`full(shape: Sequence[int], fill_value: Number, dtype=np.float64, **kwargs) -> Storage`
    Allocate an array with values initialized to the scalar given in :code:`fill_value`.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`fill_value: Number`. The number to which the storage is initialized.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.

:code:`from_array(data: array_like = None, device_data: array_like = None, *, dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Used to allocate an array with values initialized to those of a given array.

    Parameters:
        + :code:`data: array_like`. The original array from which the storage is initialized.

        + :code:`device_data: array_like`. The original array in case copying to a gpu buffer is
          desired. The same buffer could also be passed through `data` in that case, however this
          parameter is here to provide the same interface like the :code:`as_storage` function.

        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization.


Optional Keyword-Only Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, these **optional** keyword-only parameters are accepted:

:code:`aligned_index: Sequence[int]`
    The index of the grid point to which the memory is aligned. Note that this only partly takes the
    role of the deprecated :code:`default_origin` parameter, since it does not imply anything about the
    origin or domain when passed to a stencil. It defaults to the lower indices of the
    :code:`halo` parameter. (See :code:`__gt_origin__` interface instead.)

:code:`dimensions: Optional[Sequence[str]]`
    Sequence indicating the semantic meaning of the dimensions of this storage. This is used to
    determine the default layout for the storage. Currently supported will be :code:`"I"`,
    :code:`"J"`, :code:`"K"` and additional dimensions as string representations of integers,
    starting at :code:`"0"`. (This information is not retained in the resulting array, and needs to be specified instead
    with the :code:`__gt_dims__` interface. )

:code:`layout: Optional[Sequence[int]]`
    A permutation of integers in :code:`[0 .. ndim-1]`. It indicates the order of strides in
    decreasing order. I.e. :code:`0` indicates that the stride in that dimension is the largest, while the
    largest entry in the layout sequence corresponds to the dimension with the smallest stride, which
    typically is contiguous in memory.

    Default values as indicated by the :code:`backend` parameter depend on the dimensions. E.g.
    if :code:`backend` is any of the compiled GridTools backends, the default value is defined
    according to the semantic meaning of each dimension. For example for the :code:`"gt:cpu_kfirst"`
    backend, the smallest stride is always in the K dimension, independently of which index
    corresponds to the K dimension.
