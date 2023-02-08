======================================================
GDP 3 — A New Storage Implementation using Duck Typing
======================================================

:Author: Linus Groner <linus.groner@cscs.ch>
:Author: Enrique G. Paredes <enrique.gonzalez@cscs.ch>
:Status: Declined
:Type: Feature
:Created: 08-05-2020
:Discussion PR: https://github.com/GridTools/gt4py/pull/28


Abstract
--------

We propose to replace the current storage implementation by a new `storage` class hierarchy
which does not inherit from NumPy `ndarray`. Instead we propose to support the also established
:code:`__array_interface__` and :code:`__cuda_array_interface__` interfaces. These new storage
classes shall also introduce the possibility to be constructed from externally allocated memory
buffers and provide more control over the underlying memory.

Further, we propose the :code:`__gt_data_interface__` attribute which can be added to any object to
provide the necessary information about how it can be used as a field in gt4py stencils or when
creating and interacting with storages. We also propose to accept the :code:`__array_interface__`
and :code:`__cuda_array_interface__` interfaces if these provide all necessary information.

Lastly, these storages shall focus on providing a mechanism that allows users to write code with
only minimal dependence on the backend while achieving optimal performance. Since this goal is
fundamentally difficult to achieve under some operations that we previously targeted such as
`universal functions`, we propose to remove these functionalities.

Motivation and Scope
--------------------

In the current state of GT4Py, we implemented storages as subclasses of NumPy :code:`ndarrays`.
Although this strategy is fully documented and supported by NumPy API, it presents some drawbacks
such as that one-sided changes to coupled host/device buffers managed by GT4Py storages (e.g.
:code:`ExplicitlySyncedGPUStorage` class) cannot be tracked reliably, resulting in validation errors
which are hard to find and fix.

The current implementation of GT4Py storages as :code:`ndarray` subclasses was needed to use
storages transparently with third-party frameworks relying on NumPy :code:`ndarray` implementation
details. Nowadays, however, most of the Python scientific ecosystem supports the more generic
interface known as the
`Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_, which can be
implemented by types to provide information about their use as a n-dimensional array. It is an
approach based on `duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_ to allow integration
of generic buffers that are not instances of a particular type in libraries. A similar interface for
GPU buffer is defined as the
`CUDA Array Interface <https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html>`_,
also with good and still increasing adaption in the ecosystem. Therefore, reimplementing
GT4Py storages based on these interfaces allows GT4Py to retain full control over the internal
behavior of complex operations, while keeping interoperability with third-party scientific
frameworks.

We propose to take this opportunity to improve the interaction with existing codes by
allowing the initialization of storages from external buffers without requiring a copying operation.
To use this feature, some additional information about the provided buffers needs to be specified to
both the :code:`__init__` method of the storages.

With the adaption of our storages to the generic array interfaces, we will also add support
for third-party types implementing said interfaces that can then be passed to the stencils directly.
While these interfaces provide all information needed to understand a buffer as an n-dimensional
array, this may not be enough for all use cases of GT4Py. For example, objects may hold buffers in
both the main memory and on a GPU. In this case, updating the respective other after changes to one
is a concern. Also, with coming developments to gt4py, the need to provide additional semantic
information with the buffers will arise. For example, the dimensions will evolve away from only the
:code:`"I"`, :code:`"J"` and :code:`"K"` axes which arise from indexing 3-dimensional fields on
cartesian grids.

To resolve these limitations of the array interfaces we propose to define another property, the
:code:`__gt_data_interface__`, which can be implemented by third-party objects. It shall summarize
the buffer infos of multiple devices while also adding information about the semantic meaning of
dimensions and function handles to interact with the object to address the concerns of data
synchronization between devices. If only default array interfaces are implemented, these shall
nevertheless work, with a well-defined default behavior.

Finally, without internally keeping information about the semantic meaning of dimensions, e.g. the
best layout and proper broadcasting for the resulting storage can not be determined. Further,
implementations would depend on the availability of a library implementing the operations for a
given device. We have already observed performance problems when using cupy on AMD hardware. For
future hardware, these libraries might be entirely unavailable. Therefore, we will not commit to
supporting such operations.

Backward Compatibility
----------------------

The implementation of this GDP breaks the integration with external libraries which require a NumPy
`ndarray` subclass. Further, we propose some API changes like renaming or repurposing of keyword
arguments and attributes. These changes are expected to break most of existing user codes. However,
existing GT4Py functionality will remain or be extended and thus updating codebases to the new
interface amounts mostly to updating attribute and keyword argument names.

The removal of universal functions will require codes relying on ufuncs to be re-written by
implementing these using third-party libraries.


Detailed Description
--------------------

Functionality
^^^^^^^^^^^^^

We chose to use internally NumPy and CuPy `ndarrays` to store CPU and GPU buffers, respectively,
since they are standard and reliable components of the Python ecosystem. We rely on those libraries
to implement part of the functionality, like :code:`__setitem__` and :code:`__getitem__`, by
forwarding the calls to the appropriate NumPy/CuPy function call. As a consequence, support for
`dtypes` and other functionality is restricted to the common denominator of CuPy and NumPy.

.. note:: In this document, references to NumPy and CuPy objects or functions use the standard
    shorted prefiex for these libraries, that is :code:`np.` for NumPy and :code:`cp.` for CuPy.

Data Interface
^^^^^^^^^^^^^^

Objects implementing the data interface have the attribute :code:`__gt_data_interface__`. This
attribute is a dictionary which maps `device identifiers` to a dictionary similar to those
defined as the :code:`__array_interface__` and the :code:`__cuda_array_interface__`.

A device identifier can be one of:

+ :code:`None` denoting a buffer in main memory
+ :code:`"gpu"` denoting a buffer on a GPU.

The mapped dictionary in turn must contain the following keys and mapped objects of pairs. Their
meaning is the same as in the NumPy :code:`__array_interface__` and the
:code:`__cuda_array_interface__`:

+ :code:`"shape": Tuple[int, ...]`
+ :code:`"typestr": str`
+ :code:`"data": Tuple[int, bool]`
+ :code:`"strides": Tuple[int, ...]`

In Addition, the following optional keys can be contained:

+ :code:`"acquire": Optional[Callable[[], Any]]` Is called on all objects that are passed to a
  stencil, before running computations. It can be used to trigger a copy to the respective device.
  If the key is not in the dictionary or if the value is :code:`None`, no action is taken.
+ :code:`"dims": Optional[Sequence[str]]]` Specifies the semantic dimensions to which the
  respective dimensions of the object correspond. Currently meaningful are :code:`"I"`,
  :code:`"J"`, :code:`"K"`.
+ :code:`"halo": Optional[Sequence[Tuple[int, int]]]` A tuple of length ndim with entries which are
  2-tuples of ints. Specifies the start and end boundary sizes on the respective dimension.
  At stencil call time, this property is used to infer the compute domain.
  :code:`"J"`, :code:`"K"`.
+ :code:`"release": Optional[Callable[[], Any]]` Is called on all objects that are passed to a
  stencil after all computations have completed. If the key is not in the dictionary or if the value
  is :code:`None`, no action is taken. We do not have the intention to use it in our own storage
  implementation and it is added here to complement the :code:`"acquire"` method.
+ :code:`"touch": Optional[Callable[[], Any]]` Is called on all objects for which the underlying
  memory has been changed after all computations have completed. If the key is not in the dictionary
  or if the value is :code:`None`, no action is taken.

Note that other entries can be contained in these buffer info dictionaries, but they will not have
any effect. It is therefore legal to forward the :code:`__array_interface__` or
:code:`__cuda_array_interface__` of NumPy and CuPy ndarrays, respectively.

If the passed object does not have the :code:`__gt_data_interface__` attribute, the
:code:`__array_interface__` and :code:`__cuda_array_interface__` attributes will be treated as
descriptions of main memory or gpu buffers, respectively.

Each backend is compiled for computation on either cpu or gpu. When calling the stencil, will use
the buffer on the same device as the computation is to be performed. If no such buffer is present,
but a buffer is present on the respective other device, the other buffer will be copied to a newly
allocated buffer on the compute device and copied back after successful completion. In the latter
case, a warning is printed, since these operations are typically expensive.

Default `xarray` Data Interface
===============================

For xarray :code:`DataArray` s, we propose to add a default accessor upon importing the root gt4py
module.
The behavior for a :code:`data_array` of type :code:`DataArray` shall be as follows:

1) If `data_array.data` implements the :code:`__gt_data_interface__`, then this is returned, while
   for each of the dictionaries per device, one of the following behaviors will apply:

   * If :code:`"dims"` is a key in the dictionary, an error is raised if it does not agree with
     :code:`data_array.dims`
   * Otherwise, the `"dims"` key is set to be `data_array.dims`.

2) If `data_array.data` does not implement the :code:`__gt_data_interface__`, the
   :code:`__array_interface__` and :code:`__cuda_array_interface__` properties of
   :code:`data_array.data` are used as interfaces for the :code:`None` and :code:`"gpu"` device
   keys, respectively. The :code:`"dims"` are then added based on :code:`data_array.dims` to each.

Users can still override this accessor and define their own behavior. In this case, xarray will
raise a warning when defining the accessor.

.. _constructors:

Storage Creation
^^^^^^^^^^^^^^^^

The :code:`Storage` base class is exposed in the API mainly to enable type checking. For the actual
creation and initialization of GT4Py storages we propose the following set of functions which
closely resemble their NumPy counterparts (meaning of the common parameters is explained below):

:code:`empty(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with uninitialized (undefined) values.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.


:code:`empty_like(data: Storage, dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with uninitialized (undefined) values, while choosing the not explicitly
    overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage`
          Not explicitly overridden parameters are chosen as the value used in this.
          :code:`Storage`

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

    The common keyword-only arguments can also be overridden. Please see below for their description.

    Note that :code:`shape` is not a parameter and can not be overridden.

:code:`zeros(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 0.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        For common keyword-only arguments, please see below.

:code:`zeros_like(data: Storage, dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 0, while choosing the not explicitly
    overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage`
          Not explicitly overridden parameters are chosen as the value used in this
          :code:`Storage`

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

    The common keyword-only arguments can also be overridden. Please see below for their
    description.

    Note that :code:`shape` is not a parameter and can not be overridden.


:code:`ones(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 1.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.

:code:`ones_like(data: Storage, dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 1, while choosing the not explicitly
    overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage`
          Not explicitly overridden parameters are chosen as the value used in this
          :code:`Storage`

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

    The common keyword-only arguments can also be overridden. Please see below for their
    description.

    Note that :code:`shape` is not a parameter and can not be overridden.


:code:`full(shape: Sequence[int], fill_value: Number, dtype=np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to the scalar given in :code:`fill_value`.

    Parameters:
        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions) with the
          shape of the storage.

        + :code:`fill_value: Number`. The number to which the storage is initialized.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

    For common keyword-only arguments, please see below.

:code:`full_like(shape: Sequence[int], fill_value: Number, dtype=np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to the scalar given in :code:`fill_value`, while
    choosing the not explicitly overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage` Not explicitly overridden parameters are chosen as the value used in
          this :code:`Storage`

        + :code:`fill_value: Number`. The number to which the storage is initialized.

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

    The common keyword-only arguments can also be overridden. Please see below for their description.

    Note that :code:`shape` is not a parameter and can not be overridden.

:code:`as_storage(data: array_like = None, device_data: array_like = None, *, sync_state: Storage.SyncState = None, **kwargs) -> Storage`
    Wrap an existing buffer in a GT4Py storage instance, without copying the buffer's contents.

    Parameters:
        + :code:`data: array_like`. The memory buffer or storage from which the storage is
          initialized.

        + :code:`device_data: array_like`. The device buffer or storage in case wrapping
          existing buffers on both the device and main memory is desired.

    Keyword-only parameters:
        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization. If
          the buffers have previously been extracted from a Storage, the :code:`SyncState` object
          must also be the one extracted from that same original Storage through the
          :code:`sync_state` attribute. For more details see :ref:`sync_state`.

        For common keyword-only arguments, please see below.

:code:`storage(data: array_like = None, device_data: array_like = None, *, dtype: dtype_like = np.float64, copy=True, **kwargs) -> Storage`
    Used to allocate a storage with values initialized to those of a given array. If the argument
    :code:`copy` is set to :code:`False`, the behavior is that of :code:`as_storage`.

    Parameters:
        + :code:`data: array_like`. The original array from which the storage is initialized.

        + :code:`device_data: array_like`. The original array in case copying to a gpu buffer is
          desired. The same buffer could also be passed through `data` in that case, however this
          parameter is here to provide the same interface like the :code:`as_storage` function.

        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization.

    Keyword-only parameters:
        + :code:`copy: bool`. Allocate a new buffer and initialize it with a copy of the data or
          wrap the existing buffer.

        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization.

        For common keyword-only arguments, please see below.

    If :code:`copy=False` and neither :code:`data` nor :code:`device_data` are provided, the other
    arguments are used to allocate an appropriate buffer without initialization (equivalent to call
    :code:`empty()`). If :code:`data` or :code:`device_data` is provided, the consistency of the
    parameters with the buffers is validated.

Optional Keyword-Only Parameters
================================

Additionally, these **optional** keyword-only parameters are accepted:

:code:`aligned_index: Sequence[int]`
    The index of the grid point to which the memory is aligned. Note that this only partly takes the
    role of the former :code:`default_origin` parameter, since it does not imply anything about the
    origin or domain when passed to a stencil. It defaults to the lower indices of the
    :code:`halo` parameter.

:code:`alignment_size: Optional[int]`
    The buffers are allocated such that :code:`mod(aligned_addr, alignment_size) == 0`, where
    :code:`aligned_addr` is the memory address of the grid point denoted by :code:`aligned_index`.

    It defaults to :code:`1`, which indicates no alignment.

:code:`defaults: Optional[str]`
    It can be used in the way of the current :code:`backend` parameter. For each backend, as well
    as for the keys :code:`"F"` and :code:`"C"` (equivalent to the same values in the :code:`order`
    parameter for NumPy allocation routines) a preset of suitable parameters is provided. Explicit
    definitions of additional parameters are possible and they override its default value from the
    preset.

:code:`device: Optional[str]`
    Indicates whether the storage should contain a buffer on an accelerator device. Currently it
    only accepts :code:`"gpu"` or :code:`None`. Defaults to :code:`None`.

:code:`dims: Optional[Sequence[str]`
    Sequence indicating the semantic meaning of the dimensions of this storage. This is used to
    determine the default layout for the storage. Currently supported will be :code:`"I"`,
    :code:`"J"`, :code:`"K"` and additional dimensions as string representations of integers,
    starting at :code:`"0"`.

:code:`halo: Optional[Sequence[Union[int, Tuple[int, int]]]`
    Sequence of length :code:`ndim` where each entry is either an :code:`int` or a 2-tuple
    of :code:`int` s. A sequence of integer numbers represent a symmetric halo with the specific
    size per dimension, while a sequence of 2-tuple specifies the start and end boundary sizes on
    the respective dimension, which can be used to denote asymmetric halos. It defaults to no halo,
    i.e. :code:`(0, 0, 0)`. (See also Section :ref:`domain_and_halo`)

:code:`layout: Optional[Sequence[int]]`
    A permutation of integers in :code:`[0 .. ndim-1]`. It indicates the order of strides in
    decreasing order. I.e. "0" indicates that the stride in that dimension is the largest, while the
    largest entry in the layout sequence corresponds to the dimension with the smallest stride, which
    typically is contiguous in memory.

    Default values as indicated by the :code:`defaults` parameter may depend on the dimensions. E.g.
    if :code:`defaults` is any of the compiled GridTools backends, the default value is defined
    according to the semantic meaning of each dimension. For example for the :code:`"gtx86"`
    backend, the smallest stride is always in the K dimension, independently of which index
    corresponds to the K dimension. On the other hand, we assume that if a storage is created from
    an existing FORTRAN array, the first index has the smallest stride, irrespective of its
    corresponding axis. I.e. the layout of a 3d storage is always :code:`(2, 1, 0)` for both IJK and
    KJI storages.

    .. list-table:: Default :code:`layout` parameter when given :code:`defaults` and :code:`dims`
       :header-rows: 1
       :stub-columns: 1

       * -
         - :code:`defaults="F"`
         - :code:`defaults="C"`
         - :code:`defaults="gtx86"`
         - :code:`defaults="gtcuda"`

       * - :code:`dims="IJK"`
         - :code:`layout=(2, 1, 0)`
         - :code:`layout=(0, 1, 2)`
         - :code:`layout=(0, 1, 2)`
         - :code:`layout=(2, 1, 0)`

       * - :code:`dims="KJI"`
         - :code:`layout=(2, 1, 0)`
         - :code:`layout=(0, 1, 2)`
         - :code:`layout=(2, 1, 0)`
         - :code:`layout=(0, 1, 2)`

    The rationale behind this is that in this way, storages allocated with :code:`defaults` set to a
    backend will always get optimal performance, while :code:`defaults` set to :code:`"F"` or
    :code:`"C"` will have expected behavior when wrapping FORTRAN or C buffers, respectively.

:code:`managed: Optional[str]`
    :code:`None`, :code:`"gt4py"` or :code:`"cuda"`. It only has effect if :code:`device="gpu"` and
    it specifies whether the synchronization between the host and device buffers is not done
    (:code:`None`), GT4Py (:code:`"gt4py"`) or CUDA (:code:`"cuda"`). It defaults to :code:`"gt4py"`

The values of parameters which are not explicitly defined by the user will be inferred from the
first alternative source where the parameter is defined in the following search order:

1. The provided :code:`defaults` parameter set.
2. The provided :code:`data` or :code:`device_data` parameters.
3. A fallback default value specified above. The only case where this is not available is
   :code:`shape`, in which case an exception is raised.


.. _domain_and_halo:

Storage Attributes and NumPy API functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An initial proposal of supported features is presented here. By features we mean NumPy functions
(:code:`np.function()` -like) that work well with GT4Py storages, as well as attributes
(:code:`ndarray.attribute`) and methods (:code:`ndarray.method()`) of the :code:`ndarray` class.

NumPy Functions
===============

:code:`np.transpose`
    Return a view of the buffers with the strides permuted in the order indicated by
    :code:`axes`.

Attributes and Properties
=========================
:code:`Storage` s have the following attributes:

:code:`__array_interface__: Dict[str, Any]`
    The *Array Interface* descriptor of this storage (only supported on instances with an
    actual host buffer).

:code:`__cuda_array_interface__: Dict[str, Any]`
    The *CUDA Array Interface* descriptor of this storage (only supported on instances with an
    actual GPU device buffer).

:code:`__gt_data_interface__: Dict[str, Dict[str, Any]]`
    The high-level descriptor of this storage as documented above. The :code:`None` and
    :code:`"gpu"` keys will point to the :code:`__array_interface__` and
    :code:`__cuda_array_interface__` respectively, with the added :code:`"acquire"` and
    :code:`"touch"` keys of each interface set to point to the :code:`device_to_host`,
    :code:`host_to_device`, :code:`set_host_modified` and :code:`set_device_modified` methods,
    respectively. The :code:`"dims"` and :code:`"release"` keys will not be used.

:code:`data: Optional[memoryview]`
    If the instance contains a host memory buffer, the :code:`data` attribute of the underlying
    :code:`np.ndarray` instance backing the host memory buffer, :code:`None` otherwise.

:code:`device: Optional[str]`
    If the instance contains a device memory buffer, the device identifier where the device
    buffer is allocated, :code:`None` otherwise.

:code:`device_data: Optional[cp.cuda.MemoryPointer]`
    If the instance contains a device memory buffer, the :code:`data` attribute of the underlying
    :code:`cp.ndarray` instance backing the device memory buffer, :code:`None` otherwise.

:code:`domain_view: Storage`
    A view of the buffer with the halo removed. In the returned view instance, the index
    :code:`[0, 0, 0]` corresponds to the first point in the domain.

:code:`dtype: np.dtype`
    The NumPy :code:`dtype` of the storage.

:code:`halo: Tuple[Tuple[int, int], ...]`
    A tuple of length ndim with entries which are 2-tuples of ints. Specifies the start and end
    boundary sizes on the respective dimension. This property can be modified at run-time and
    therefore has a corresponding setter, where values of the type
    :code:`Tuple[Union[int,Tuple[int, int]], ...]` are accepted with the same meaning as for the
    halo parameter of the storage creation functions. Not however, that this will not readjust the
    gridpoint which is aligned, since this would require re-allocation.

:code:`nbytes: int`,
    Size of the buffer in bytes (excluding padding).

:code:`ndim: int`
    Number of allocated dimensions.

:code:`shape: Tuple[int, ...]`
    The shape of the buffer, i.e., a tuple of length :code:`ndim` with entries corresponding to the
    axes indicated by :code:`axes`.

:code:`strides: Tuple[int, ...]`
    The strides of the buffer, i.e., a tuple of length :code:`ndim` with entries corresponding to
    the axes indicated by :code:`axes`.

:code:`sync_state: gt4py.storage.SyncState`
    Indicates which buffer is currently modified in case of a :code:`SoftwareManagedGPUStorage`. For
    more details on :code:`gt4py.storage.SyncState`, see :ref:`sync_state`. Only an attribute of the
    :code:`SoftwareManagedGPUStorage` storage.

Methods
=======

:code:`__array__(self: Storage) -> Union[np.ndarray, cp.ndarray]`
    A view of :code:`self` as a NumPy ndarray (if this instance contains a host buffer), or as a
    CuPy ndarray if this instance only contains a device buffer.

:code:`__deepcopy__(self: Storage, memo: Optional[Dict] = None) -> Storage`
    Used if :code:`copy.deepcopy()` is called on a :code:`Storage` instance.

:code:`__getitem__(self: Storage, key) -> Union[Number, Storage, cp.ndarray, np.ndarray]`
    Get a value at a certain index, a storage view of a subregion of the underlying buffer or
    a ndarray of values at selected locations.

    Otherwise, i.e. in the case of "Basic Indexing", axes for which a single index is selected
    are removed from :code:`axes` in the returned Storage, while slices do not reduce
    dimensionality.

    Parameters:
        + :code:`key: index_like` Indicates the indices from which the data of the storage is to be
          returned. The same keys as in
          `NumPy Indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_ are
          allowed, with the addition that keys can be any object implementing the interfaces
          discussed in this proposal whenever a :code:`np.ndarray` is valid.


:code:`__setitem__(self: Storage, key: key_like, Value) -> None`
    Set the data of the storage at a certain index, in a subregion or
    at selected locations of the underlying buffer.

        + :code:`key: index_like` Indicates the locations at which the values are to be changed. The
          same keys as for :code:`__getitem__` are supported.

        + :code:`value: Union[Number, array_like]` the values that are copied to the storage at the
          locations indicated by :code:`key`.

:code:`copy(self: Storage) -> Storage`
    Create a new Storage instance with the same parameters as this instance and a copy of the data.

:code:`to_cupy(self: Storage) -> cp.ndarray`
    Return a view of the underlying device buffer (CuPy :code:`ndarray`) if present or raise a
    :code:`GTNoSuchBufferError` if this instance does not contain a device buffer.

:code:`to_ndarray(self: Storage) -> Union[np.ndarray, cp.ndarray]`
    Return a view of the device buffer (CuPy :code:`ndarray`) if present or a view of the host
    buffer (NumPy :code:`ndarray`) otherwise.

:code:`to_numpy(self: Storage) -> np.ndarray`
    Return a view of the underlying host buffer (NumPy :code:`ndarray`) if present or raise a
    :code:`GTNoSuchBufferError` if this instance does not contain a host buffer.

:code:`transpose(self: Storage, *axes: Optional[Sequence[int]]) -> Storage`
    Return a view of the underlying buffers with the strides permuted in the order indicated by
    :code:`axes`.

The following methods are used to ensure that one-sided modifications to the host or device
buffers of a storage instance are tracked properly when the synchronization is managed by GT4Py.
The use of these methods should only be necessary if a reference to the internal Storage buffers
is kept or modified outside of GT4Py, which is generally not recommended. For Storage instances
with a different synchronization option they are valid methods implemented as no-ops functions so
user code can be agnostic of the backend and the synchronization mode.

:code:`device_to_host(self: Storage, *, force: bool = False) -> None`
    Triggers a copy from device buffer to the sibling in host memory if the device is marked as
    modified or the method is called with `force=True`. After the call the buffers are flagged as
    synchronized.

:code:`host_to_device(self: Storage, *, force: bool = False) -> None`,
    Triggers a copy from host buffer to the sibling in device memory if the host is marked as
    modified or the method is called with `force=True`. After the call the buffers are flagged as
    synchronized.

:code:`set_device_modified(self: Storage) -> None`
    Mark the device buffer as modified, so that a copy from device to host is automatically
    triggered before the next access to the host buffer.

:code:`set_host_modified(self: Storage) -> None`
    Mark the host buffer as modified, so that a copy from host to devcie is automatically triggered
    before the next access to the device buffer.

:code:`set_synchronized(self: Storage) -> None`
    Mark host and device buffers as synchronized, meaning they are equal. (In case the user has done
    this synchronization manually).

:code:`synchronize(self: Storage) -> None`,
    Triggers a copy between host and device buffers if the host or device, respectively are
    marked as modified. After the call the buffers are flagged as
    synchronized.

Choosing the Device
===================

For the synchronized memory classes (be it by CUDA or by GT4Py), the the device where data is
written in :code:`__setitem__` is chosen depending on

:code:`CudaManagedGPUStorage`
    The device is chosen to be GPU if and only if the value is GPU-enabled. A value is considered
    GPU enabled, if:

    1. It implements the :code:`__gt_data_interface__` and information for a :code:`"gpu"` buffer is
       provided. In this case, the :code:`"acquire"` method is called before reading.
    2. It does not implement the data interface but it is compatible with :code:`cp.asarray`, which
       includes values implementing the :code:`__cuda_array_interface__`.

:code:`SoftwareManagedGPUStorage`
    The device is chosen to be GPU if and only if the value is considered as on GPU. If the value
    is itself a :code:`SoftwareManagedGPUStorage`, it is considered as on GPU, if the buffers are
    either in sync or the GPU buffer is modified. If however, the value is not a
    :code:`SoftwareManagedGPUStorage`, the same logic applies as for the
    :code:`CudaManagedGPUStorage`. Note that :code:`xarray` :code:`DataArray` will be treated based
    on the data interface and not the underlying storage.

.. _storage_types:

Storage Types
^^^^^^^^^^^^^

GT4Py Storages objects type should be subclasses of the main :code:`Storage` class. Depending on
the choice of the :code:`device` and :code:`managed` values (see Section :ref:`constructors`), the
type is one of :code:`CPUStorage`, :code:`GT4PySyncedGPUStorage`, :code:`CUDASyncedGPUStorage`
or :code:`GPUStorage`.

Their purpose is as follows:

:code:`CUDAManagedGPUStorage`
    Internally holds a reference to a `NumPy <https://numpy.org/>`_ `ndarray`. The memory is however
    allocated as CUDA unified memory, meaning that the same memory can be accessed from GPU, and
    synchronization is taken care of by the CUDA runtime.

:code:`CPUStorage`
    It holds a reference to a `NumPy <https://numpy.org/>`_ :code:`ndarray`.

:code:`GPUStorage`
    Internally holds a reference to a `CuPy <https://cupy.chainer.org/>`_ `ndarray`. This storage
    does not have a CPU buffer.

:code:`SoftwareManagedGPUStorage`
    Internally holds a reference to both a `NumPy <https://numpy.org/>`_ and a
    `CuPy <https://cupy.chainer.org/>`_ :code:`ndarray`. Synchronization is taken care of by GT4Py.

.. _sync_state:

Sync State
^^^^^^^^^^

The :code:`gt4py.storage.SyncState` is used to track which buffer of a
:code:`SoftwareManagedGPUStorage` is modified. Since multiple storages can be views of the same
underlying buffers, or only different parts of it, changing the :code:`sync_state` of one such
storage must also change the state of all other views of the same base buffer. They therefore share
the same :code:`SyncState` instance, which can be accessed through the :code:`sync_state` attribute
of the storage. The :code:`state` attribute of the :code:`SyncState` instance can assume the values
:code:`SyncState.SYNC_CLEAN`, :code:`SyncState.SYNC_HOST_DIRTY` or
:code:`SyncState.SYNC_DEVICE_DIRTY`.

Alternatives
------------


Subclassing
^^^^^^^^^^^

For the implementation strategy, a viable alternative could be to implement GT4Py storages as a
NumPy `ndarray` subclass as in the current implementation. Due to the issues mentioned in the
introduction, we consider that this strategy imposes more limitations than using `duck typing`.

Retaining `dims` information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In an earlier version of this proposal, we proposed to also hold the information that can now be
passed through the :code:`"dims"` of the :code:`__gt_data_interface__` in the gt4py implementation
which would have allowed us to

However, these would still not have covered all cases, while taking away some freedom to implement
the desired behavior from users. Further the interface proposed here was done with the move to
:code:`GridTools 2.0` with which the `Stencil Iterable Data (SID)` concept will be supported in the
generated code. With it, generated code will be valid for any stride order, although performance may
still be better for certain combinations. With this change, the conservation of the layout under
ufunc operations will be less important. We believe that the costs of having the :code:`dims` in the
storage implementation rather than the interface proposed here will then outweigh the benefits.

Implementing ufuncs and other NumPy API functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, it was possible to call mathematical operations on the storages, and an earlier version
of this GDP proposed to implement this using the functionality offered by the `NumPy Enhancment
Proposals` NEP13 and NEP18. However, it could not be guaranteed that in this way, the requirements
for the best performance for a given backend could always be infered under these operations.
Further, approaches to implementation of these interfaces depend on the availability of third party
libraries implementing the operations on a lower level. However, this can not be assumed to be
extensible for upcoming hardware.

No storages
^^^^^^^^^^^

Alternatively, instead of providing custom storages, a small set of utilities to facilitate
allocation of buffers with properties desireable for performance can be provided. All operations
on the data that are not in stencils are then to be performed in third party frameworks.
Meta information like dimensionality, origins etc. can still be provided by an interface similar
to the `__gt_data_interface__` described in this GDP. 

Copyright
---------

This document has been placed in the public domain.
