======================================================
GDP 3 — A New Storage Implementation using Duck Typing
======================================================

:Author: Linus Groner <linus.groner@cscs.ch>
:Author: Enrique G. Paredes <enrique.gonzalez@cscs.ch>
:Status: Draft
:Type: Feature
:Created: 08-05-2020
:Discussion PR: https://github.com/GridTools/gt4py/pull/28


Abstract
--------

We propose to replace the current storage implementation by a new `storage` class hierarchy
which does not inherit from NumPy `ndarray`. Further, the new storage classes shall introduce
the possibility to be constructed from externally allocated memory buffers and provide more
control over the underlying memory.


Motivation and Scope
--------------------

In the current state of GT4Py, we implemented storages as subclasses of NumPy :code:`ndarrays`.
Although this strategy is fully documented and supported by NumPy API, it presents some drawbacks
such as the possibility of missing some buffer metadata under certain operations of the NumPy API.
Further, some NumPy API calls cause one-sided changes to coupled host/device buffers managed by
GT4Py storages (e.g. :code:`ExplicitlySyncedGPUStorage` class) which cannot be tracked reliably,
resulting in validation errors which are hard to find and fix.

The current implementation of GT4Py storages as :code:`ndarray` subclasses was needed to use
storages transparently with third-party frameworks relying on NumPy :code:`ndarray` implementation
details. Nowadays, however, most of the Python scientific ecosystem supports the more generic
interface specified in the :emphasis:`NumPy Enhancement Proposal`
`NEP18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_, which allows the seamless
integration of NumPy code with custom implementations of the NumPy API by means of
`duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_. Thus, reimplementing GT4Py storages
using the interface introduced in the NEP18 allows GT4Py to retain full control over the internal
behavior of complex operations, while keeping interoperability with third-party scientific
frameworks.

Additionally, we propose to take this opportunity to improve the interaction with existing codes by
allowing the initialization of storages from external buffers without requiring a copying operation.
To use this feature, some additional information about the provided buffers needs to be specified to
both the :code:`__init__` method of the storages as well to GT4Py stencils at compile time.


Backward Compatibility
----------------------

The implementation of this GDP breaks the integration with external libraries which require a NumPy
`ndarray` subclass. Further, we propose some API changes like renaming or repurposing of keyword
arguments and attributes. These changes are expected to break most of existing user codes. However,
existing GT4Py functionality will remain or be extended and thus updating codebases to the new
interface amounts mostly to updating attribute and keyword argument names.


Detailed Description
--------------------

Functionality
^^^^^^^^^^^^^

Our main goal is to make GT4Py `storages` behave as close as possible to NumPy `ndarrays`
for the base cases while providing additional domain-specific functionality. This means that,
ideally, users can treat GT4Py Storages as regular NumPy arrays in their codes and expect the
same results. The specific cases where behavior may differ because of the limitations of the NumPy
API or the GT4Py Storage requirements, should be mentioned here and in the GT4Py documentation.


We chose to use internally NumPy and CuPy `ndarrays` to store CPU and GPU buffers, respectively,
since they are standard and realiable components of the Python ecosystem. We rely on those libraries
to implement part of the functionality, like mathematical operators, by forwarding the calls to
the appropriate NumPy/CuPy function call. As a consequence, support for `dtypes` and other
functionality is restricted to the common denominator of CuPy and NumPy.

.. note:: In this document, references to NumPy and CuPy objects or functions use the standard
    shorted prefiex for these libraries, that is :code:`np.` for NumPy and :code:`cp.` for CuPy.


.. _constructors:

Storage Creation
^^^^^^^^^^^^^^^^

The :code:`Storage` base class is exposed in the API mainly to enable type checking. For the actual
creation and initialization of GT4Py storages we propose the following set of functions which
closely resemble their NumPy counterparts (meaning of the common parameters is explained below):

:code:`empty(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with uninitialized (undefined) values.

    Parameters:
        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the
          shape of the storage, that is, the full addressable space in the allocated memory buffer.
          (See also Section :ref:`domain_and_halo`)

    For common keyword-only arguments, please see below.


:code:`empty_like(data: Storage, dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with uninitialized (undefined) values, while choosing the not explicitly
    overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage`
          Not explicitly overridden parameters are chosen as the value used in this
          :code:`Storage`

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

    The common keyword-only arguments can also be overriden. Please see below for their description.

    Note that :code:`shape` is not a parameter and can not be overridden, implying that also the
    :code:`axes` can not be overridden.

:code:`zeros(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 0.

    Parameters:
        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the
          shape of the storage, that is, the full addressable space in the allocated memory buffer.
          (See also Section :ref:`domain_and_halo`)

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

    Note that :code:`shape` is not a parameter and can not be overridden, implying that also the
    :code:`axes` can not be overridden.


:code:`ones(shape: Sequence[int], dtype: dtype_like = np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to 1.

    Parameters:
        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the
          shape of the storage, that is, the full addressable space in the allocated memory buffer.
          (See also Section :ref:`domain_and_halo`)

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

    Note that :code:`shape` is not a parameter and can not be overridden, implying that also the
    :code:`axes` can not be overridden.


:code:`full(shape: Sequence[int], fill_value: Number, dtype=np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to the scalar given in :code:`fill_value`.

    Parameters:
        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`np.float64`.

        + :code:`fill_value: Number`. The number to which the storage is initialized.

        + :code:`shape: Sequence[int]`
          Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the
          shape of the storage, that is, the full addressable space in the allocated memory buffer.
          (See also Section :ref:`domain_and_halo`)

    For common keyword-only arguments, please see below.

:code:`full_like(shape: Sequence[int], fill_value: Number, dtype=np.float64, **kwargs) -> Storage`
    Allocate a storage with values initialized to the scalar given in :code:`fill_value`, while
    choosing the not explicitly overridden parameters according to :code:`data`.

    Parameters:
        + :code:`data: Storage` Not explicitly overridden parameters are chosen as the value used in
          this :code:`Storage`

        + :code:`dtype: dtype_like`
          The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
          :code:`data.dtype`

        + :code:`fill_value: Number`. The number to which the storage is initialized.

    The common keyword-only arguments can also be overridden. Please see below for their description.

    Note that :code:`shape` is not a parameter and can not be overridden, implying that also the
    :code:`axes` can not be overridden.

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
    Used to allocate a storage with values initialized to those of a given array.
    If the argument :code:`copy` is set to :code:`False`, the behavior is that of :code:`as_storage`.

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

    If the field is not 3-D, as indicated by :code:`axes`, the length of parameters
    :code:`aligned_index` and :code:`shape`, may either be of length 3 or of the actual dimension
    of the storage, where the not needed entries are ignored in the latter case.


The definitions of the common parameters accepted by all the previous functions are the following:

:code:`dtype: dtype_like`
    The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
    :code:`np.float64`.

:code:`shape: Sequence[int]`
    Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the shape
    of the storage, that is, the full addressable space in the allocated memory buffer. (See also
    Section :ref:`domain_and_halo`)

Additionally, these **optional** keyword-only parameters are accepted:

:code:`aligned_index: Sequence[int]`
    The point to which the memory is aligned, defaults to the lower indices of the halo attribute.

:code:`alignment: Optional[int]`
    Indicates on a boundary of how many elements the point :code:`aligned_index` is aligned. It
    defaults to :code:`1`, which indicates no alignment.

:code:`axes: str`
    Any permutation of a sub-sequence of the :code:`"IJK"` string indicating the spatial dimensions
    along which the field extends and their order for indexing operations in Python. The default
    value is :code:`"IJK"`.

:code:`defaults: Optional[str]`
    It can be used in the way of the current :code:`backend` parameter. For each backend, as well
    as for the keys :code:`"F"` and :code:`"C"` (equivalent to the same values in the :code:`order`
    parameter for NumPy allocation routines) a preset of suitable parameters is provided. Explicit
    definitions of additional parameters are possible and they override its default value from the
    preset.

:code:`device: Optional[str]`
    Indicates whether the storage should contain a buffer on an accelerator device. Currently it
    only accepts :code:`"gpu"` or :code:`None`. Defaults to :code:`None`.

:code:`halo: Optional[Sequence[Union[int, Tuple[int, int]]]`
    Sequence of length :code:`ndim` where each entry is either an :code:`int` or a 2-tuple
    of :code:`int` s. A sequence of integer numbers represent a symmetric halo with the specific
    size per dimension, while a sequence of 2-tuple specifies the start and end boundary sizes on
    the respective dimension. It defaults to no halo, i.e. :code:`(0, 0, 0)`. (See also Section
    :ref:`domain_and_halo`)

:code:`layout: Optional[str]`
    A length-3 string with the name of axes. The sequence indicates the order of strides in
    decreasing order, i.e. the first entry in the sequence corresponds to the axis with the largest
    stride. The layout map is always of length 3. The default value is "IJK".

    Default values as indicated by the :code:`defaults` parameter may depend on the axes. E.g. if
    :code:`defaults` is any of the compiled GridTools backends, the default value is defined
    according to the semantic meaning of each dimension. For example for the :code:`"gtx86"`
    backend, the layout is always IJK, meaning the smallest stride is in the K dimension,
    independently which index corresponds to the K dimension. On the other hand, we assume that if a
    storage is created from an existing FORTRAN array, the first index has the smallest stride,
    irrespective of its corresponding axis. I.e. the first index has the smallest stride in FORTRAN
    for both IJK and KJI storages.

    ==================  ====================  ====================  ========================  =========================
     Default Layout     :code:`defaults="F"`  :code:`defaults="C"`  :code:`defaults="gtx86"`  :code:`defaults="gtcuda"`
    ==================  ====================  ====================  ========================  =========================
    :code:`axes="IJK"`  :code:`layout="KJI"`  :code:`layout="IJK"`  :code:`layout="IJK"`      :code:`layout="KJI"`
    :code:`axes="KJI"`  :code:`layout="IJK"`  :code:`layout="KJI"`  :code:`layout="IJK"`      :code:`layout="KJI"`
    ==================  ====================  ====================  ========================  =========================

    The rationale behind this is that in this way, storages allocated with :code:`defaults` set to a
    backend will always get optimal performance, while :code:`defaults` set to :code:`"F"` or
    :code:`"C"` will have expected behavior when wrapping FORTRAN or C buffers, respectively.

    The :code`layout` parameter always has to be of length 3, so that in the case a storage is 1 or
    2-dimensional, the place of the missing dimension is known. In this way, the result of ufunc's
    involving only storages that were allocated for a certain backend, will always again result in
    compatible storages. (See also Section :ref:`output_storage_parameters`)

:code:`managed: Optional[str]`
    :code:`None`, :code:`"gt4py"` or :code:`"cuda"`. It only has effect if :code:`device="gpu"` and
    it specifies whether the synchronization between the host and device buffers is handled manually
    by the user (:code:`None`), GT4Py (:code:`"gt4py"`) or CUDA (:code:`"cuda"`). It defaults to
    :code:`"gt4py"`

The values of parameters which are not explicitly defined by the user will be inferred from the
first alternative source where the parameter is defined in the following search order:

1. The provided :code:`defaults` parameter set.
2. The provided :code:`data` or :code:`device_data` parameters.
3. A fallback default value specified above. The only case where this is not available is
   :code:`shape`, in which case an exception is raised.


.. _domain_and_halo:

Storage Domain and Halo
^^^^^^^^^^^^^^^^^^^^^^^

Semantically, the :code:`halo` parameter is just a convenient marker for users to delimit the
expected inner domain part. This separation is used to infer the compute domain when calling
stencils, unless an origin and domain are explicitly specified. Further, the domain can be accessed
in the storage through the :code:`domain` attribute of the storage. The result is again a storage
with no halo, such that the index :code:`storage.domain[0, 0, 0]` refers to a corner of the inner
domain denoted by the halo.

The halo can be changed after allocation by assigning to :code:`storage.halo` using the same values
as in the :code:`halo` parameter above.

The halo passed when allocating is used as a hint for alignment. If no :code:`aligned_index` is
explicitly specified, the lower indices of the :code:`halo` parameter are used to align the memory
which, when coinciding with the origin when calling stencils, potentially has performance benefits.
When changing the halo after allocation, this has however no impact on the alignment and it is in
general best to specify the same halo that is used in the computations already when allocating.

Storage Attributes and NumPy API functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we mentioned above, keeping compatibility with NumPy is an essential requirement for GT4Py
Storages and therefore they should support the most relevant parts of the NumPy API. An
initial proposal of supported features is presented here. By features we mean NumPy functions
(:code:`np.function()` -like) that work well with GT4Py storages, as well as attributes
(:code:`ndarray.attribute`) and methods (:code:`ndarray.method()`) of the :code:`ndarray` class.

We are aware that this is by no means an exhaustive list and we ask for additional input from the
community to collect other features that should be supported.

NumPy Functions
===============
:code:`np.all`
    same semantics as :code:`np.logical_and.reduce`, when applied to all axes.

:code:`np.any`
    same semantics as :code:`np.logical_or.reduce`, when applied to all axes.

:code:`np.max`
    same semantics as :code:`np.max.reduce`, when applied to all axes.

:code:`np.min`
    same semantics as :code:`np.min.reduce`, when applied to all axes.

:code:`np.transpose`
    permutation of the axes. In addition to the parameters of :code:`np.transpose`, when applied to
    :code:`ndarray`'s, :code:`axes` can be the usual strings to represent the :code:`axes` attribute
    of the resulting storage. See also the :code:`reinterpret` method below.


Attributes and Properties
=========================
:code:`Storage` s have the following attributes:

:code:`__array_interface__: Dict[str, Any]`
    The *Array Interface* descriptor of this storage (only supported on instances with an
    actual host buffer).

:code:`__cuda_array_interface__: Dict[str, Any]`
    The *CUDA Array Interface* descriptor of this storage (only supported on instances with an
    actual GPU device buffer).

:code:`alignment: int`
    Alignment size value given at creation time

:code:`aligned_index: Tuple[int]`
    The index of the grid point to which the memory is aligned, given at creation time.
    Note that this only partly takes the role of the former :code:`default_origin` parameter,
    since part of the functionality is now taken over by the :code:`halo` attribute.

:code:`axes: str`
    Domain axes represented in the current instance, e.g. :code:`"JI"` for a 2d field spanning
    longitude and latitude but not the vertical, where the first index corresponds to the "J"
    axis.

:code:`data: Optional[memoryview]`
    If the instance contains a host memory buffer, the :code:`data` attribute of the underlying
    :code:`np.ndarray` instance backing the host memory buffer, :code:`None` otherwise.

:code:`device: Optional[str]`
    If the instance contains a device memory buffer, the device identifier where the device
    buffer is allocated, :code:`None` otherwise.

:code:`device_data: Optional[cp.cuda.MemoryPointer]`
    If the instance contains a device memory buffer, the :code:`data` attribute of the underlying
    :code:`cp.ndarray` instance backing the device memory buffer, :code:`None` otherwise.

:code:`device_flags: Optional[cp.core.flags.Flags]`
    If the instance contains a device memory buffer, the :code:`flags` attribute of the underlying
    :code:`cp.ndarray` instance backing the device memory buffer, :code:`None` otherwise.

:code:`domain_view: Storage`
    A view of the buffer with the halo removed. In the returned view instance, the index
    :code:`[0, 0, 0]` corresponds to the first point in the domain.

:code:`dtype: np.dtype`
   The NumPy :code:`dtype` of the storage.

:code:`flags: Optional[np.flagsobj]`
    If the instance contains a host memory buffer, the :code:`flags` attribute of the underlying
    :code:`np.ndarray` instance backing the host memory buffer, :code:`None` otherwise.

:code:`halo: Tuple[Tuple[int, int], ...]`
    A n-dimensional tuple of 2-tuples of ints, in the same format as the halo parameter of the
    construction functions. This property can be modified at run-time and therefore has a
    corresponding setter, where values of the type :code:`Tuple[Union[int,Tuple[int, int]], ...]`
    are accepted with the same meaning as for the :code:`halo` parameter of the storage creation
    functions.

:code:`nbytes: int`,
    Size of the buffer in bytes (excluding padding).

:code:`ndim: int`
    Number of allocated dimensions.

:code:`shape: Tuple[int, ...]`
    The shape of the buffer, i.e., a tuple of length :code:`ndim` with entries corresponding
    to the axes indicated by :code:`axes`.

:code:`strides: Tuple[int, ...]`
    The strides of the buffer, i.e., a tuple of length :code:`ndim` with entries corresponding
    to the axes indicated by :code:`axes`.

:code:`sync_state: gt4py.storage.SyncState`
    Indicates which buffer is currently modified in case of a :code:`SoftwareManagedGPUStorage`.
    For more details on :code:`gt4py.storage.SyncState`, see :ref:`sync_state`.
    Only an attribute of the :code:`SoftwareManagedGPUStorage` storage.

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

    In case of those keys described as "Advanced Indexing" in the
    `NumPy Indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_ page, the
    result will be a linear 1-d array without any reference to spatial dimensions like in
    fields. Since returning a Storage in this case would be misleading, the result is a CuPy
    ndarray if the storage is GPU enabled, or a NumPy ndarray otherwise.

    Otherwise, i.e. in the case of "Basic Indexing", axes for which a single index is selected
    are removed from :code:`axes` in the returned Storage, while slices do not reduce
    dimensionality.

    Parameters:
        + :code:`key: index_like` Indicates the indices from which the data of the storage is to be
          returned. The same keys as in
          `NumPy Indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
          are allowed, with the addition that

          + keys can be :code:`cp.ndarrays` whenever a :code:`np.ndarray` is valid or

          + a boolean or integer GT4Py storage with the same shape as :code:`self` (possibly with
            permuted axes).


:code:`__setitem__(self: Storage, key: key_like, Value) -> None`
    Set the data of the storage at a certain index, in a subregion or
    at selected locations of the underlying buffer

    The :ref:`broadcasting` behavior and device selection are equivalent to that of a unary ufunc
    with a provided output buffer. For example, :code:`stor_out[key] = stor2d` would be
    equivalent to :code:`np.positive(stor2d, out=stor_out[key]`)

    Parameters:
        + :code:`key: index_like` Indicates the locations at which the values are to be changed. The
          same keys as for :code:`__setitem__` are supported.

        + :code:`value: Union[Number, Storage, cp.ndarray, np.ndarray]` the values that are copied
          to the storage at the locations indicated by :code:`key`.

:code:`copy(self: Storage) -> Storage`
    Create a new Storage instance with the same parameters as this instance and a copy of the data.

:code:`reinterpret(self, axes) -> Storage`
    Parameters:
        + :code:`axes: Sequence[str]` the axes of the resulting storage

    A view of the buffer, with the axes relabeled according to :code:`axes`. The behavior seems
    similar to transpose, however the shape and strides remain unchanged, meaning that not only the
    order of indexing changes, but also the semantic meaning attached to the data. For example, if
    :code:`self` is an IJK-storage with :code:`shape==(10,20,30)`, and we call
    :code:`self.reinterpret("KJI")`, a KJI-storage with the same shape as before is returned,
    meaning that e.g. the size in the K-dimension will change from 30 to 10 in the result.

:code:`to_cupy(self: Storage) -> cp.ndarray`
    Return a view of the underlying device buffer (CuPy :code:`ndarray`) if present or raise an
    exception if this instance does not contain a device buffer.

:code:`to_ndarray(self: Storage) -> Union[np.ndarray, cp.ndarray]`
    Return a view of the device buffer (CuPy :code:`ndarray`) if present or a view of the host
    buffer (NumPy :code:`ndarray`) otherwise.

:code:`to_numpy(self: Storage) -> np.ndarray`
    Return a view of the underlying host buffer (NumPy :code:`ndarray`) if present or raise an
    exception if this instance does not contain a host buffer.

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
    marked as modified. The buffers are marked as in sync after the operation.

Universal Functions (ufuncs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Universal Functions <https://numpy.org/doc/stable/reference/ufuncs.html>`_ are a subset of the
NumPy API which mostly implements mathematical operator functions and have a particular structure:
They are subclasses of :code:`np.ufunc` and can be invoked through the
`methods <https://numpy.org/doc/stable/reference/ufuncs.html#methods>`_ :code:`reduce`,
:code:`accumulate`, :code:`reduceat`, :code:`outer`, :code:`at` and :code:`__call__`. We propose to
use the `mixin` functionality and to implement the :code:`__array_ufunc__` interface to support
these functions: NumPy provides the :code:`numpy.lib.mixins.NDArrayOperatorsMixin` class, from which
a duck array can inherit from. Doing so forwards mathematical operators using python syntax (such as
binary :code:`+` or unary :code:`-`) to the :code:`__array_ufunc__` method where own behavior can
be defined.

Using mathematical operators with the mixin is equivalent to calling the ufuncs through the
:code:`__call__` method. (E.g. :code:`np.add(storage1, storage2)` or
:code:`np.negative(storage)`) alternatively, some ufuncs can be used to perform reductions.
In this case, one can explicitly call the :code:`reduce` method (e.g. :code:`np.add.reduce(axis=1)`
to accumulate the values along a given axis.

We propose to support the methods :code:`__call__` and :code:`reduce` of the NumPy ufunc
mechanism.

If the :code:`reduce` method of `ufuncs` is used, this results in a Storage with the axis along
which the reduction was performed removed from the Storage. (For example taking the sum over the K
axis of an IJK storage will result in an IJ storage). In addition to vanilla numpy behavior, the
:code:`axis` keyword of the :code:`reduce` method accepts the axis along which the reduction is
performed as a string.

In the following subsections we describe the proposed behaviour for GT4Py storages when used in
conjuction with NumPy ufuncs.

.. _broadcasting:

Broadcasting
============

With the term "broadcasting", NumPy describes the ways that different shapes are combined in
assignments and mathematical operations. GT4Py storages should override the default NumPy behavior
so that fields are broadcast along the same spatial dimension. For example, adding an :code:`IJ`
field :code:`A` of shape :code:`(2, 3)` with a :code:`K` field :code:`B` of shape :code:`(4,)` will
result in an :code:`IJK` field :code:`C` of shape :code:`(2, 3, 4)`, with `C[i,j,k] = A[i,j]+B[k]`.

Similarly, fields of lower dimension are assigned to such of higher dimension by broadcasting along
the missing axes. To keep compatibility with NumPy, dimensions of size 1 are treated like missing
axes when broadcasting.

Similarly, fields of lower dimension are assigned to such of higher dimension by broadcasting along
the missing dimensions. To keep compatibility with NumPy, dimensions of size 1 are treated like
missing axes when broadcasting.


.. _output_storage_parameters:

Output Storage Parameters
=========================

If no output buffer is provided, the constructor parameters of the output storage have to be
inferred using the available information from the inputs.

:code:`aligned_index`
    It is chosen to be as the largest value per dimension across all inputs which are a GT4Py
    Storage.

:code:`alignment`
   The resulting alignment is chosen as the least common multiple of the alignments of all inputs
   which are a GT4Py storage.

:code:`axes`
    If the :code:`axes` parameters of all operands agree, the output will have the same
    :code:`axes`. Otherwise, if one input contains all axes of the other input, that input
    determines the axes of the output. If none of these conditions are met, axes are chosen as the
    union of all input storages. Their order will then be a sub-sequence of "IJK".

:code:`dtype`
   The resulting dtype is determined by NumPy behavior.

:code:`halo`
    It is chosen s.t. the resulting domain is the intersection of all individual domains.

:code:`layout`
    The layout is chosen as the layout of the first input argument which is a GT4Py Storage


Mixing Types
============

If a binary `ufunc` is applied to a storage and a non-storage array, the storage determines the
behavior. Since non-storage arrays do not carry the necessary information to apply the usual
broadcasting rules, we only implement the cases where:

* the array has the same shape as the input storage or as the broadcast shape when considering a
  provided output buffer

* the array has a 3d shape where dimensions with shape :code:`1` in the array are broadcast.

Mixing Devices
==============

For the synchronized memory classes (be it by CUDA or by GT4Py), the compute device is chosen
depending on

:code:`CudaManagedGPUStorage`
    The compute device is chosen to be GPU if and only if the inputs are compatible with
    :code:`cp.asarray`.

:code:`SoftwareManagedGPUStorage`
    Here, the array is considered a GPU array if it is compatible with :code:`cp.asarray`. If a
    storage is modified on CPU, it is considered a CPU array here. The compute device is chosen as
    GPU unless all inputs are not GPU arrays (including if all inputs are
    :code:`SoftwareManagedGPUStorage` but are modified on CPU).

We assume that mixing these in the same application is not a common case. Should it nevertheless
appear, the object that handles the ufunc will determine the behavior. (Where each of the classes
will treat the other as on GPU.)

For pure CPU storages, all inputs and output need to be compatible with `np.asarray`, for GPU
storages with `cp.asarray`, otherwise an exception is raised.

:code:`CudaManagedGPUStorage` and :code:`SoftwareManagedGPUStorage` shall both have a
:code:`__array_priority__` set to :code:`11`, while for :code:`CPUStorage` and :code:`GPUStorage` it
is set to :code:`10`, meaning that managed storages have priority in handling these cases.

Implementation
--------------
Operators and `ufuncs` are handled by inheriting from :code:`numpy.NDArrayOperatorsMixin` and
implementing the :code:`__array_ufunc__` interface. The internal implementation of the
:code:`__array_ufunc__` will determine the proper broadcasting, output shape and compute device,
and then allocate the appropriate output buffers and  dispatch the actual computation to NumPy or
CuPy, respectively. This strategy should work perfectly since all CPU buffers are implemented using
NumPy `ndarrays` and GPU buffers are stored as CuPy `ndarrays`, except for the CUDA-managed GPU
storages, where CuPy views of the buffer are created as necessary. (see :ref:`storage_types`).

Other numpy API functions will be handled by means of the :code:`__array_function__`
protocol.

.. _storage_types:

Storage Types
^^^^^^^^^^^^^

GT4Py Storages objects type should be subclasses of the main :code:`Storage` clas. Depending on
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

The main aspects of this proposal are

* construction from existing buffers

* duck array versus subclassing

We believe the former to be non-controversial since it follows NumPy conventions. For the actual
implementation strategy, the only viable alternative could be to implement GT4Py storages as a
NumPy `ndarray` subclass as in the current implementation. Due to the issues mentioned in the
introduction, we consider that this strategy imposes more limitations than using `duck typing`.

Copyright
---------

This document has been placed in the public domain.
