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

The current implementation of GT4Py storages as `ndarray` subclasses was needed to use storages
transparently with third-party frameworks relying on NumPy `ndarray` implementation details.
Nowadays, however, most of the Python scientific ecosystem supports the more generic interface
specified in the :emphasis:`NumPy Enhancement Proposal` `NEP18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_,
which allows the seamless integration of NumPy code with custom implementations of the NumPy API by
means of `duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_. Thus, reimplementing GT4Py
storages using the interface introduced in the NEP18 allows GT4Py to retain full control over the
internal behavior of complex operations, while keeping interoperability with third-party scientific
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


Storage Creation
^^^^^^^^^^^^^^^^

The :code:`Storage` base class is exposed in the API mainly to enable type checking. For the actual
creation and initialization of GT4Py storages we propose the following set of functions which
closely resemble their NumPy counterparts (meaning of the common parameters is explained below):

:code:`empty(shape, dtype=np.float64, **kwargs)`
    Allocate a storage with uninitialized (undefined) values.

:code:`zeros(shape, dtype=np.float64, **kwargs)`
    Allocate a storage with values initialized to 0.


:code:`ones(shape, dtype=np.float64, **kwargs)`
    Allocate a storage with values initialized to 1.


:code:`full(shape, fill_value, dtype=np.float64, **kwargs)`
    Allocate a storage with values initialized to the given scalar.

    Parameters:
        + :code:`fill_value: Number`.

:code:`as_storage(data=None, device_data=None, *, sync_state=None, **kwargs)`
    Wrap an existing buffer in a GT4Py storage instance, without copying the buffer's contents.

    Parameters:
        + :code:`data: array_like`. The memory buffer or storage from which the storage is
          initialized.
        + :code:`device_data: array_like`. The device buffer or storage in case wrapping
          existing buffers on both the device and main memory is desired.

    Keyword-only parameters:
        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization.

:code:`storage(data=None, device_data=None, *, copy=True, **kwargs)`
    Used to allocate a storage with values initialized to those of a given array.
    If the argument :code:`copy` is set to :code:`False`, the behavior is that of :code:`as_storage`.

    Parameters:
        + :code:`data: array_like`. The original array from which the storage is initialized.
        + :code:`device_data: array_like`. The original array in case copying to a gpu buffer is desired.
            The same buffer could also be passed through `data` in that case, however this parameter is here to
            provide the same interface like the :code:`as_storage` function.
        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
            provided buffers, `data` or `device_data`, is up to date at the time of initialization.

    Keyword-only parameters:
        + :code:`copy: bool`. Allocate a new buffer and initialize it with a copy of the data or
          wrap the existing buffer.
        + :code:`sync_state: gt4py.storage.SyncState`. If `managed="gt4py"` indicates which of the
          provided buffers, `data` or `device_data`, is up to date at the time of initialization.

The definitions of the common parameters accepted by all the previous functions is the following:

:code:`dtype: np.dtype_like`
    The dtype of the storage (NumPy dtype or accepted by :code:`np.dtype()`). It defaults to
    :code:`np.float64`.

:code:`shape: Sequence[int]`
    Sequence of length :code:`ndim` (:code:`ndim` = number of dimensions, maximum 3) with the shape
    of the storage, that is, the full addressable space in the allocated memory buffer.

Additionally, these **optional** keyword-only parameters are accepted:

:code:`aligned_index: Sequence[int]`
    The point to which the memory is aligned, defaults to the lower indices of the halo attribute.

:code:`alignment: int`
    Sequence of length :code:`ndim` which indicates on a boundary of how many elements the point
    :code:`aligned_index` is aligned. It defaults to :code:`1`, which indicates no alignment.

:code:`axes: str`
    Any permutation of a sub-sequence of the :code:`"IJK"` string indicating the spatial dimensions
    along which the field extends and their order for indexing operations in Python. The default
    value is :code:`"IJK"`.

:code:`defaults: str`
    It can be used in the way of the current :code:`backend` parameter. For each backend, as well
    as for the keys :code:`"F"` and :code:`"C"` (equivalent to the same values in the :code:`order`
    parameter for NumPy allocation routines) a preset of suitable parameters is provided. Explicit
    definitions of additional parameters are possible and they override its default value from the
    preset.

:code:`device: str`
    Indicates whether the storage should contain a buffer on an accelerator device. Currently it
    only accepts :code:`"gpu"` or :code:`None`. Defaults to :code:`None`.

:code:`halo: Sequence[Union[int, Tuple[int, int]]`
    Sequence of length :code:`ndim` where each entry is either an :code:`int` or a 2-tuple
    of :code:`int` s. A sequence of integer numbers represent a symmetric halo with the specific
    size per dimension, while a sequence of 2-tuple specifies the start and end boundary sizes on
    the respective dimension. It defaults to no halo, i.e. :code:`(0, 0, 0)`

:code:`layout: str`
    A length-3 string with the name of axes (or a callable returning such a string)
    dimensions. The sequence indicates the order of strides in decreasing order, i.e. the first
    entry in the sequence corresponds to the axis with the largest stride. The layout map
    is always of, length 3. The default
    value is "IJK".

    Default values as indicated by the :code:`defaults` parameter may depend on the axes. E.g. if the defaults is any
    of the compiled GridTools backends, the default value is defined according to the semantic meaning of each
    dimension. For example for the :code:`"gtx86"` backend, the layout is always IJK, meaning the smallest stride is in
    the 3rd dimension, independently which dimension is the K dimension. On the other hand, we assume that if a storage
    is created from an existing FORTRAN array, the first index has the smallest stride, irrespective of its
    corresponding axis. I.e. the first index has the smallest stride in FORTRAN for both IJK and KJI storages.

    ==================  ====================  ====================  ========================  =========================
     Default Layout     :code:`defaults="F"`  :code:`defaults="C"`  :code:`defaults="gtx86"`  :code:`defaults="gtcuda"`
    ==================  ====================  ====================  ========================  =========================
    :code:`axes="IJK"`  :code:`layout="KJI"`  :code:`layout="IJK"`  :code:`layout="IJK"`      :code:`layout="KJI"`
    :code:`axes="KJI"`  :code:`layout="IJK"`  :code:`layout="KJI"`  :code:`layout="IJK"`      :code:`layout="KJI"`
    ==================  ====================  ====================  ========================  =========================

    The rationale behind this is that in this way, storages allocated with :code:`defaults` set to a backend will
    always get optimal performance, while :code:`defaults` set to :code:`"F"` or :code:`"C"` will have expected behavior
    when wrapping FORTRAN or C buffers, respectively.

    The :code`layout` parameter always has to be of length 3, so that in the case a storage is 1 or 2-dimensional,
    the place of the missing dimension is known. In this way, the result of ufunc's involving only storages that were
    allocated for a certain backend, will always again result in compatible storages. (See also Section
    :ref:`output_storage_parameters`)

:code:`managed: str`
    :code:`None`, :code:`"gt4py"` or :code:`"cuda"`. It only has effect if :code:`device="gpu"` and
    it specifies whether the synchronization between the host and device buffers is handled manually
    by the user (:code:`None`), GT4Py (:code:`"gt4py"`) or CUDA (:code:`"cuda"`). It defaults to
    :code:`"gt4py"`
.. COMMENT If a parameter is not explicitly specified, it is inferred from the default parameter set. If there
.. COMMENT is no default parameter set provided or it does not provide the required information, it is gathered
.. COMMENT from the :code:`data` or :code:`device_data` parameters. If this does not provide this information,
.. COMMENT a trivial default value is assumed. If no default value is available, an error is raised that the
.. COMMENT parameters are underdetermined.

- :code:`default_parameters: str`
   can be used in the way of the current :code`backend` parameter. for each backend, as well as for the keys
   :code:`'F'` and :code:`'C'`, a default parameter set is provided. Not all default parameter sets provide defaults
   for all other parameters. defining the other arguments explicitly overrides the defaults
- :code:`halo: Sequence[int]`
   Sequence of length :code:`3` or :code:`ndim`, each entry is either an int or a 2-tuple of ints. ints represent a
   symmetric halo in that dimension, while a 2-tuple specifies the halo on the respective boundary for that dimension.
   defaults to no halo, i.e. :code:`(0, 0, 0)`
- :code:`shape: Iterable[int]`
   iterable of ints, the shape of the storage
- :code:`np.dtype`
   the dtype of the storage (numpy-like)
- :code:`axes: str`
  string, permutation of a sub-sequence of "IJK", indicating the spatial dimensions along which the field extends and
   their order when indexing.
- :code:`aligned_index: Sequence[int]`
   the point to which the memory is aligned, defaults to the lower indices of the halo attribute
- :code:`alignment_size`
   integer, indicates on a boundary of how many elements the point :code:`alignment_index` is aligned. defaults to
   :code:`1` which indicates no alignment
- :code:`gpu: bool`
   boolean, indicates whether the storage has a GPU buffer, defaults to :code:`False`
- :code:`layout_map`
   iterable of numbers or a callable returning such an iterable when given the number of dimensions. The iterable
   indicates the order of strides in decreasing order, i.e. the entry :code:`0` in the iterable corresponds to the
   dimension with the largest stride. The layout map is always of length 3, and the entries corresponds to the axes in
   "IJK" order. Default values may however depend on the order of the axes.
- :code:`managed`
   :code:`False`, :code:`"gt4py"` or :code:`"cuda"`, optional. only has effect if :code:`gpu=True`
   defaults to "gt4py". can be used to choose whether the copying to GPU is handled by the user (:code:`False`),
   GT4Py (:code:`"gt4py"`) or CUDA (:code:`"cuda"`).

If a parameter is not explicitly specified, it is inferred from the default parameter set. If there is no default
parameter set provided or it does not provide the required information, it is gathered from the :code:`data` or
:code:`device_data` parameters. If this does not provide this information, a trivial default value is assumed. If no
default value is available, an error is raised that the parameters are underdetermined.

If :code:`copy=False` and neither :code:`data` nor :code:`device_data` are provided, the other arguments are used to
allocate an appropriate buffer. If :code:`data` or :code:`device_data` is provided, the consistency of the parameters
with the buffers is validated.

If the field is not 3-D, as indicated by :code:`axes`, the length of parameters :code:`aligned_index` and
:code:`shape`, may either be of length 3 or of the actual dimension of the storage, where the not needed entries are
ignored in the latter case.

We further expose the :code:`Storage` base class, mainly to enable type checking.

Storage Attributes and NumPy API functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While we aim at supporting as many features as possible, we have not compiled an exhaustive list of features yet and we
expressly ask for suggestions here (focusing on NumPy functions of the form :code:`np.function` or attributes and
methods of ndarrays of the form :code:`ndarray.attribute` or :code:`ndarray.method()`.)

Supported numpy functions:

:code:`np.all`, :code:`np.any`
   same semantics as :code:`np.logical_and.reduce` and :code:`np.logical_or.reduce`, respectively
:code:`np.transpose`
   It permutes the axes.

.. _constructors:

Attributes
==========
:code:`Storage` s have the following attributes:

:code:`dtype`
   the dtype as numpy dtype
:code:`ndim`
   number of (unmasked) dimensions
:code:`shape`
    tuple of length :code:`ndim`, the shape, with entries corresponding to the axes indicated by :code:`axes`
:code:`strides`
    tuple of length :code:`ndim`, the strides, with entries corresponding to the axes indicated by :code:`axes`
:code:`data`, :code:`flags`
   returns :code:`data` attribute of the underlying numpy ndarray if a main memory buffer is present, :code:`None`
   otherwise
:code:`device_data`
   returns :code:`data` attribute of the underlying cupy ndarray if a gpu buffer is present, :code:`None`
   otherwise
:code:`alignment`
   the value given in the constructor
:code:`axes`
   string of unmasked axes, e.g. :code:`"JI"` for a 2d field spanning longitude and latitude but not the vertical, where
   the first index corresponds to the "J" axis.
:code:`aligned_index`
   the value given in the constructor indicating the grid point to which the memory is aligned. Note that this only
   partly takes the role of the former :code:`default_origin` parameter, since that functionaly is now taken over by the
   :code:`halo` attribute.
:code:`nbytes`,
   size of the buffer in bytes (excluding padding)
:code:`gpu`
   boolean, indicating whether the storage has a gpu buffer
:code:`halo`
   n-dimensional tuple of 2-tuples of ints, in the same format as the halo parameter of the constructor methods.
   this property has a corresponding setter
:code:`domain_shape`
   the shape of the inner part of the field, i.e. the shape with the halo subtracted.
:code:`domain_view`
   a view of the buffer, again as a storage, with the halo removed. That is, the index :code:`[0, 0, 0]` corresponds
   to the first point in the domain.

Methods
=======

:code:`__array__()`
   returns either a numpy ndarray (if a CPU buffer is available), or a cupy ndarray otherwise

:code:`__array_interface__`
    only supported for storages with an actual CPU buffer

:code:`__cuda_array_interface__`
   only for GPU-enabled storages.

:code:`__deepcopy__` and :code:`copy` methods
   allocate new buffers and copy the contents

:code:`__getitem__`
   dimensions, for which a certain index is selected are returned as masked, while slices do not reduce dimensionality.
   advanced indexing is not supported, since the result would be a 1-d buffer rather than a field.

:code:`__setitem__`
   :ref:`broadcasting: and device selection is equivalent to that of a unary ufunc with a provided output buffer.
   For example, :code:`stor_out[:,3:5, 0] = stor2d` would be equivalent to
   :code:`np.positive(stor2d, out=stor_out[:,3:5, 0]`)
   advanced indexing is supported in assignments

:code:`to_ndarray`
   returns a view of the buffer which is a cupy ndarray if a storage is GPU enabled, and a numpy ndarray otherwise.
:code:`to_numpy`, :code:`to_cupy`
   returns a view of the buffer which is a view of the underlying buffers in numpy or cupy, or raises an exception
   if no buffer is available on the respective device.

The following methods are used to ensure one-sided modifications to CPU or GPU buffers of the
`SoftwareManagedGPUStorage` are tracked properly. They are no-ops for all other storage classes, but are there so that
user code can be backend-agnostic in these cases.

The use of these methods should only be necessary, if a reference to the storage buffers is kept and modified outside
of GT4Py, which is generally not recommended.

:code:`set_device_modified`, :code:`set_host_modified`, :code:`set_device_synchronized`
   mark a buffer as modified, so that it can be synchronized before the respective other buffer is accessed.

:code:`host_to_device` (:code:`device_to_host`)
   Triggers a copy from host (device) buffer to the sibling in device (host) memory, if the host (device) is marked as
   modified or the method is called with `force=True`. After a call to either of these methods, the buffers are flagged
   as synchronized.

:code:`synchronize`
   Triggers a copy between host and device buffers if the host or device, respectively are marked as modified. The
   buffers are marked as in sync as a consequence.


Universal Functions
^^^^^^^^^^^^^^^^^^^

Universal functions, such as mathematical binary operations and logical operators are supported through the
:code:`numpy.lib.mixins.NDArrayOperatorsMixin` base type and the `__array_ufunc__` interface. We support the methods
`__call__` and `reduce` of the numpy ufunc mechanism.

If the :code:`reduce` method of ufuncs is used, this results in a Storage with the dimensions masked along which the
reduction was performed. (e.g. taking the sum over the K axis of an IJK storage will result in an IJ storage)

.. _broadcasting:

Broadcasting
============

With the term "broadcasting", NumPy describes the ways that different shapes are combined in assignments and
mathematical operations. We override the default NumPy behavior so that fields are broadcast along the same spatial
dimension. I.e. adding an :code:`IJ` field :code:`A` of shape :code:`(2, 3)` with a :code:`K` field :code:`B` of shape
:code:`(4,)` will result in an :code:`IJK` field :code:`C` of shape :code:`(2, 3, 4)`, with `C[i,j,k] = A[i,j]+B[k]`.

Similarly, fields of lower dimension are assigned to such of higher dimension by broadcasting along the missing
dimensions.

To keep compatibility with numpy, dimensions of size 1 are treated like masked dimension when broadcasting.

Further, the output buffer can have higher dimensionality than the determined broadcast shape. In this case, the result
is replicated along the missing dimensions.


.. _output_storage_parameters:

Output Storage Parameters
=========================

If no output buffer is provided, the constructor parameters of the output storage have to be inferred using the
available information from the inputs.

:code:`aligned_index`
   it is chosen to be as the largest value per dimension across all inputs which are a GT4Py Storage
:code:`halo`
   it is chosen s.t. the resulting domain is the intersection of all individual domains.
:code:`layout_map`
   the layout map is chosen as the layout map of the first input argument which is a GT4Py Storage
:code:`axes`
   if the :code:`axes` parameters of all operands agree, the output will have the same :code:`axes`.
   otherwise, the axes are chosen as the union of all input storages. the order will be a 
   sub-sequence of "IJK" in this case.
:code:`alignment`
   the resulting alignment is chosen as the least common multiple of the alignments of all inputs which are a GT4Py
   Storage
:code:`dtype`
   the resulting dtype is determined by NumPy behavior


Mixing Types
============

If a binary ufunc is applied to a storage and a non-storage array, the storage determines the behavior.
Since non-storage arrays do not carry the necessary information to apply the usual broadcasting rules,
we only implement the cases where

* the array has the same shape as the input storage or as the broadcast shape when considering a provided output buffer
* the array has a 3d shape where dimensions with shape :code:`1` in the array are broadcast.

Mixing Devices
==============

For the synchronized memory classes (be it by CUDA or by GT4Py), the compute device is chosen depending on

:code:`CudaManagedGPUStorage`
   The compute device is chosen to be GPU iff inputs are comptaible with `cp.ndarray`.

:code:`SoftwareManagedGPUStorage`
   Here, array is considered a GPU array if it is compatible with :code:`cp.asarray`. If a storage is modified on CPU,
   it is considered a CPU array here. The compute device is chosen as GPU unless all inputs are not GPU arrays.
   (including if all inputs are :code:`SoftwareManagedGPUStorage` but are modified on CPU)

We assume that mixing these in the same application is not a common case. Should it nevertheless appear, the object that
handles the ufunc will determine the behavior. (Where each of the classes will treat the other as on GPU.)

For pure CPU storages, all inputs and output need to be compatible with `np.asarray`, for GPU storages with `cp.asarray`,
otherwise an exception is raised.

:code:`CudaManagedGPUStorage` and :code:`SoftwareManagedGPUStorage` shall both have a :code:`__array_priority__` set to
:code:`11`, while for :code:`CPUStorage` and :code:`GPUStorage` it is set to :code:`10`, meaning that managed storages
have priority in handling these cases.

Annotation of Stencils
^^^^^^^^^^^^^^^^^^^^^^

Currently, field arguments are annotated with :code:`Field[dtype]` in the function signature. The assumed layout and
alignment in the generated code is then based on the :code:`backend` parameter of the :code:`stencil` decorator.
This will continue to work, but in case the storage passed at call-time uses other settings than the backend's default
settings, these must also be specified to the stencil. We propose the following arguments for the :code:`Field`
annotation, where dtype and axes are specified as positional arguments, while the others use the notation
(:code:`Argument[value]`):

:code:`dtype`
   correspoinds to the `dtype` argument, can alternatively be a placeholder string, which can be bound to a dtype using
   the :code:`dtypes` parameter in the stencil decorator.
:code:`axes`
   corresponds to the `axes` argument. Note that the order of the axes here only indicates what the order is of the
   axes of the storages which are passed as a field at call time. In gtscript, offset-indexing is always in order 'IJK'.
:code:`LayoutMap`
   corresponds to the `layout_map` argument
:code:`Alignment`
   corresponds to the `alignment` argument
:code:`DefaultParameters`
   corresponds to the `default_parameters` argument.
   Either :code:`'F'` for FORTRAN layout, :code:`'C'` for C/C++-layout or one of the backend identifier strings.

The dtype is required, all others optional. The dtype and axes are specified as positional arguments, while all others
have to be specified using the bracket notation. If any parameter is specified both explicitly and in the default
parameter set, the explicit value takes precedence. All symbols, including the `Axes` arguments can be imported from
:code:`gt4py.gtscript`. If any of the parameters :code:`LayoutMap`, :code:`Alignment`, :code:`DefaultParameters` is
specified, the backend has no influence on these parameters for that field. If however none of those are specified,
the behavior is the same if only the dtype, optionally the axes and the :code:`DefaultParameters` of the backend
are specified.

.. note::
   While the storage constructors take the `gpu` argument, it is not necessary to declare this in the stencil
   signature. The compute device is a property of the backend and can not be set on a per-field basis. If a storage
   with only a CPU (GPU) buffer is passed to a stencil which is computed on GPU (CPU), an exception is raised.

Examples
========

For a single-precision 3d field which was allocated in FORTRAN without taking further care about alignment, a simple
copy-stencil could then read:

.. code-block:: python

   import numpy as np
   from gt4py import gtscript
   from gtscript import Field, DefaultParameters

   FieldAnnotation = Field[np.float32, DefaultParameters['F']]

   @gtscript.stencil(backend="debug")
   def copy(field_in: FieldAnnotation, field_out: FieldAnnotation):
       field_out[...] = field_in

For a storage which is compatible with the default layout of the :code:`"gtmc"` backend, the annotation could instead
be defined as :code:`FieldAnnotation = Field[DType[np.float32], Alignment:[8], LayoutMap[(0, 2, 1)]]`.
However, if the backend actually is :code:`backend="gtmc"`, the following will continue to work:
:code:`FieldAnnotation = Field[np.float32]`

.. note::
    Both currently and with the implementation of this GDP, fields with masked axes can be specified. However, since
    they are not supported in the analysis and code generated yet, we decided to not enable this here yet either,
    but it shall be part of a later GDP.

Run-time Checks
---------------
When calling the stencil, an exception is raised if a field does not conform with the previously specified information,
if going forward would trigger undefined behavior. If it is safe to go on, only a warning is raised.

This implies that e.g. for the :code:`"debug"` and :code:`"numpy"` backends, the specification of the fields only ever
causes warnings, which may turn into exceptions for the compiled backends.

It is not required that the fields are actually gt4py storage containers, as long as they can be converted to NumPy or
CuPy ndarrays, respectively.


Implementation
--------------
Internally, all CPU buffers are kept as NumPy ndarrays, ufunc calls are forwarded after allocating the appropriate
output buffers. GPU buffers are stored as CuPy ndarrays, except for the :code:`CudaManagedGPUStorage`.

Universal functions are handled by inheriting from :code:`numpy.NDArrayOperatorsMixin` and implementing the
:code:`__array_ufunc__` interface, which will determine the proper broadcasting, output shape and compute device,
and then dispatch the actual computation to NumPy or CuPy, respectively. Other numpy API functions will be handled
by means of the :code:`__array_function__` protocol.

.. _storage_types:

Storage Types
^^^^^^^^^^^^^

Storages are objects whose type is a subclass of :code:`Storage`. Depending on the choice of the :code:`device` and
:code:`synchronize` attributes discussed in Section :ref:`constructors`, the type is one of :code:`CPUStorage`,
:code:`GT4PySyncedGPUStorage`, :code:`CUDASyncedGPUStorage` or :code:`GPUStorage`.

Their purpose is as follows:

:code:`CPUStorage`
    It holds a reference to a `NumPy <https://numpy.org/>`_ :code:`ndarray` plus
:code:`SoftwareManagedGPUStorage`
    Internally holds a reference to both a `NumPy <https://numpy.org/>`_ and a `CuPy <https://cupy.chainer.org/>`_
    :code:`ndarray`.
:code:`CUDAManagedGPUStorage`
    Internally holds a reference to a `NumPy <https://numpy.org/>`_ `ndarray`. The memory is however allocated as CUDA
    unified memory, meaning that the same memory can be accessed from GPU, and synchronization is taken care of by the
    CUDA runtime.
:code:`GPUStorage`
    Internally holds a reference to a `CuPy <https://cupy.chainer.org/>`_ `ndarray`. This storage does not have a CPU
    buffer.

Alternatives
------------

The different aspects of this proposal are

* construction from existing buffers
* duck array versus subclassing
* non-default layouts

We believe the former to be non-controversial. For the latter two, alternatives could be:


Duck Array Versus Subclassing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is in principle possible to implement the other aspects of this proposal as a NumPy subclass. We believe that this
imposes more limitations than the proposed options due to the issues mentioned in the introduction and believe that
not subclassing is the better option.

Non-default Layouts
^^^^^^^^^^^^^^^^^^^

Instead of the bracket notation, other notations could be implemented for declaring parameters in the stencil
signature. One option is to use slices, resulting in syntax like
:code:`FieldAnnotation = Field["dtype":np.float32, "alignment":8, "layout_map":(0, 2, 1)]`


Copyright
---------

This document has been placed in the public domain.
