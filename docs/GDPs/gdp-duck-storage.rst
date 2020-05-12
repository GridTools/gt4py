===========================
GDP 3 — Storage Refactoring
===========================

:Author: Linus Groner <linus.groner@cscs.ch>
:Status: Draft
:Type: Feature
:Created: 08-05-2020
:Discussion PR: TBD


Abstract
--------

We propose to replace the current storage classes by a class which does not inherit
from NumPy ndarrays.
Further, the new storage classes shall introduce the possibility to be constructed
from existing memory and provide more control over the underlying memory.


Motivation and Scope
--------------------

In the current state of GT4Py, we implemented storages as subclasses of NumPy :code:`ndarrays`.
This has some drawbacks such as the missing possibility to maintain some information about the buffers
under certain operations of the NumPy API. Further, some one-sided changes to buffers of
the `ExplicitlySyncedGPUStorage` could not be tracked reliably, resulting in validation errors which are hard to
debug and avoid.

Nevertheless, the storages are currently implemented as ndarray subclasses. This was necessary
so the storages could be used in some third party frameworks. Since
then, these frameworks have extended their compatibility to the interface that are
specified in the :emphasis:`Numpy Enhancement Proposal`
`NEP18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.

Implementing the interface introduced in the NEP18 allows GT4Py to retain full control
over the copying behavior, while allowing the use of the mentioned frameworks.

Besides this substantial change, we propose to take the opportunity of this development
to also improve the interaction with existing codes by allowing the initialization of
storages from external buffers without copying. To use this feature, some additional
information about the provided buffers needs to be specified to both the :code:`__init__`
method of the storages as well to the stencils at compile time.

Despite the move away from NumPy as a base class, the storage should integrate with the
scientific python ecosystem as seamlessly as possible and mimic a NumPy ndarray, which is an established standard,
by means of `duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_, to the extent that this is feasible.

Backward Compatibility
----------------------

The implementation of this GDP breaks the integration with external libraries which require a NumPy ndarray subclass.
Further, we propose some API changes like renaming or repurposing of keyword arguments and attributes.
This will break all existing user codes. However, all existing GT4Py functionality will remain or be extended. Updating
codebases to the new interface amounts mostly to updating attribute and keyword argument names.


Detailed Description
--------------------

Supported Functionality
^^^^^^^^^^^^^^^^^^^^^^^

Ideally, storages would behave in the same way as ndarrays. We limit this if it is not possible to maintain the
consistency of the underlying structured memory. We chose to internally use NumPy and CuPy ndarrays
to store CPU and GPU buffers, respectively. We also implement some functionality like mathematical operations by
forwarding to those libraries. Therefore, support for dtypes and other functionality is restricted to the common
denominator of CuPy and NumPy.


Storage Constructors
^^^^^^^^^^^^^^^^^^^^

There will be 6 functions that should be used to initialize the storages. These are:

:code:`empty`
   Used to allocate a storage with uninitialized (undefined) values.
:code:`zeros`
   Used to allocate a storage with values initialized to 0.
:code:`ones`
   Used to allocate a storage with values initialized to 1.
:code:`full`
   Used to allocate a storage with values initialized to a given scalar value.
:code:`asstorage`
   Used to wrap an existing buffer in a storage object, without copying the buffer's contents.
:code:`storage`
   Used to allocate a storage with values initialized to those of a given array. If the argument
   :code:`copy` is set to :code:`False`, the behavior is that of :code:`asstorage`.

All of these take the following keyword arguments:

:code:`default_parameters`
   can be used in the way of the current :code`backend` parameter. for each backend, as well as for the keys
   :code:`'F'` and :code:`'C'`, a default parameter set is provided. Not all default parameter sets provide defaults
   for all other parameters. defining the other arguments explicitly overrides the defaults
:code:`default_origin`
   the point to which the memory is aligned, as well as the point that is used as the origin of the storage if none is
   provided at stencil call time.
:code:`shape`
   iterable of ints, the shape of the storage
:code:`dtype`
   the dtype of the storage (numpy-like)
:code:`axes`
   string, subset of "IJK", indicating the spatial dimensions along which the field extends
:code:`alignment`
   integer, indicates on a boundary of how many elements the point :code:`default_origin` is aligned. defaults to
   :code:`1` which indicates no alignment
:code:`gpu`
   boolean, indicates whether the storage has a GPU buffer, defaults to :code:`False`
:code:`layout_map`
   iterable of numbers or a callable returning such an iterable when given the number of dimensions. the iterable
   indicates the order of strides in decreasing order, i.e. the entry :code:`0` in the iterable corresponds to the dimension
   with the largest stride.
:code:`managed`
   :code:`False`, :code:`"gt4py"`, :code:`"cuda:`, optional. only has effect if :code:`gpu=True`
   defaults to "gt4py". can be used to choose whether the copying to GPU is handled by the user (:code:`False`),
   GT4Py (:code:`"gt4py"`) or CUDA (:code:`"cuda"`).

In addition, some of the functions support additional positional or keyword arguments:

:code:`value`
   supported by the :code:`full` method. it indicates the value to which the array is initialized
:code:`data`
   supported by the :code:`asstorage` and :code:`storage` functions. It is used to specify the buffer from which the
   storage is initialized (with or without copying the values)
:code:`device_data`
   supported by the :code:`asstorage` and :code:`storage` functions. It is used to specify the device buffer in case
   allocation from existing buffers on both the device and main memory is desired.
:code:`sync_state`:
   gt4py.storage.SyncState, supported by the :code:`asstorage` and :code:`storage` functions,  only has effect if
   :code:`managed="gt4py"`. indicates which of the provided buffers (among :code:`data`, :code:`device_data`) is up to
   date at the time of initialization.
:code:`copy`
   Supported by the :code:`storage` function. It can be used to specify whether the value given by :code:`data` or
   :code:`device_data` is copied or not.

If a parameter is not explicitly specified, it is inferred from the default parameter set. If there is no default
parameter set provided or it does not provide the required information, it is gathered from the :code:`data` or
:code:`device_data` parameters. If this does not provide this information, a trivial default value is assumed. If no
default value is available, an error is raised that the parameters are underdetermined.

If :code:`copy=False` and neither :code:`data` nor :code:`device_data` are provided, the other arguments are used to
allocate an appropriate buffer. If :code:`data` or :code:`device_data` is provided, the consistency of the parameters
with the buffers is validated.

If the field is not 3-D, as indicated by :code:`axes`, the length of parameters :code:`default_origin` and
:code:`shape`, may either be of length 3 or of the actual dimension of the storage.

We further expose the :code:`Storage` base class, mainly to enable type checking. It can alternatively be used in the
same way as :code:`storage` to initialize storages. On the other hand, constructors of the derived, hardware-specific
storage types (See Section :ref:`storage_types`) are not intended to be used directly.


Storage Attributes
^^^^^^^^^^^^^^^^^^

While we aim at supporting as many features as possible, we have not compiled an exhaustive list of features yet and we
expressly ask for suggestions here (focusing on NumPy functions of the form :code:`np.function` or attributes and
methods of ndarrays of the form :code:`ndarray.attribute` or :code:`ndarray.method()`.)

Supported numpy functions:

:code:`np.all`, :code:`np.any`
   same semantics as :code:`np.logical_and.reduce` and :code:`np.logical_or.reduce`, respectively

Features not supported due to unclear semantics:

:code:`transpose`
   It does not seem to make sense to swap spatial dimensions

.. _constructors:

Properties
==========

:code:`ndims`
   number of (unmasked) dimensions
:code:`shape`
    tuple of length :code:`ndims`,
:code:`strides`
    tuple of length :code:`ndims`,
:code:`data`
   returns :code:`data` attribute of the underlying ndarray
:code:`alignment`
   the value given in the constructor
:code:`axes`
   string of unmasked axes, e.g. :code:`"IJ"` for a 2d field spanning longitude and latitude but not the vertical.
:code:`mask`
   similar to axes, but a tuple of booleans. :code:`(True, True, False)` would be a 2d field spanning longitude and
latitude but not the vertical

:code:`default_origin`
   the value given in the constructor indicating the grid point to which the memory is aligned.

:code:`gpu`
   boolean, indicating whether the storage has a gpu buffer

ToDo

Methods
=======

:code:`__array__`, :code:`__array_interface__` and :code:`__cuda_array_interface__`
   where the former two are only supported for storages with an actual CPU buffer, the latter only for GPU-enabled
   storages

:code:`__deepcopy__` and :code:`copy` methods
   allocate new buffers and copy the contents

:code:`__getitem__`
   dimensions, for which a certain index is selected are returned as masked, while slices do not reduce dimensionality.

:code:`__setitem__`
   :ref:`broadcasting: and device selection is equivalent to that of a unary ufunc with a provided output buffer.
   For example, :code:`stor_out[:,3:5, 0] = stor2d` would be equivalent to
   :code:`np.positive(stor2d, out=stor_out[:,3:5, 0]`)

:code:`copy`
   allocate new buffers and copy the contents

The following methods are used to ensure one-sided modifications to CPU or GPU buffers of the
`SoftwareManagedGPUStorage` are tracked properly. They are no-ops for all other storage classes, but are there so that
user code can be backend-agnostic in these cases.

The use of these methods should only be necessary, if a reference to the storage buffers is kept and modified outside
of GT4Py.

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
:code:`numpy.lib.mixins.NDArrayOperatorsMixin` type and the `__array_ufunc__` interface. We support the methods
`__call__` and `reduce` of the numpy ufunc mechanism.

If the :code:`reduce` method of ufuncs is used, this results in a Storage with the dimensions masked along which the
reduction was performed.

.. _broadcasting:

Broadcasting
============

With the term "broadcasting", NumPy describes the ways that different shapes are combined in assignments and
mathematical operations. We override the default NumPy behavior so that fields are broadcast along the same spatial
dimension. I.e. adding an :code:`IJ` field :code:`A` of shape :code:`(2, 3)` with a :code:`K` field :code:`B` of shape
:code:`(4,)` will result in an :code:`IJK` field :code:`C` of shape :code:`(2, 3, 4)`, with `C[i,j,k] = A[i,j]+B[k]`.

Similarly, fields of lower dimension are assigned to such of higher dimension by broadcasting along the missing
dimensions.

Further, the output buffer can have higher dimensionality than the determined broadcast shape. In this case, the result
is replicated along the missing dimensions.



Output Storage Parameters
=========================

If no output buffer is provided, the constructor parameters of the output storage have to be inferred using the
available information from the inputs.

:code:`default_origin`
   it is chosen to be as the largest value per dimension across all inputs which are a GT4Py Storage
:code:`layout_map`
   the layout map is chosen as the layout map of the first input argument which is a GT4Py Storage
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

For CPU storages, all inputs and output need to be compatible with `np.asarray`, for GPU storages with `cp.asarray`,
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
annotation, which are specified using the notation (:code:`Argument[value]`):

:code:`DType`
   correspoinds to the `dtype` argument
:code:`LayoutMap`
   corresponds to the `layout_map` argument
:code:`Alignment`
   corresponds to the `alignment` argument
:code:`DefaultParameters`
   corresponds to the `default_parameters` argument.
   Either :code:`'F'` for FORTRAN layout, :code:`'C'` for C/C++-layout or one of the backend identifier strings.
:code:`Axes`
   corresponds to the `axes` argument can be a string or as it is now one of :code:`I`, :code:`J`, :code:`K`,
   :code:`IJ` :code:`IK`, :code:`JK`, :code:`IJK`

The dtype is required, all others optional. The dtype and axes can be specified as positional arguments or using the
bracket notation, while all others have to be specified using the bracket notation. If any parameter is specified both
explicitly and in the default parameter set, the explicit value takes precedence. All symbols, including the `Axes`
arguments can be imported from :code:`gt4py.gtscript`.

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
if going forward would trigger undefined behavior. If it is safe to go on only a warning is raised.

This implies that e.g. for the :code:`"debug"` and :code:`"numpy"` backends, the specification of the fields only ever
causes warnings, which may turn into exceptions for the compiled backends.

It is not required that the fields are actually gt4py storage containers, as long as they can be converted to NumPy or
CuPy ndarrays, respectively.


Implementation
--------------
Internally, all CPU buffers are kept as NumPy ndarrays, ufunc calls are forwarded after allocating the appropriate
output buffers. GPU buffers are stored as CuPy ndarrays.

Universal functions are handled by inheriting from :code:`numpy.NDArrayOperatorsMixin` and implementing the
:code:`__array_ufunc__` interface, which will determine the proper broadcasting, output shape and compute device,
and then dispatch the actual computation to NumPy or CuPy, respectively. Other numpy API functions will be handled
by means of the :code:`__array_function__` protocol.

ToDo

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
