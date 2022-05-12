# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from . import core as core, validators as validators  # noqa: F401  # imported but unused
from .core import *  # noqa:   # star unused import


"""Data Model classes and related utils and validators.

Data Models can be considered as enhanced  `attrs <https://www.attrs.org>`_
/ `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ providing
additional features like automatic run-time type validation. Values assigned to fields
at initialization can be validated with automatic type checkings using the
field type definition. Custom field validation methods can also be added with
the :func:`validator` decorator, and global instance validation methods with
:func:`root_validator`.

The datamodels API tries to follow ``dataclasses`` API conventions when possible,
but it is not an exact copy. Implementation-wise, Data Model classes are just
customized ``attrs`` classes, and therefore external tools compatible with ``attrs``
classes should also work with Data Models classes.

A valid ``__init__`` method for the Data Model class is always generated. If the class
already defines a custom ``__init__`` method, the generated method will be named
``__auto_init__`` and should be called from the custom ``__init__`` to profit from
datamodels features. Additionally, if custom ``__pre_init__(self) -> None`` or
``__post_init__(self) -> None`` methods exist in the class, they will be automatically
called from the generated ``__init__`` before and after the instance creation.

The order of execution of the different components at initialization is the following:

    1. ``__init__()``

        a.  If a custom ``__init__`` already exists in the class, it will not be overwritten.
            It is your responsability to call ``__auto_init__`` from there to obtain
            the described behavior.
        b.  If there is not custom ``__init__``, the one generated by datamodels
            will be called first.

    2. ``__pre_init__()``, if it exists.
    3. For each field, in the order it was defined:

        a. default factory
        b. converter

    4. all field validators
    5. root validators following the reversed class MRO order
    6. ``__post_init__()``, if it exists


Examples:
    >>> @datamodel
    ... class SampleModel:
    ...     name: str
    ...     amount: int
    ...
    ...     @validator('name')
    ...     def _name_validator(self, attribute, value):
    ...         if len(value) < 3:
    ...             raise ValueError(
    ...                 f"Provided value '{value}' for '{attribute.name}' field is too short."
    ...             )

    >>> SampleModel("Some Name", 10)
    SampleModel(name='Some Name', amount=10)

    >>> SampleModel("A", 10)
    Traceback (most recent call last):
        ...
    ValueError: Provided value 'A' for 'name' field is too short.

    >>> class AnotherSampleModel(DataModel):
    ...     name: str
    ...     friends: List[str]
    ...
    ...     @root_validator
    ...     def _root_validator(cls, instance):
    ...         if instance.name in instance.friends:
    ...             raise ValueError("'name' value cannot appear in 'friends' list.")

    >>> AnotherSampleModel("Eve", ["Liam", "John"])
    AnotherSampleModel(name='Eve', friends=['Liam', 'John'])

    >>> AnotherSampleModel("Eve", ["Eve", "John"])
    Traceback (most recent call last):
        ...
    ValueError: 'name' value cannot appear in 'friends' list.

    >>> @datamodel
    ... class CustomModel:
    ...     value: float
    ...     num_instances: ClassVar[int] = 0
    ...
    ...     def __init__(self, a: int, b: int) -> None:
    ...         self.__auto_init__(a/b)
    ...
    ...     def __pre_init__(self) -> None:
    ...         self.__class__.num_instances += 1
    ...
    ...     def __post_init__(self) -> None:
    ...         print(f"Instance {self.num_instances} == {self.value}")

    >>> CustomModel(3, 2)
    Instance 1 == 1.5
    CustomModel(value=1.5)

"""
