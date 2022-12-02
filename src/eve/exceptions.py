# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Definitions of specific Eve exceptions."""


from __future__ import annotations

from .extended_typing import Any, Dict, Optional


class EveError:
    """Base class for Eve-specific exceptions.

    Notes:
        This base class has to be always inherited together with a standard
        exception, and thus it should not be used as direct superclass
        for custom exceptions. Inherit directly from :class:`EveTypeError`,
        :class:`EveTypeError`, etc. instead.

    """

    message_template = "Generic Eve error [{info}]"
    info: Dict[str, Any]

    def __init__(self, message: Optional[str] = None, **kwargs: Any) -> None:
        self.info = kwargs
        super().__init__(  # type: ignore  # super() call works as expected when using multiple inheritance
            message
            or type(self).message_template.format(
                **self.info, info=", ".join(f"{key}={value}" for key, value in self.info.items())
            )
        )


class EveTypeError(EveError, TypeError):
    """Base class for Eve-specific type errors."""

    message_template = "Invalid or unexpected type [{info}]"


class EveValueError(EveError, ValueError):
    """Base class for Eve-specific value errors."""

    message_template = "Invalid value [{info}]"


class EveRuntimeError(EveError, RuntimeError):
    """Base class for Eve-specific run-time errors."""

    message_template = "Runtime error [{info}]"
