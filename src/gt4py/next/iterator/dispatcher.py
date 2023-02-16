# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import functools
from typing import Any, Callable, Dict, List


# TODO test


class _fun_dispatcher:
    def __init__(self, dispatcher, fun) -> None:
        self.dispatcher = dispatcher
        self.fun = fun
        functools.update_wrapper(self, fun)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.dispatcher.key is None:
            return self.fun(*args, **kwargs)
        else:
            return self.dispatcher._funs[self.dispatcher.key][self.fun.__name__](*args, **kwargs)

    def register(self, key):
        self.dispatcher.register_key(key)

        def _impl(fun):
            self.dispatcher._funs[key][self.fun.__name__] = fun
            return fun

        # required for direct call to dispatched functions (see roundtrip).
        return _impl


class Dispatcher:
    def __init__(self) -> None:
        self._funs: Dict[str, Dict[str, Callable]] = {}
        self.key_stack: List[str] = []

    @property
    def key(self):
        return self.key_stack[-1] if self.key_stack else None

    def register_key(self, key):
        if key not in self._funs:
            self._funs[key] = {}

    def push_key(self, key):
        if key not in self._funs:
            raise RuntimeError(f"Key {key} not registered")
        self.key_stack.append(key)

    def pop_key(self):
        self.key_stack.pop()

    def clear_key(self):
        self.key_stack = []

    def __call__(self, fun):
        return self.dispatch(fun)

    def dispatch(self, fun):
        return _fun_dispatcher(self, fun)
