# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, Optional

from gt4py.definitions import FieldInfo
# TODO(eddied) Handle circular reference...
# from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject


class FutureStencil:
    """
    A stencil object that is compiled by another node in a distributed context.
    """

    _builder: Optional["StencilBuilder"] = None

    def __init__(self):
        self._stencil_object: Optional[StencilObject] = None
        self._sleep_time: float = 0.1
        self._timeout: float = 60.0

    @property
    def stencil_object(self) -> StencilObject:
        if not self._stencil_object:
            self.wait_for_cache_info()
        return self._stencil_object

    @property
    def field_info(self) -> Dict[str, FieldInfo]:
        return self.stencil_object.field_info

    def wait_for_cache_info(self):
        cache_info_path = self._builder.caching.cache_info_path
        time_elapsed = 0.0
        while not cache_info_path.exists() and time_elapsed < self._timeout:
            time.sleep(self._sleep_time)
            time_elapsed += self._sleep_time
        if time_elapsed >= self._timeout:
            node_id = self._builder.caching._distrib_ctx[0]
            raise RuntimeError(
                f"Timeout while waiting for stencil '{cache_info_path.stem}' to compile on R{node_id}"
            )

        stencil_class = self._builder.backend.load()
        self._stencil_object = stencil_class()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        (self.stencil_object)(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.stencil_object.run(*args, **kwargs)
