# -*- coding: utf-8 -*-
from typing import Any, Dict

from typing_extensions import Protocol


class StencilFunc(Protocol):
    __name__: str
    __module__: str

    def __call__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        ...


class AnnotatedStencilFunc(StencilFunc, Protocol):
    _gtscript_: Dict[str, Any]
