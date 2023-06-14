import sys
import traceback
from . import exceptions
from typing import Callable


def compilation_error_hook(fallback: Callable, type_: type, value: exceptions.CompilationError, _):
    if issubclass(type_, exceptions.CompilationError):
        print("".join(traceback.format_exception(value, limit=0)), file=sys.stderr)
    else:
        fallback(type_, value, traceback)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)