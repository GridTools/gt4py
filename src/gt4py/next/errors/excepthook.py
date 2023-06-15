from . import formatting
from . import exceptions
from typing import Callable
import sys

def compilation_error_hook(fallback: Callable, type_: type, value: exceptions.CompilerError, tb):
    if issubclass(type_, exceptions.CompilerError):
        print("".join(formatting.format_compilation_error(type_, value.message, value.location_trace)), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)