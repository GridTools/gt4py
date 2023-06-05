from gt4py.eve import SourceLocation
from typing import Any
from . import tools

class CompilationError(SyntaxError):
    def __init__(self, location: SourceLocation, message: str):
        try:
            source_code = tools.get_code_at_location(location)
        except ValueError:
            source_code = None
        super().__init__(
            message,
            (
                location.source,
                location.line,
                location.column,
                source_code,
                location.end_line,
                location.end_column
            )
        )

    @property
    def location(self):
        return SourceLocation(
            source=self.filename,
            line=self.lineno,
            column=self.offset,
            end_line=self.end_lineno,
            end_column=self.end_offset
        )


class UndefinedSymbolError(CompilationError):
    def __init__(self, location: SourceLocation, name: str):
        super().__init__(location, f"name '{name}' is not defined")


class UnsupportedPythonFeatureError(CompilationError):
    def __init__(self, location: SourceLocation, feature: str):
        super().__init__(location, f"unsupported Python syntax: '{feature}'")


class MissingParameterTypeError(CompilationError):
    def __init__(self, location: SourceLocation, param_name: str):
        super().__init__(location, f"parameter '{param_name}' is missing type annotations")


class InvalidParameterTypeError(CompilationError):
    def __init__(self, location: SourceLocation, param_name: str, type_: Any):
        super().__init__(location, f"parameter '{param_name}' has invalid type annotation '{type_}'")


class IncorrectArgumentCountError(CompilationError):
    def __init__(self, location: SourceLocation, num_expected: int, num_provided: int):
        super().__init__(location, f"expected {num_expected} arguments but {num_provided} were provided")


class UnexpectedKeywordArgError(CompilationError):
    def __init__(self, location: SourceLocation, provided_names: str):
        super().__init__(location, f"unexpected keyword argument(s) '{provided_names}' provided")


class MissingAttributeError(CompilationError):
    def __init__(self, location: SourceLocation, attr_name: str):
        super().__init__(location, f"object does not have attribute '{attr_name}'")