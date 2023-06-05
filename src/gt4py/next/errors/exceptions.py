from gt4py.eve import SourceLocation


class CompilationError(SyntaxError):
    def __init__(self, location: SourceLocation, message: str):
        super().__init__(
            message,
            (
                location.source,
                location.line,
                location.column,
                None,
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