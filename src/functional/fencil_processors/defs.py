from dataclasses import dataclass
from typing import Sequence, Type


@dataclass
class ScalarParameter:
    name: str
    type_: Type


@dataclass
class BufferParameter:
    name: str
    dimensions: Sequence[str]
    scalar_type: Type


@dataclass
class Function:
    name: str
    parameters: Sequence[ScalarParameter | BufferParameter]


@dataclass
class LibraryDependency:
    name: str
    version: str


@dataclass
class SourceCodeModule:
    entry_point: Function
    source_code: str
    library_deps: Sequence[LibraryDependency]
    language: str


@dataclass
class BindingCodeModule:
    source_code: str
    library_deps: Sequence[LibraryDependency]
