"""
GTScript import machinery.

Usage Example
-------------

::

    from gt4py import gtsimport

    # simple usage
    gtsimport.install()  # for allowing .gt.py everywhere in sys.path

    # advanced usage
    gtsimport.install(
        search_path=[<path1>, <path2>, ...],  # for allowing only in search_path
        generate_path=<mybuildpath>,  # for generating python modules in a specific dir
        in_source=False,  # set True to generate python modules next to gtscfipt files
    )

"""
import importlib
import pathlib
import sys
import tempfile
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


GTS_EXTENSIONS = [".gt.py"]
GTS_COMMENT = "# [GT] using-dsl: gtscript"
GTS_IMPORT = (
    "from gt4py.gtscript import *  "
    "# This file is automatically generated, changes may be overwritten."
)


def _add_extension(path: pathlib.Path, extension: str):
    """Add an extension to the filename of a path."""
    return path.parent / (path.name + extension)


class GtsFinder:
    """
    Implements the :class:`importlib.abc.PathFinder` protocol.

    This finder is responsible for finding GTScript files within a set of
    search paths.

    Parameters
    ----------
    search_path :
        Search path for `gtscript` sources, defaults to `sys.path`

    generate_path :
        Path to generate py modules in. use a temp dir by default

    in_source :
        If True, py modules are built next to the gtscript files
        (generate_path is ignored).
    """

    def __init__(
        self,
        search_path: Optional[List[Union[str, pathlib.Path]]] = None,
        generate_path: Optional[Union[str, pathlib.Path]] = None,
        in_source: bool = False,
    ):
        if in_source:
            self.generate_path = None
        elif generate_path:
            self.generate_path = pathlib.Path(generate_path)
        else:
            self.generate_path = pathlib.Path(tempfile.gettempdir())

        self.search_path = search_path

    def get_generate_path(self, src_file_path: pathlib.Path):
        """Find the out-of-source or in-source generate directory."""
        return self.generate_path or src_file_path.parent

    def iter_search_candidates(self, fullname, path):
        """Iterate possible source file paths."""
        search_paths = [p for p in self.search_path or sys.path]
        search_paths.extend(path or [])
        filepath = pathlib.Path(fullname.rsplit(".")[-1])
        for search_path in search_paths:
            search_path = pathlib.Path(search_path)
            for extension in GTS_EXTENSIONS:
                yield search_path.absolute() / _add_extension(filepath, extension)

    def find_spec(
        self, fullname, path=None, target=None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Create a module spec for the first matching source file path."""
        if fullname in sys.modules:
            return None

        for candidate in self.iter_search_candidates(fullname, path):
            if candidate.exists():
                spec = importlib.machinery.ModuleSpec(
                    name=fullname,
                    loader=GtsLoader(
                        fullname, candidate, generate_at=self.get_generate_path(candidate)
                    ),
                    origin=str(candidate),
                    is_package=False,
                )
                spec.has_location = True
                return spec

        return None

    def install(self):
        sys.meta_path.append(self)
        if self.search_path:
            sys.path.extend([str(p) for p in self.search_path])

    def load_module(self, fullname, path=None, target=None):
        if fullname in sys.modules:
            return sys.modules[fullname]
        spec = self.find_spec(fullname, path, target)
        if not spec:
            raise ImportError("could not find gtscript module %s", fullname)
        module = spec.loader.create_module(spec)
        spec.loader.exec_module(module)
        return module


class GtsLoader(importlib.machinery.SourceFileLoader):
    """
    Extend :class:`importlib.machinery.SourceFileLoader` for GTScript files.

    Generate a python module for a GTScript file and use the super class to
    load that instead.
    """

    def __init__(self, fullname: str, path: pathlib.Path, generate_at: pathlib.Path):
        self.module_file = generate_at / (path.stem.split(".")[0] + ".py")
        super().__init__(fullname, str(path.absolute()))

    @property
    def plpath(self):
        return pathlib.Path(self.path)

    def get_filename(self, fullname: str) -> str:
        """
        Generate a py module if an up to date one doesn't exist yet.

        Create the generation directory if necessary.
        Use file stats to check if the module needs to be updated.


        Parameters
        ----------
        fullname : `str`
            Dotted name corresponding to the gtscript module.

        Returns:

            The file path of the generated py module as a string
        """
        if not self.module_file.parent.exists():
            self.module_file.parent.mkdir(exist_ok=True)
        if not self.module_file.exists():
            self.module_file.touch()

        if self.path_stats(self.path) != self.path_stats(str(self.module_file.absolute())):
            self.module_file.write_text(self.get_source_code(fullname))
        return str(self.module_file)

    def get_source_code(self, fullname: str) -> str:
        return self.plpath.read_text().replace(GTS_COMMENT, GTS_IMPORT)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = ModuleType(name=spec.name)
        return module


def install(
    *,
    search_path: Optional[List[Union[str, pathlib.Path]]] = None,
    generate_path: Optional[Union[str, pathlib.Path]] = None,
    in_source: bool = False,
) -> GtsFinder:
    """
    Install GTScript import extensions.

    Parameters are passed through to the constructor of :py:class:`GtsFinder`.
    """
    finder = GtsFinder(search_path=search_path, generate_path=generate_path, in_source=in_source)
    finder.install()
    return finder


@contextmanager
def allow_import_gtscript(**kwargs: Any) -> Iterator:
    """
    Create a context within which GTScript extensions can be imported.

    Example
    -------
    .. code-block: python

        mystencil = None
        with allow_import_gtscript(search_path=[pathlib.Path("my/gtscript/extensions/")]):
            import some_stencil  # works
            mystencil = some_stencil.some_stencil

        ## use mystencil

        import some_other_stencil  # in the same directory as some_stencil.gt.py
        ## import error
    """
    backup_import_system: Tuple[
        List[str], List[importlib.abc.MetaPathFinder], Dict[str, ModuleType]
    ] = (
        sys.path.copy(),
        sys.meta_path.copy(),
        sys.modules.copy(),
    )
    try:
        yield install(**kwargs)
    finally:
        sys.path, sys.meta_path, sys.modules = backup_import_system
