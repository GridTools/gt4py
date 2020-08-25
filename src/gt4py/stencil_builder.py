import pathlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import gt4py
from gt4py.definitions import BuildOptions, StencilID
from gt4py.ir import StencilDefinition, StencilImplementation
from gt4py.type_hints import AnnotatedStencilFunc, StencilFunc


if TYPE_CHECKING:
    from gt4py.backend.base import Backend as BackendType
    from gt4py.frontend.base import Frontend as FrontendType


class StencilBuilder:
    """
    Orchestrates code generation and compilation.

    Parameters
    ----------
    definition_func:
        Stencil definition function, before or after annotation.

    backend:
        Backend class to be instanciated for this build

    options:
        Build options, defaults to name and module of the :code:`definition_func`

    frontend:
        Frontend class to be used

    Notes
    -----
    Backends can use the :py:meth:`with_backend_data` method the :py:meth:`backend_data` property
    to set and retrieve backend specific information for use in later steps.

    """

    def __init__(
        self,
        definition_func: Union[StencilFunc, AnnotatedStencilFunc],
        *,
        backend: Optional[Type["BackendType"]] = None,
        options: Optional[BuildOptions] = None,
        frontend: Optional["FrontendType"] = None,
    ):
        self._definition = definition_func
        # type ignore explanation: Attribclass generated init not recognized by mypy
        self.options = options or BuildOptions(  # type: ignore
            name=definition_func.__name__, module=definition_func.__module__
        )
        self.backend: "BackendType" = backend(self) if backend else gt4py.backend.from_name(
            "debug"
        )(self)
        self.frontend: "FrontendType" = frontend or gt4py.frontend.from_name("gtscript")
        self.caching = gt4py.caching.strategy_factory("jit", self)
        self._build_data: Dict[str, Any] = {}
        self._externals: Dict[str, Any] = {}

    def with_caching(
        self: "StencilBuilder", caching_strategy_name: str, *args: Any, **kwargs: Any
    ) -> "StencilBuilder":
        """
        Fluidly set the caching strategy from the name.

        Parameters
        ----------
        caching_strategy_name:
            Name of the caching strategy to be passed to the factory.

        *args:
            Passed through to the caching strategy factory

        **kwargs:
            Passed through to the caching strategy factory

        Notes
        -----
        Resets all cached build data.
        """
        self._build_data = {}
        self.caching = gt4py.caching.strategy_factory(caching_strategy_name, self, *args, **kwargs)
        return self

    def with_options(
        self: "StencilBuilder", *, name: str, module: str, **kwargs: Any
    ) -> "StencilBuilder":
        """
        Fluidly set the build options.

        Parameters
        ----------
        name:
            Name of the stencil

        module:
            qualified module name of the stencil's module

        kwargs:
            passed through to the options instance

        """
        self._build_data = {}
        # mypy attribkwclass bug
        self.options = BuildOptions(name=name, module=module, **kwargs)  # type: ignore
        return self

    def with_backend(self: "StencilBuilder", backend_name: str) -> "StencilBuilder":
        """
        Fluidly set the backend type from backend name.

        Parameters
        ----------
        backend_name:
            Name of the backend type.

        Notes
        -----
        Resets all cached build data.


        """
        self._build_data = {}
        self.backend = gt4py.backend.from_name(backend_name)(self)
        return self

    @property
    def definition(self) -> AnnotatedStencilFunc:
        return self._build_data.get("prepared_def") or self._build_data.setdefault(
            "prepared_def",
            self.frontend.prepare_stencil_definition(self._definition, self.externals),
        )

    @property
    def externals(self) -> Dict[str, Any]:
        return self._build_data.get("externals") or self._build_data.setdefault(
            "externals", self._externals.copy()
        )

    def with_externals(self: "StencilBuilder", externals: Dict[str, Any]) -> "StencilBuilder":
        """
        Fluidly set externals for this build.

        Resets all cached build data.
        """
        self._build_data = {}
        self._externals = externals
        return self

    @property
    def backend_data(self) -> Dict[str, Any]:
        return self._build_data.get("backend_data", {}).copy()

    def with_backend_data(self: "StencilBuilder", data: Dict[str, Any]) -> "StencilBuilder":
        self._build_data["backend_data"] = {**self.backend_data, **data}
        return self

    @property
    def stencil_id(self) -> StencilID:
        return self._build_data.setdefault("id", self.caching.stencil_id)

    @property
    def root_pkg_name(self) -> str:
        return self._build_data.setdefault(
            "root_pkg_name", gt4py.config.code_settings["root_package_name"]
        )

    def with_root_pkg_name(self: "StencilBuilder", name: str) -> "StencilBuilder":
        self._build_data["root_pkg_name"] = name
        return self

    @property
    def pkg_name(self) -> str:
        return self.options.name

    @property
    def pkg_qualname(self) -> str:
        return self.root_pkg_name + "." + self.options.qualified_name

    @property
    def pkg_path(self) -> pathlib.Path:
        return self.caching.backend_root_path.joinpath(*self.options.qualified_name.split("."))

    @property
    def definition_ir(self) -> StencilDefinition:
        return self._build_data.get("ir") or self._build_data.setdefault(
            "ir", self.frontend.generate(self.definition, self.externals, self.options)
        )

    @property
    def implementation_ir(self) -> StencilImplementation:
        return self._build_data.get("iir") or self._build_data.setdefault(
            "iir", gt4py.analysis.transform(self.definition_ir, self.options)
        )

    @property
    def module_name(self) -> str:
        return self.caching.module_prefix + self.options.name + self.caching.module_postfix

    @property
    def module_qualname(self) -> str:
        return f"{self.pkg_qualname}.{self.module_name}"

    @property
    def module_path(self) -> pathlib.Path:
        file_name = self.module_name + ".py"
        return self.pkg_path / file_name

    @property
    def stencil_source(self) -> str:
        """
        Read the stencil source from the written module file.

        Raises
        ------
        :py:class:`FileNotFoundError`
            if the file has not been written yet
        """
        if not self.module_path.exists():
            return ""
        return self.module_path.read_text()

    @property
    def class_name(self) -> str:
        return self.caching.class_name

    @property
    def is_build_data_empty(self) -> bool:
        return not bool(self._build_data)
