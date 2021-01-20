# -*- coding: utf-8 -*-
import pathlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import gt4py
from gt4py.definitions import BuildOptions, StencilID
from gt4py.type_hints import AnnotatedStencilFunc, StencilFunc


if TYPE_CHECKING:
    from gt4py.backend.base import Backend as BackendType
    from gt4py.backend.base import CLIBackendMixin
    from gt4py.frontend.base import Frontend as FrontendType
    from gt4py.ir import StencilDefinition, StencilImplementation
    from gt4py.stencil_object import StencilObject


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
            **self.default_options_dict(definition_func)
        )
        self.backend: "BackendType" = (
            backend(self) if backend else gt4py.backend.from_name("debug")(self)
        )
        self.frontend: "FrontendType" = frontend or gt4py.frontend.from_name("gtscript")
        self.caching = gt4py.caching.strategy_factory("jit", self)
        self._build_data: Dict[str, Any] = {}
        self._externals: Dict[str, Any] = {}

    def build(self) -> Type["StencilObject"]:
        """Generate, compile and/or load everything necessary to provide a usable stencil class."""
        # load or generate
        stencil_class = None if self.options.rebuild else self.backend.load()
        if stencil_class is None:
            stencil_class = self.backend.generate()
        return stencil_class

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        """Generate the stencil source code, fail if backend does not support CLI."""
        return self.cli_backend.generate_computation()

    def generate_bindings(self, targe_language: str) -> Dict[str, Union[str, Dict]]:
        """Generate ``target_language`` bindings source, fail if backend does not support CLI."""
        return self.cli_backend.generate_bindings(targe_language)

    def with_caching(
        self: "StencilBuilder", caching_strategy_name: str, *args: Any, **kwargs: Any
    ) -> "StencilBuilder":
        """
        Fluidly set the caching strategy from the name.

        Parameters
        ----------
        caching_strategy_name:
            Name of the caching strategy to be passed to the factory.

        args:
            Passed through to the caching strategy factory

        kwargs:
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

    def with_changed_options(self: "StencilBuilder", **kwargs: Dict[str, Any]) -> "StencilBuilder":
        old_options = self.options.as_dict()
        # BuildOptions constructor expects ``impl_opts`` keyword
        # but BuildOptions.as_dict outputs ``_impl_opts`` key
        old_options["impl_opts"] = old_options.pop("_impl_opts")
        return self.with_options(**{**old_options, **kwargs})

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

    @classmethod
    def default_options_dict(
        cls, definition_func: Union[StencilFunc, AnnotatedStencilFunc]
    ) -> Dict[str, Any]:
        return {"name": definition_func.__name__, "module": definition_func.__module__}

    @classmethod
    def name_to_options_args(cls, name: Optional[str]) -> Dict[str, str]:
        """Check for qualified name, extract also module option in that case."""
        if not name:
            return {}
        components = name.rsplit(".")
        data = {"name": name}
        if len(components) > 1:
            data = {"module": components[0], "name": components[1]}
        return data

    @classmethod
    def nest_impl_options(cls, options_dict: Dict[str, Any]) -> Dict[str, Any]:
        impl_opts = options_dict.setdefault("impl_opts", {})
        # The following is not a dict comprehension because:
        # The backend-specific options (starting with ``_``) are nested under
        # options_dict["impl_opts"] *and at the same time removed* from
        # options_dict itself.
        for impl_key in (k for k in options_dict.keys() if k.startswith("_")):
            impl_opts[impl_key] = options_dict.pop(impl_key)
        return options_dict

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
    def definition_ir(self) -> "StencilDefinition":
        return self._build_data.get("ir") or self._build_data.setdefault(
            "ir", self.frontend.generate(self.definition, self.externals, self.options)
        )

    @property
    def implementation_ir(self) -> "StencilImplementation":
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

    @property
    def cli_backend(self) -> "CLIBackendMixin":
        from gt4py.backend.base import CLIBackendMixin

        if not isinstance(self.backend, CLIBackendMixin):
            raise RuntimeError("backend of StencilBuilder instance is not CLI enabled.")
        return self.backend
