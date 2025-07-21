# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import copy
import hashlib
import os
import pathlib
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol

from typing_extensions import deprecated

from gt4py import storage as gt_storage
from gt4py.cartesian import definitions as gt_definitions, utils as gt_utils
from gt4py.cartesian.backend import pyext_builder
from gt4py.cartesian.backend.module_generator import (
    BaseModuleGenerator,
    ModuleData,
    make_args_data_from_gtir,
)


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_builder import StencilBuilder
    from gt4py.cartesian.stencil_object import StencilObject


class Backend(abc.ABC):
    #: Backend name
    name: ClassVar[str]

    #: Backend-specific options:
    #:
    #:  + info:
    #:    - versioning: is versioning on?
    #:    - description [optional]
    #:    - type
    options: ClassVar[dict[str, Any]]

    #: Backend-specific storage parametrization:
    #:
    #:  - "alignment": in bytes
    #:  - "device": "cpu" | "gpu"
    #:  - "layout_map": callback converting a mask to a layout
    #:  - "is_optimal_layout": callback checking if a storage has compatible layout
    storage_info: ClassVar[gt_storage.layout.LayoutInfo]

    #: Language support:
    #:
    #:  - "computation": name of the language in which the computation is implemented.
    #:  - "bindings": names of supported language bindings / wrappers.
    #:    If a high-level language is compatible out-of-the-box, it should not be listed.
    #:
    #:  Languages should be spelled using the official spelling
    #:  but lower case ("python", "fortran", "rust").
    languages: ClassVar[dict[str, Any] | None] = None

    # __impl_opts:
    #   "disable-code-generation": bool
    #   "disable-cache-validation": bool

    builder: StencilBuilder

    def __init__(self, builder: StencilBuilder) -> None:
        self.builder = builder

    @classmethod
    def filter_options_for_id(
        cls, options: gt_definitions.BuildOptions
    ) -> gt_definitions.BuildOptions:
        """Filter a copy for options that should not be part of the stencil ID."""
        filtered_options = copy.deepcopy(options)
        id_names = set(name for name, info in cls.options.items() if info["versioning"])
        non_id_names = set(filtered_options.backend_opts.keys()) - id_names
        for name in non_id_names:
            del filtered_options.backend_opts[name]

        return filtered_options

    @abc.abstractmethod
    def load(self) -> type[StencilObject] | None:
        """
        Load the stencil class from the generated python module.

        type:
            The generated stencil class after loading through python's import API


        This assumes that :py:meth:`Backend.generate` has been called already on the same stencil.
        The python module therefore already exists and any python extensions have been compiled.
        In case the stencil changed since the last generate call, it will not be rebuilt.
        """
        pass

    @abc.abstractmethod
    def generate(self) -> type[StencilObject]:
        """
        Generate the stencil class from GTScript's internal representation.

        Returns
        -------
        type:
            The generated stencil class after loading through python's import API

        In case the stencil class for the given ID is found in cache, this method can avoid
        rebuilding it. Rebuilding can however be triggered through the :code:`options` parameter
        if supported by the backend.
        """
        pass

    @property
    def extra_cache_info(self) -> dict[str, Any]:
        """Provide additional data to be stored in cache info file (subclass hook)."""
        return {}

    @property
    def extra_cache_validation_keys(self) -> list[str]:
        """List keys from extra_cache_info to be validated during consistency check."""
        return []


REGISTRY = gt_utils.Registry[type[Backend]]()


def from_name(name: str) -> type[Backend]:
    """Return a backend by name."""
    backend_cls = REGISTRY.get(name, None)
    if backend_cls is None:
        raise NotImplementedError(
            f"Backend '{name}' is not implemented. Options are: {REGISTRY.names}."
        )
    return backend_cls


def register(backend_cls: type[Backend]) -> type[Backend]:
    """Register a backend."""
    assert issubclass(backend_cls, Backend) and backend_cls.name is not None

    if isinstance(backend_cls.name, str):
        gt_storage.register(backend_cls.name, backend_cls.storage_info)
        return REGISTRY.register(backend_cls.name, backend_cls)

    raise ValueError(
        f"Invalid 'name' attribute ('{backend_cls.name}') in backend class '{backend_cls}'."
    )


class BaseBackend(Backend):
    MODULE_GENERATOR_CLASS: ClassVar[type[BaseModuleGenerator]]

    def load(self) -> type[StencilObject] | None:
        build_info = self.builder.options.build_info
        if build_info is not None:
            start_time = time.perf_counter()

        stencil_class = None
        if self.builder.stencil_id is not None:
            self.check_options(self.builder.options)
            validate_hash = not self.builder.options._impl_opts.get(
                "disable-cache-validation", False
            )
            if self.builder.caching.is_cache_info_available_and_consistent(
                validate_hash=validate_hash
            ):
                stencil_class = self._load()

        if build_info is not None:
            build_info["load_time"] = time.perf_counter() - start_time

        return stencil_class

    def generate(self) -> type[StencilObject]:
        self.check_options(self.builder.options)
        return self.make_module()

    def _load(self) -> type[StencilObject]:
        stencil_class_name = self.builder.class_name
        file_name = str(self.builder.module_path)
        stencil_module = gt_utils.make_module_from_file(stencil_class_name, file_name)
        stencil_class: type[StencilObject] = getattr(stencil_module, stencil_class_name)
        stencil_class.__module__ = self.builder.module_qualname
        stencil_class._gt_id_ = self.builder.stencil_id.version
        stencil_class.definition_func = staticmethod(self.builder.definition)

        return stencil_class

    def check_options(self, options: gt_definitions.BuildOptions) -> None:
        assert self.options is not None
        unknown_options = set(options.backend_opts.keys()) - set(self.options.keys())
        if unknown_options:
            warnings.warn(
                f"Unknown options '{unknown_options}' for backend '{self.name}'",
                RuntimeWarning,
                stacklevel=2,
            )

    def make_module(self) -> type[StencilObject]:
        build_info = self.builder.options.build_info
        if build_info is not None:
            start_time = time.perf_counter()

        file_path = self.builder.module_path
        module_source = self.make_module_source()

        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(module_source)
            self.builder.caching.update_cache_info()

        module = self._load()

        if build_info is not None:
            build_info["module_time"] = time.perf_counter() - start_time

        return module

    def make_module_source(self, *, args_data: ModuleData | None = None) -> str:
        """Generate the module source code with or without stencil id."""
        args_data = args_data or make_args_data_from_gtir(self.builder.gtir_pipeline)
        return self.MODULE_GENERATOR_CLASS(self.builder)(args_data)


class MakeModuleSourceCallable(Protocol):
    def __call__(self, *, args_data: ModuleData | None = None) -> str: ...


class BasePyExtBackend(BaseBackend):
    @property
    def pyext_module_name(self) -> str:
        return self.builder.module_name + "_pyext"

    @property
    def pyext_module_path(self) -> str:
        return self.builder.module_qualname + "_pyext"

    @property
    def pyext_class_name(self) -> str:
        return self.builder.class_name + "_pyext"

    @property
    def pyext_build_dir_path(self) -> pathlib.Path:
        return self.builder.pkg_path.joinpath(self.pyext_module_name + "_BUILD")

    @property
    def extra_cache_info(self) -> dict[str, Any]:
        pyext_file_path = self.builder.backend_data.get("pyext_file_path", None)
        pyext_md5 = ""
        if pyext_file_path:
            pyext_md5 = hashlib.md5(pathlib.Path(pyext_file_path).read_bytes()).hexdigest()
        return {
            **super().extra_cache_info,
            "pyext_file_path": pyext_file_path,
            "pyext_md5": pyext_md5,
        }

    @property
    def extra_cache_validation_keys(self) -> list[str]:
        keys = super().extra_cache_validation_keys
        if self.extra_cache_info["pyext_md5"]:
            keys.append("pyext_md5")
        return keys

    @abc.abstractmethod
    def generate(self) -> type[StencilObject]:
        pass

    def build_extension_module(
        self,
        pyext_sources: dict[str, Any],
        pyext_build_opts: dict[str, str],
        *,
        uses_cuda: bool = False,
    ) -> None:
        # Build extension module
        pyext_build_path = pathlib.Path(
            os.path.relpath(self.pyext_build_dir_path, pathlib.Path.cwd())
        )
        pyext_build_path.mkdir(parents=True, exist_ok=True)
        sources = []
        for key, source in pyext_sources.items():
            src_file_path = pyext_build_path / key
            src_ext = src_file_path.suffix
            if src_ext not in [".h", ".hpp"]:
                sources.append(str(src_file_path))

            if source is not gt_utils.NOTHING:
                src_file_path.write_text(source)

        pyext_target_file_path = self.builder.pkg_path
        qualified_pyext_name = self.pyext_module_path

        pyext_build_args: dict[str, Any] = dict(
            name=qualified_pyext_name,
            sources=sources,
            build_path=str(pyext_build_path),
            target_path=str(pyext_target_file_path),
            **pyext_build_opts,
        )

        if uses_cuda:
            module_name, file_path = pyext_builder.build_pybind_cuda_ext(**pyext_build_args)
        else:
            module_name, file_path = pyext_builder.build_pybind_ext(**pyext_build_args)

        assert module_name == qualified_pyext_name

        self.builder.with_backend_data(
            {"pyext_module_name": module_name, "pyext_file_path": file_path}
        )


def disabled(message: str, *, enabled_env_var: str) -> Callable[[type[Backend]], type[Backend]]:
    # We push for hard deprecation here by raising by default and warning if enabling has been forced.
    enabled = bool(int(os.environ.get(enabled_env_var, "0")))
    if enabled:
        return deprecated(message)

    def _decorator(cls: type[Backend]) -> type[Backend]:
        def _no_generate(obj) -> type[StencilObject]:
            raise NotImplementedError(
                f"Disabled '{cls.name}' backend: 'f{message}'\n",
                f"You can still enable the backend by hand using the environment variable '{enabled_env_var}=1'",
            )

        # Replace generate method with raise
        if not hasattr(cls, "generate"):
            raise ValueError(f"Coding error. Expected a generate method on {cls}")
        # Flag that it got disabled for register lookup
        cls.disabled = True  # type: ignore
        cls.generate = _no_generate  # type: ignore
        return cls

    return _decorator
