# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Caching strategies for stencil generation."""

from __future__ import annotations

import abc
import inspect
import pathlib
import pickle
import sys
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cached_property import cached_property

from gt4py.cartesian import config as gt_config, utils as gt_utils
from gt4py.cartesian.definitions import StencilID


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_builder import StencilBuilder


class CachingStrategy(abc.ABC):
    name: str

    def __init__(self, builder: StencilBuilder):
        self.builder = builder

    @property
    @abc.abstractmethod
    def root_path(self) -> pathlib.Path:
        """Calculate and if necessary create the root path for stencil generation operations."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def backend_root_path(self) -> pathlib.Path:
        """
        Calculate and if necessary create the caching base path for the backend being used.

        Avoid trying to create the root_path automatically.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cache_info_path(self) -> Optional[pathlib.Path]:
        """Calculate the file path where caching info for the current process should be stored."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_cache_info(self) -> Dict[str, Any]:
        """
        Generate the cache info dict.

        Backend specific additions can be added via a hook properly on the backend instance.
        Override :py:meth:`gt4py.backend.base.Backend.extra_cache_info` to store extra
        info.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_cache_info(self) -> None:
        """
        Regenerate cache info and update the cache info file.

        Parameters
        ----------
        extra_cache_info:
            Additional info to be written

        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_cache_info_available_and_consistent(self, *, validate_hash: bool) -> bool:
        """
        Check if the cache can be read and is consistent.

        If the cache info file can be read, try to validate it against the current info
        as returned by py:meth:`generate_cache_info`.

        Parameters
        ----------
        validate_hash:
            Should the stencil fingerprint and source code hash be validated or not
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stencil_id(self) -> StencilID:
        """Calculate the stencil ID used for naming and cache consistency checks."""
        pass

    @property
    @abc.abstractmethod
    def cache_info(self) -> Dict[str, Any]:
        """
        Read currently stored cache info from file into a dictionary.

        Empty if file is not found or not readable or no caching is intended.
        """
        pass

    @property
    def module_prefix(self) -> str:
        """
        Generate the prefix for stencil modules.

        Backends use this property to determine the stencil module filename.
        This may help distinguish python modules from other files.

        The default is an empty string and is suitable for situations where no caching
        is required and the generated module is intended to be imported manually.
        """
        return ""

    @property
    def module_postfix(self) -> str:
        """
        Generate the postfix for stencil modules, can depend on the fingerprint.

        Backends use this property when generating the module filename.

        .. code-block: python
            :title: Example

            @property
            def module_postfix(self) -> str:
                return f"_{self.stencil_id.version}"

        The example code would cause the generated module and class names to depend on the
        fingerprint of the stencil. That would ensure that after changing and regenerating a
        stencil it's module can be loaded without unloading the previous version first.  The
        default is empty and is suitable for when no caching is used.

        Can also be re-used elsewhere for similar purposes.
        """
        return ""

    @property
    def class_name(self) -> str:
        """Calculate the name for the stencil class, default is to read from build options."""
        return self.builder.options.name

    def capture_externals(self) -> Dict[str, Any]:
        """Extract externals from the annotated stencil definition for fingerprinting. Freezes the references."""
        return {}


class JITCachingStrategy(CachingStrategy):
    """
    Caching strategy for JIT stencil generation.

    Assign a fingerprint to the stencil being generated based on the definition function
    and some of the build options. During generation, store this fingerprint along with
    other information in a cache info file.

    In order to decide whether the stencil can be loaded from cache, check if a cache info file
    exists in the location corresponding to the current stencil. If it exists, compare it to
    the additional caching information for the current stencil. If the cache is consistent, a
    rebuild can be avoided.
    """

    name = "jit"

    _root_path: str
    _dir_name: str

    def __init__(
        self,
        builder: StencilBuilder,
        *,
        root_path: Optional[str] = None,
        dir_name: Optional[str] = None,
    ):
        super().__init__(builder)
        self._root_path = root_path or gt_config.cache_settings["root_path"]
        self._dir_name = dir_name or gt_config.cache_settings["dir_name"]

    @property
    def root_path(self) -> pathlib.Path:
        cache_root = pathlib.Path(self._root_path) / self._dir_name

        if not cache_root.exists():
            gt_utils.make_dir(str(cache_root), is_cache=True)

        return cache_root

    @property
    def backend_root_path(self) -> pathlib.Path:
        cpython_id = "py{version.major}{version.minor}_{api_version}".format(
            version=sys.version_info, api_version=sys.api_version
        )
        backend_root = self.root_path / cpython_id / gt_utils.slugify(self.builder.backend.name)
        if not backend_root.exists():
            if not backend_root.parent.exists():
                backend_root.parent.mkdir(parents=False, exist_ok=True)
            backend_root.mkdir(parents=False, exist_ok=True)
        return backend_root

    @property
    def cache_info_path(self) -> Optional[pathlib.Path]:
        """Get the cache info file path from the stencil module path."""
        return self.builder.module_path.parent / f"{self.builder.module_path.stem}.cacheinfo"

    def generate_cache_info(self) -> Dict[str, Any]:
        return {
            "backend": self.builder.backend.name,
            "stencil_name": self.builder.stencil_id.qualified_name,
            "stencil_version": self.builder.stencil_id.version,
            "module_shash": gt_utils.shash(self.builder.stencil_source),
            **self.builder.backend.extra_cache_info,
        }

    def update_cache_info(self) -> None:
        if not self.cache_info_path:
            return
        cache_info = self.generate_cache_info()
        self.cache_info_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_info_path.open("wb") as cache_info_file:
            pickle.dump(cache_info, cache_info_file)

    def is_cache_info_available_and_consistent(
        self, *, validate_hash: bool, catch_exceptions: bool = True
    ) -> bool:
        result = True
        if not self.cache_info_path and catch_exceptions:
            return False
        try:
            cache_info = self.generate_cache_info()
            cache_info_ns = types.SimpleNamespace(**cache_info)
            validate_extra = {
                k: v
                for k, v in self.builder.backend.extra_cache_info.items()
                if k in self.builder.backend.extra_cache_validation_keys
            }
            source = self.builder.module_path.read_text()
            module_shash = gt_utils.shash(source)

            if validate_hash:
                result = (
                    cache_info_ns.backend == self.builder.backend.name
                    and cache_info_ns.stencil_name == self.stencil_id.qualified_name
                    and cache_info_ns.stencil_version == self.stencil_id.version
                    and cache_info_ns.module_shash == module_shash
                )
                if validate_extra:
                    result &= all(
                        [cache_info[key] == validate_extra[key] for key in validate_extra]
                    )
        except Exception as err:
            if not catch_exceptions:
                raise err
            result = False

        return result

    @property
    def cache_info(self) -> Dict[str, Any]:
        if not self.cache_info_path:
            return {}
        if not self.cache_info_path.exists():
            return {}
        return self._unpickle_cache_info_file(self.cache_info_path)

    @staticmethod
    def _unpickle_cache_info_file(cache_info_path: pathlib.Path) -> Dict[str, Any]:
        with cache_info_path.open("rb") as cache_info_file:
            return pickle.load(cache_info_file)

    @property
    def options_id(self) -> str:
        if hasattr(self.builder.backend, "filter_options_for_id"):
            return self.builder.backend.filter_options_for_id(self.builder.options).shashed_id
        return self.builder.options.shashed_id

    def capture_externals(self) -> Dict[str, Any]:
        """Extract externals from the annotated stencil definition for fingerprinting."""
        return self._externals

    @cached_property
    def _externals(self) -> Dict[str, Any]:
        """Extract externals from the annotated stencil definition for fingerprinting."""
        return {
            name: value._gtscript_["canonical_ast"] if hasattr(value, "_gtscript_") else value
            for name, value in self.builder.definition._gtscript_["externals"].items()
        }

    def _extract_api_annotations(self) -> List[str]:
        """Extract API annotations from the annotated stencil definition for fingerprinting."""
        return [str(item) for item in self.builder.definition._gtscript_["api_annotations"]]

    @property
    def stencil_id(self) -> StencilID:
        fingerprint = {
            "__main__": self.builder.definition._gtscript_["canonical_ast"],
            "docstring": inspect.getdoc(self.builder.definition),
            "api_annotations": f"[{', '.join(self._extract_api_annotations())}]",
            **self._externals,
        }
        debug_mode = self.builder.options.backend_opts.get("debug_mode", False)
        fingerprint["debug_mode"] = debug_mode
        if not debug_mode and self.builder.backend.name != "numpy":
            fingerprint["opt_level"] = self.builder.options.backend_opts.get(
                "opt_level", gt_config.GT4PY_COMPILE_OPT_LEVEL
            )
            fingerprint["extra_opt_flags"] = self.builder.options.backend_opts.get(
                "extra_opt_flags", gt_config.GT4PY_EXTRA_COMPILE_OPT_FLAGS
            )
            fingerprint["extra_compile_args"] = self.builder.options.backend_opts.get(
                "extra_compile_args", gt_config.GT4PY_EXTRA_COMPILE_ARGS
            )
        if self.builder.backend.name == "dace:gpu":
            fingerprint["default_block_size"] = gt_config.DACE_DEFAULT_BLOCK_SIZE

        # typeignore because attrclass StencilID has generated constructor
        return StencilID(  # type: ignore
            self.builder.options.qualified_name,
            gt_utils.shashed_id(gt_utils.shashed_id(fingerprint), self.options_id),
        )

    @property
    def module_prefix(self) -> str:
        return "m_"

    @property
    def module_postfix(self) -> str:
        backend_name = gt_utils.slugify(self.builder.backend.name)
        id_version = self.stencil_id.version
        return f"__{backend_name}_{id_version}"

    @property
    def class_name(self) -> str:
        name = self.builder.options.name
        return f"{name}__{self.module_postfix}"


class NoCachingStrategy(CachingStrategy):
    """
    Apply no caching, useful for CLI.

    Instead of calculating paths and names based on a stencil fingerprint, use
    user input or by default the current working dir to output source and build files.

    This caching strategy is incapable of avoiding rebuilds and after rebuilding at runtime
    the old stencil and it's module has to be unloaded by the user before reloading for any
    changes to take effect.

    All methods related to generating or checking the cache info are noops and cache validation
    always fails.

    Parameters
    ----------
    builder:
        A stencil builder instance

    output_path:
        The path to where all source code should be written
    """

    name = "nocaching"

    def __init__(self, builder: StencilBuilder, *, output_path: pathlib.Path = pathlib.Path(".")):
        super().__init__(builder)
        self._output_path = output_path

    @property
    def root_path(self) -> pathlib.Path:
        """Get stencil code output path set during initialization."""
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)
        return self._output_path

    @property
    def backend_root_path(self) -> pathlib.Path:
        """Simply pass through the user chosen output path."""
        return self.root_path

    @property
    def cache_info_path(self) -> Optional[pathlib.Path]:
        return None

    def generate_cache_info(self) -> Dict[str, Any]:
        return {}

    def update_cache_info(self) -> None:
        pass

    def is_cache_info_available_and_consistent(self, *, validate_hash: bool) -> bool:
        return False

    @property
    def cache_info(self) -> Dict[str, Any]:
        return {}

    @property
    def stencil_id(self) -> StencilID:
        """Get a fingerprint-less stencil id."""
        # ignore type because mypy does not understand attrib classes
        return StencilID(  # type: ignore
            qualified_name=self.builder.options.qualified_name, version=""
        )


def strategy_factory(
    name: str, builder: StencilBuilder, *args: Any, **kwargs: Any
) -> CachingStrategy:
    strategies = {"jit": JITCachingStrategy, "nocaching": NoCachingStrategy}
    return strategies[name](builder, *args, **kwargs)
