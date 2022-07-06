from typing import Callable

from functional.fencil_processors import defs as defs
from functional.fencil_processors.callables.cache import Strategy as CacheStrategy, get_cache_folder
from functional.fencil_processors.callables.modules import load_module

from . import bindings, build


def create_callable(
    source_module: defs.SourceCodeModule, cache_strategy=CacheStrategy.SESSION
) -> Callable:
    cache_folder = get_cache_folder(
        source_module.entry_point.name, source_module.source_code, cache_strategy
    )
    module_file = build.CMakeProject.get_binary(cache_folder, source_module.entry_point.name)
    try:
        return load_module(module_file)[source_module.entry_point.name]
    except ModuleNotFoundError:
        pass

    src_header_file = source_module.entry_point.name + ".cpp.inc"
    bindings_file = source_module.entry_point.name + "_bindings.cpp"
    bindings_module = bindings.create_bindings(source_module.entry_point, src_header_file)

    deps = [*source_module.library_deps, *bindings_module.library_deps]
    sources = {
        src_header_file: source_module.source_code,
        bindings_file: bindings_module.source_code,
    }
    project = build.CMakeProject(
        name=source_module.entry_point.name, dependencies=deps, sources=sources
    )

    project.write(cache_folder)
    project.configure()
    project.build()

    return load_module(module_file)[source_module.entry_point.name]
