import importlib
import inspect
import pathlib
import sys
from typing import Callable, Dict


def import_callables(module_file: pathlib.Path) -> Dict[str, Callable]:
    folder = module_file.parent
    sys.path.append(str(folder))
    importlib.invalidate_caches()
    module = importlib.import_module(module_file.stem)
    members = inspect.getmembers(module)
    functions = filter(lambda obj: inspect.isfunction(obj[1]) or inspect.isbuiltin(obj[1]), members)
    return dict(functions)
