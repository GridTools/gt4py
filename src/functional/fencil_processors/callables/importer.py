import importlib
import importlib.util
import inspect
import pathlib
from typing import Callable, Dict


def import_callables(module_file: pathlib.Path) -> Dict[str, Callable]:
    module_name = module_file.name.split(".")[0]

    error_msg = f"Could not load module named {module_name} from {module_file}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if not spec or not spec.loader:
        raise ModuleNotFoundError(error_msg)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        raise ModuleNotFoundError(error_msg)

    members = inspect.getmembers(module)
    functions = filter(lambda obj: inspect.isfunction(obj[1]) or inspect.isbuiltin(obj[1]), members)
    return dict(functions)
