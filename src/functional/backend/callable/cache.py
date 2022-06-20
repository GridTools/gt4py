import enum
import hashlib
import pathlib
import tempfile


class Strategy(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)
_persistent_cache_dir_path = (
    pathlib.Path(tempfile.tempdir) / "gt4py_cache" if tempfile.tempdir else _session_cache_dir_path
)


def _cache_folder_name(module_name: str, module_src: str) -> str:
    hashed = hashlib.sha256(module_src.encode(encoding="utf-8"))
    hashed_str = hashed.hexdigest()
    return module_name + "_" + hashed_str


def get_cache_folder(module_name: str, module_src: str, strategy: Strategy) -> pathlib.Path:
    folder_name = _cache_folder_name(module_name, module_src)

    base_path = {
        Strategy.SESSION: _session_cache_dir_path,
        Strategy.PERSISTENT: _persistent_cache_dir_path,
    }[strategy]
    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    return complete_path
