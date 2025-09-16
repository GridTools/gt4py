from typing import Hashable, Any
import pathlib
import pickle

from gt4py.eve import utils as eve_utils
from gt4py._core import locking

class FileCache:
    path: pathlib.Path

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: Hashable) -> pathlib.Path:
        """Return the path where an item with `key` is stored."""
        key = eve_utils.content_hash(key)
        path = self.path.joinpath(key).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def __getitem__(self, key: Hashable) -> Any:
        if key not in self:
            raise KeyError(key)
        with locking.lock(self._get_path(key)):
            with open(self._get_path(key), "rb") as f:
                return pickle.load(f)

    def __setitem__(self, key: Hashable, value: Any):
        with locking.lock(self._get_path(key)):
            with open(self._get_path(key), "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __delitem__(self, key: Hashable) -> None:
        if key not in self:
            raise KeyError(key)
        with locking.lock(self._get_path(key)):
            self._get_path(key).unlink()

    def __contains__(self, key: Hashable) -> bool:
        return self._get_path(key).exists()