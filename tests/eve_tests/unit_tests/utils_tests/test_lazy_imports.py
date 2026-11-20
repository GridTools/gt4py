# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import inspect
import os
import pathlib
import random
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

from gt4py import eve
from gt4py.eve.utils import lazy_imports


class TestLazyImportModule:
    """Tests for `lazy_import_module()` using toy packages written to `tmp_path`.

    Every toy module records its execution in `lztrace.ORDER`, so tests can
    check what was executed and in which order.
    """

    @pytest.fixture
    def make_modules(self, tmp_path, monkeypatch):
        """Return a function writing toy modules (`{"pkg/__init__.py": "source"}`) to an import path."""
        saved_modules = dict(sys.modules)
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "lztrace.py").write_text("ORDER = []\n")

        def make(files: dict[str, str]) -> pathlib.Path:
            for rel_path, source in files.items():
                path = tmp_path / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(textwrap.dedent(source))
            importlib.invalidate_caches()
            return tmp_path

        yield make

        for name in set(sys.modules) - set(saved_modules):
            del sys.modules[name]
        sys.modules.update(saved_modules)

    TRACE = "import lztrace; lztrace.ORDER.append(__name__)\n"

    @staticmethod
    def trace() -> list[str]:
        import lztrace

        return lztrace.ORDER

    def test_no_code_executed(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": self.TRACE,
                "lzpkg/sub/__init__.py": self.TRACE,
                "lzpkg/sub/leaf.py": self.TRACE,
            }
        )
        modules_before = set(sys.modules)

        proxy = eve.utils.lazy_import_module("lzpkg.sub.leaf")

        assert set(sys.modules) == modules_before
        assert eve.utils.is_lazy_module(proxy)
        assert isinstance(proxy, types.ModuleType)
        assert proxy.__name__ == "lzpkg.sub.leaf"
        assert repr(proxy) == "<lazy module 'lzpkg.sub.leaf' (unloaded)>"
        assert set(sys.modules) == modules_before

        assert proxy.__file__.endswith("leaf.py")
        assert self.trace() == ["lzpkg", "lzpkg.sub", "lzpkg.sub.leaf"]  # parent first
        assert repr(proxy) == repr(sys.modules["lzpkg.sub.leaf"])

    def test_already_imported(self):
        assert eve.utils.lazy_import_module("sys") is sys
        assert eve.utils.lazy_import_module("importlib.util") is importlib.util
        assert not eve.utils.is_lazy_module(sys)
        assert eve.utils.resolve_lazy_module(sys) is sys

    def test_identity(self, make_modules):
        make_modules({"lzpkg/__init__.py": "X = 1\n"})
        proxy = eve.utils.lazy_import_module("lzpkg")
        proxy_hash = hash(proxy)

        real = eve.utils.resolve_lazy_module(proxy)

        assert real is sys.modules["lzpkg"]
        assert proxy is not real
        assert proxy != real
        assert hash(proxy) == proxy_hash
        assert eve.utils.is_lazy_module(proxy)
        assert not eve.utils.is_lazy_module(real)

    def test_attribute_forwarding(self, make_modules):
        make_modules({"lzpkg/__init__.py": "X = 1\n"})
        proxy = eve.utils.lazy_import_module("lzpkg")

        proxy.Y = 2
        real = sys.modules["lzpkg"]
        assert real.Y == 2
        del proxy.X
        assert not hasattr(real, "X")
        assert vars(proxy) is vars(real)
        assert proxy.__dict__ is real.__dict__
        assert "Y" in dir(proxy)

    def test_module_getattr(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": """
                def __getattr__(name):
                    if name == "dynamic":
                        return "from __getattr__"
                    raise AttributeError(name)
                """
            }
        )
        assert eve.utils.lazy_import_module("lzpkg").dynamic == "from __getattr__"

    def test_circular_reexport(self, make_modules):
        # The parent re-exports from the child, while the child imports the parent.
        make_modules(
            {
                "lzpkg/__init__.py": self.TRACE + "from .sub import Y\nZ = Y * 10\n",
                "lzpkg/sub/__init__.py": self.TRACE + "import lzpkg\nY = 2\n",
            }
        )
        child = eve.utils.lazy_import_module("lzpkg.sub")
        parent = eve.utils.lazy_import_module("lzpkg")

        assert child.Y == 2  # touch the child first
        assert parent.Z == 20
        assert self.trace() == ["lzpkg", "lzpkg.sub"]
        assert sys.modules["lzpkg"].sub is eve.utils.resolve_lazy_module(child)

    def test_child_import_side_effects(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": "REGISTRY = []\nfrom . import reg\nN = len(REGISTRY)\n",
                "lzpkg/reg.py": "from . import REGISTRY\nREGISTRY.append(__name__)\n",
            }
        )
        _ = eve.utils.lazy_import_module("lzpkg.reg").__file__

        assert sys.modules["lzpkg"].N == 1

    def test_parent_binding_shadows_child(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": "from .sub import sub\n",
                "lzpkg/sub.py": "def sub(): pass\n",
            }
        )
        _ = eve.utils.lazy_import_module("lzpkg.sub").__file__

        assert isinstance(sys.modules["lzpkg"].sub, types.FunctionType)

    def test_self_replacing_module(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": """
                import sys, types
                sys.modules[__name__] = types.SimpleNamespace(X="wrapped")
                """
            }
        )
        proxy = eve.utils.lazy_import_module("lzpkg")

        assert proxy.X == "wrapped"
        assert proxy.X == "wrapped"

    def test_runtime_package_path(self, make_modules, tmp_path):
        make_modules(
            {
                "lzpkg/__init__.py": f"""
                import os, sys
                __path__.append({str(tmp_path / "extra")!r})
                sys.modules[__name__ + ".alias"] = os
                """,
                "extra/plug.py": "WHERE = 'extra'\n",
            }
        )

        # Not found statically: the parent is imported to look into its runtime `__path__`
        plug = eve.utils.lazy_import_module("lzpkg.plug")
        assert "lzpkg" in sys.modules
        assert plug.WHERE == "extra"

        assert eve.utils.lazy_import_module("lzpkg.alias") is os
        assert eve.utils.lazy_import_module("lzpkg.missing", on_missing="none") is None
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module("lzpkg.missing")

    def test_alias_registered_by_parent(self, make_modules):
        # The alias is looked up before the parent is imported, and it has no `__spec__`.
        make_modules(
            {
                "lzpkg/__init__.py": """
                import sys, types
                sys.modules[__name__ + ".alias"] = ALIAS = types.ModuleType(__name__ + ".alias")
                """
            }
        )
        alias = eve.utils.lazy_import_module("lzpkg.alias")

        assert alias is sys.modules["lzpkg"].ALIAS

    @pytest.mark.parametrize(
        "fullname",
        [
            "lz_not_installed",
            "lz_not_installed.child",
            "lzpkg.missing",
            "lzpkg.missing.child",
            "lzpkg.module.child",  # parent is not a package
            "typing.numpy",  # parent is an imported module, not a package
            "lzblocked",
            "lzblocked.child",
        ],
    )
    def test_missing(self, make_modules, monkeypatch, fullname):
        make_modules({"lzpkg/__init__.py": "", "lzpkg/module.py": ""})
        monkeypatch.setitem(sys.modules, "lzblocked", None)  # standard way to block an import

        assert eve.utils.lazy_import_module(fullname, on_missing="none") is None
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module(fullname)
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module(fullname, on_missing="raise")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(fullname)  # check plain import agrees

    def test_broken_parent_in_fallback(self, make_modules):
        # The child is not found statically, so the broken parent is imported at call time.
        make_modules({"lzpkg/__init__.py": "raise ImportError('broken parent')\n"})

        with pytest.raises(ImportError, match="broken parent"):
            eve.utils.lazy_import_module("lzpkg.missing", on_missing="none")

    def test_invalid_on_missing(self):
        with pytest.raises(ValueError, match="on_missing"):
            eve.utils.lazy_import_module("sys", on_missing="ignore")

    def test_broken_library(self, make_modules):
        make_modules({"lzbroken/__init__.py": "A = 1\nraise ImportError('broken library')\n"})
        proxy = eve.utils.lazy_import_module("lzbroken")

        for _ in range(2):  # errors are not cached: every access retries the import
            with pytest.raises(ImportError, match="broken library"):
                _ = proxy.A
            assert "lzbroken" not in sys.modules

        assert not hasattr(proxy, "__wrapped__")
        with pytest.raises(TypeError):
            vars(proxy)
        inspect.stack()
        assert repr(proxy) == "<lazy module 'lzbroken' (unloaded)>"

    def test_half_initialized_module_not_cached(self, make_modules):
        # The module touches its own proxy while it is being imported, then fails.
        make_modules(
            {
                "lzholder.py": "PROXY = None\n",
                "lzhalf/__init__.py": """
                import lzholder
                lzholder.SEEN = getattr(lzholder.PROXY, "GOOD", "not yet")
                GOOD = 1
                raise ImportError("fails after being touched")
                """,
            }
        )
        import lzholder

        lzholder.PROXY = proxy = eve.utils.lazy_import_module("lzhalf")

        for _ in range(2):
            with pytest.raises(ImportError, match="fails after being touched"):
                _ = proxy.GOOD
            assert lzholder.SEEN == "not yet"

    @pytest.mark.parametrize("circular", [False, True])
    def test_threads(self, make_modules, circular):
        make_modules(
            {
                "lzpkg/__init__.py": (
                    "import time; time.sleep(0.002)\n"
                    + ("from .sub import Y\n" if circular else "")
                    + "X = 1\n"
                ),
                "lzpkg/sub/__init__.py": (
                    "import time; time.sleep(0.002)\n"
                    + ("import lzpkg\n" if circular else "")
                    + "Y = 2\n"
                ),
            }
        )
        rng = random.Random(42)
        for _ in range(20):
            for name in ("lzpkg.sub", "lzpkg"):
                sys.modules.pop(name, None)
            proxies = [eve.utils.lazy_import_module(name) for name in ("lzpkg", "lzpkg.sub")]
            errors: list[BaseException] = []

            def touch(delay, proxies=proxies, errors=errors):
                time.sleep(delay)
                for proxy in random.sample(proxies, 2):
                    try:
                        getattr(proxy, "X" if proxy.__name__ == "lzpkg" else "Y")
                    except Exception as error:
                        errors.append(error)

            threads = [
                threading.Thread(target=touch, args=(rng.random() / 500,), daemon=True)
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert not any(thread.is_alive() for thread in threads), "deadlock"

            if not circular:
                # Concurrent imports of circular packages may fail as with plain imports
                assert not errors
            for proxy in proxies:
                cached = object.__getattribute__(proxy, "__dict__")[lazy_imports._LAZY_MODULE_SLOT]
                assert cached is None or cached is sys.modules[proxy.__name__]

    @pytest.mark.parametrize("parent_init", [False, True])
    def test_namespace_packages(self, make_modules, parent_init):
        # A namespace package (no `__init__.py`) below a parent which is not imported yet
        files = {"lzns/part/leaf.py": "X = 1\n"}
        if parent_init:
            files["lzns/__init__.py"] = ""
        make_modules(files)

        part = eve.utils.lazy_import_module("lzns.part")
        leaf = eve.utils.lazy_import_module("lzns.part.leaf")
        assert eve.utils.lazy_import_module("lzns.missing", on_missing="none") is None

        assert leaf.X == 1
        assert eve.utils.resolve_lazy_module(part) is sys.modules["lzns.part"]

    def test_alias_registered_by_non_package_parent(self, make_modules):
        make_modules(
            {
                "lzmod.py": """
                import sys, types
                sys.modules[__name__ + ".child"] = CHILD = types.ModuleType(__name__ + ".child")
                """
            }
        )
        child = eve.utils.lazy_import_module("lzmod.child")

        assert child is sys.modules["lzmod"].CHILD
        assert child is importlib.import_module("lzmod.child")
        assert eve.utils.lazy_import_module("lzmod.missing", on_missing="none") is None

    def test_non_import_errors_propagate(self, make_modules):
        make_modules({"lzbroken/__init__.py": "raise RuntimeError('no device')\n"})
        proxy = eve.utils.lazy_import_module("lzbroken")

        with pytest.raises(RuntimeError, match="no device"):
            _ = proxy.X
        with pytest.raises(RuntimeError, match="no device"):  # not turned into `AttributeError`
            hasattr(proxy, "__wrapped__")
        assert "lzbroken" not in sys.modules

    def test_success_after_failure(self, make_modules, tmp_path):
        flag = tmp_path / "fail_import"
        flag.touch()
        make_modules(
            {
                "lzflaky/__init__.py": f"""
                import pathlib
                if pathlib.Path({str(flag)!r}).exists():
                    raise ImportError("transient failure")
                X = 1
                """
            }
        )
        proxy = eve.utils.lazy_import_module("lzflaky")
        for _ in range(2):
            with pytest.raises(ImportError, match="transient failure"):
                _ = proxy.X

        flag.unlink()

        assert proxy.X == 1
        assert eve.utils.resolve_lazy_module(proxy) is sys.modules["lzflaky"]

    def test_reload(self, make_modules):
        make_modules({"lzpkg/__init__.py": "X = 1\n"})
        proxy = eve.utils.lazy_import_module("lzpkg")
        _ = proxy.X

        with pytest.raises(ImportError):  # proxies are not in `sys.modules`
            importlib.reload(proxy)
        assert importlib.reload(eve.utils.resolve_lazy_module(proxy)) is sys.modules["lzpkg"]

    def test_concurrent_calls(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": "import time; time.sleep(0.01)\nX = 1\n",
                "lzpkg/sub/__init__.py": "Y = 2\n",
            }
        )
        results: list[object] = []
        errors: list[BaseException] = []

        def call_and_touch(name):
            try:
                proxy = eve.utils.lazy_import_module(name)
                results.append(eve.utils.resolve_lazy_module(proxy))
            except Exception as error:
                errors.append(error)

        names = ["lzpkg", "lzpkg.sub"] * 4
        threads = [threading.Thread(target=call_and_touch, args=(n,), daemon=True) for n in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not any(thread.is_alive() for thread in threads), "deadlock"
        assert not errors
        assert {id(module) for module in results} == {
            id(sys.modules["lzpkg"]),
            id(sys.modules["lzpkg.sub"]),
        }

    def test_introspection_does_not_import(self, make_modules, tmp_path):
        # Run in a subprocess: `inspect.getmodule()` (used by `inspect.stack()`) scans `sys.modules`.
        make_modules({"lzpkg/__init__.py": "raise ImportError('must not be imported')\n"})
        script = textwrap.dedent(
            """
            import inspect, sys
            from gt4py import eve
            proxy = eve.utils.lazy_import_module("lzpkg")
            inspect.stack()
            print("lzpkg" in sys.modules, eve.utils.is_lazy_module(proxy))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])},
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.split() == ["False", "True"]
