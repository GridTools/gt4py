# Task specification & implementation plan — modernize GT4Py to Python ≥ 3.12

Status: **draft / proposal**. Working document for a multi-PR cleanup campaign.
Delete it (or fold the durable parts into `CODING_GUIDELINES.md` and an ADR)
once the campaign lands.

Baseline measured on `main` @ `ee1bb4f6a`, `ruff 0.16.1`, `Python 3.12.13`;
361 source modules / ~98 kLOC under `src/`, 353 modules under `tests/`.

______________________________________________________________________

## 1. Task specification

### 1.1 Goal

The codebase was written for Python 3.8 and still carries that dialect in
places, even though `requires-python` is now `>=3.12, <3.15`. Bring source code,
tests, dev scripts, and documentation onto a single modern Python ≥ 3.12
dialect, and upgrade the automated QA so the codebase *stays* there.

The work has three parts, in dependency order:

1. **Make `eve` handle modern typing constructs at runtime.** `eve` resolves
   annotations at run time; several of its dispatch sites still recognise only
   the Python 3.8 spelling of a type. Until that is fixed, mechanically
   rewriting annotations corrupts them silently (§2.2).
2. **Remove the `eve.extended_typing` re-export layer.** It shadows `typing` /
   `typing_extensions` for part of the tree, and in doing so disables mypy and
   ruff on everything imported through it (§2.3).
3. **Rewrite the dialect and raise the lint floor** — the ~1900-site annotation
   sweep plus the additional rulesets in §3.

### 1.2 Non-goals (hard constraints)

1. **No behaviour change.** Every backend must produce byte-identical generated
   code and identical results. No error-message text changes except where a rule
   mechanically requires it (and then the matching test changes with it).
2. **No architectural change.** No new/removed subpackages, no changes to the
   `tach.toml` DAG, no backend-interface changes, no public-API renames beyond
   the internal `eve.extended_typing` → `eve.extra_typing` move of §5, PR 2.
   Where a modernization *would* be architectural it is called out and deferred
   to its own ADR (§7).
3. **No new runtime dependency.** Removals are in scope; additions are not.
4. **No pedantry.** Rules that generate large-scale churn without a safety,
   performance, or comprehension payoff are rejected — with the rationale
   recorded (§3.4), so the question is not reopened every six months.
5. **Google Python Style Guide stays authoritative.** GT4Py follows it with the
   deviations listed in `CODING_GUIDELINES.md` §"Code Style". No proposed rule
   may contradict it; conflicts are resolved in favour of the guide (§3.4, `EM`).

### 1.3 In scope

| Area           | Paths                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------ |
| Library code   | `src/gt4py/**`                                                                             |
| Tests          | `tests/**`, `typing_tests/**`                                                              |
| Dev tooling    | `noxfile.py`, `scripts/**`, `.pre-commit-config.yaml`, `pyproject.toml`                    |
| Notebooks      | `examples/**`, `docs/user/next/workshop/**`                                                |
| Developer docs | `AGENTS.md`, `CODING_GUIDELINES.md`, `CONTRIBUTING.md`, `README.md`, `docs/development/**` |
| User docs      | `docs/user/**`                                                                             |

### 1.4 Definition of done

- Every annotation-dispatch funnel in `eve` treats old and new spellings of the
  same type identically, enforced by the conformance matrix of §5, PR 1.
- `eve.extended_typing` is gone: `eve.extra_typing` holds only definitions that
  `typing` / `typing_extensions` do not provide, and the rest of the tree imports
  from them directly.
- `uv run pre-commit run -a` is clean with the ruleset of §3 active.
- `ruff` floor bumped to the current release; `uv.lock` regenerated.
- `uv run nox -s test_eve`, `test_cartesian`, `test_next`, `test_storage`,
  `test_package`, `test_typing_exports` green on 3.12, 3.13 and 3.14.
- No remaining stdlib back-compat shim for a feature available in 3.12 (§2.4).
- No remaining `sys.version_info` branch whose lower arm is unreachable at 3.12
  (§2.5).
- `CODING_GUIDELINES.md` documents the dialect rules the linter now enforces, and
  its stale cross-references are fixed (§6).
- `CHANGELOG.md` has an entry per merged PR.

______________________________________________________________________

## 2. Measured baseline

The tree is **currently clean** under its own configuration:

```
ruff check src/ scripts/ noxfile.py   → All checks passed!
ruff format --check (730 files)       → all formatted
```

So every number below is newly surfaced debt, not a pre-existing failure.

### 2.1 Legacy-dialect inventory (`src/`)

| Pattern                                    |    Count | Rule                                  |
| ------------------------------------------ | -------: | ------------------------------------- |
| `Optional[X]`                              |      642 | `UP045`                               |
| `typing.List/Dict/Tuple/Type/Set[...]`     |      502 | `UP006`                               |
| `Union[X, Y]`                              |      264 | `UP007`                               |
| deprecated `typing` imports                |      217 | `UP035`                               |
| `X: TypeAlias = ...`                       |      134 | `UP040` — **blocked**, §4.4           |
| PEP 695 generic fn / class candidates      |  54 / 48 | `UP047` / `UP046` — **blocked**, §4.4 |
| quoted annotations                         |       21 | `UP037`                               |
| misc (`UP004/018/029/032/033/034/042/044`) |       21 | —                                     |
| **UP total**                               | **1905** | 1531 safe-fixable                     |

Tests, examples and `typing_tests` add a further 411 `UP` findings.

### 2.2 `eve` resolves annotations at run time

`eve.datamodels` builds validators and converters from live annotation objects,
and dispatches on annotation **identity** at six independent sites:

| Funnel                       | Location                                               |
| ---------------------------- | ------------------------------------------------------ |
| type validator construction  | `eve/type_validation.py` `__call__`                    |
| converter construction       | `eve/datamodels/core.py` `_make_type_converter`        |
| immutability analysis        | `eve/datamodels/core.py` `_is_strictly_immutable_type` |
| represented-types extraction | `eve/extended_typing.py` `get_represented_types`       |
| forward-ref resolution       | `eve/extended_typing.py` `eval_forward_ref`            |
| `next` type translation      | `next/type_system/type_translation.py`                 |

A site that has not been taught a new spelling **does not raise** — it falls
through to its no-match branch. `extended_typing.py` documents this for PEP 695
aliases in as many words: *"a funnel left out does not raise, it falls through to
its no-match branch […] silently making every `isinstance()` against the result
`False`."*

The same hazard already exists for PEP 604 unions, and is confirmed (§4.1):
`_make_type_converter` tests `origin_type is xtyping.Union`, but
`get_origin(int | None)` is `types.UnionType`, so `int | None` builds a
nonsensical converter where `Optional[int]` works. `type_validation.py:229` and
`_is_strictly_immutable_type` (line 1114) already normalize both spellings; this
one funnel does not.

This is why the annotation sweep cannot come first.

### 2.3 The `eve.extended_typing` re-export layer

`extended_typing.py` (1012 lines) star-imports `typing` and `typing_extensions`,
re-exports 131 names, and adds a module `__getattr__` that forwards anything else
to `typing_extensions` then `typing`. Its own contribution is ~46 definitions.

**It blinds mypy.** mypy cannot model a module `__getattr__`, so it types every
unknown attribute as `Any`:

```
$ mypy -c 'from gt4py.eve import extended_typing as xtyping; reveal_type(xtyping.TotallyMadeUpSymbol)'
note: Revealed type is "Any"
Success: no issues found in 1 source file
```

Every one of the 446 `xtyping.*` references in the tree is a name mypy will not
check — in the module that defines the project's type infrastructure.

**It blinds ruff, and the workaround is load-bearing.** `pyproject.toml` sets
`typing-modules = ['gt4py.eve.extended_typing']`. Verified: remove that line and
`UP045` stops firing on `from gt4py.eve.extended_typing import Optional`
entirely. The §2.1 sweep currently depends on a non-standard escape hatch to see
part of the codebase.

**The codebase has already voted.** 243 of 361 source modules import plain
`typing` directly; only 49 touch `extended_typing`; **29 import both**. The
re-export layer is not the house style — it produces modules that import `Any`
from one place and `Optional` from another.

**The migration surface is small.** Across `src/` and `tests/`, 74 distinct
symbols are pulled from `extended_typing` — 446 uses in 56 files (48 `src`, 8
`tests`):

| Destination                                  | Symbols | Uses |
| -------------------------------------------- | ------: | ---: |
| stays in `eve.extra_typing` (genuine extras) |      30 |  176 |
| → `from typing import …`                     |      42 |  262 |
| → `from typing_extensions import …`          |       2 |    8 |

What remains is a real module, not a husk: the array/DLPack protocols
(`ArrayInterface`, `CUDAArrayInterface`, `DLPackBuffer`, `SupportsArray` and
their guards), the nested-collection aliases (`NestedTuple`,
`MaybeNestedInTuple`, …), the descriptor and dataclass protocols
(`DataDescriptor`, `DataclassABC`, `HasCustomHash`), and the annotation
utilities (`eval_type_alias`, `eval_forward_ref`, `get_partial_type_hints`,
`infer_type`, `get_represented_types`).

Removing the layer also retires the `_DEPRECATED_TYPING_ALIASES`
pop-from-`globals()` hack, the `__dir__` override, the pure re-export
`is_protocol = _typing_extensions.is_protocol`, the ruff `typing-modules`
setting, and the mypy `warn_unused_ignores = false` override for the module.

`extended_typing` is **not** exported from `gt4py.eve.__all__` — `__init__.py`
mentions it only in the module-dependency docstring — so the rename is an
internal change, though downstream projects can import internal `eve` modules
(§5, PR 2).

### 2.4 Stdlib shims still in place

| Shim                                                          | Sites                                                                  | Modern replacement                                                                                                                                    |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `from cached_property import cached_property`                 | `cartesian/caching.py`, `cartesian/lazy_stencil.py`                    | `functools.cached_property` — both sites are plain classes with no `__slots__`, so it is a drop-in; drop the `cached-property` **runtime dependency** |
| `class StrEnum(str, Enum)` in `eve/type_definitions.py`       | 1 def                                                                  | `enum.StrEnum` (3.11)                                                                                                                                 |
| `class TransientMemoryMode(str, enum.Enum)`                   | `next/…/dace/transformations/auto_optimize.py:115`                     | `enum.StrEnum`                                                                                                                                        |
| `from typing_extensions import Self`                          | `next/otf/workflow.py:18`                                              | `typing.Self` (3.11)                                                                                                                                  |
| `from typing_extensions import Protocol`                      | `cartesian/type_hints.py:11`, `cartesian/gtc/gtcpp/oir_to_gtcpp.py:17` | `typing.Protocol` (but see §4.3)                                                                                                                      |
| `# TODO: add xtyping.Buffer once we update typing_extensions` | `storage/allocators.py:46`                                             | `collections.abc.Buffer` exists on 3.12 and **is** `typing_extensions.Buffer` — the TODO is closable now                                              |

`typing-extensions` itself **stays** a runtime dependency: `extra_typing` will
still need `TypeAliasType`, `TypeIs` and `is_protocol`.

Separately, `frozendict` is declared in `[project].dependencies` but its only
import in the tree is `tests/eve_tests/unit_tests/test_type_validation.py` — it
belongs in the `test` dependency group. Packaging fix, no behaviour change.

### 2.5 Version-gated code

Only four `sys.version_info` sites exist. Three must stay; one is dead:

- `next/config.py:77` — `>= (3, 13)` for `os.process_cpu_count()`. **Keep** while
  3.12 is supported.
- `cartesian/caching.py:200`, `eve/codegen.py:103` — read the running version,
  not gates. **Keep.**
- `tests/eve_tests/unit_tests/test_extended_typing.py:297` —
  `sys.version_info >= (3, 12)` is now always true. **Remove the branch.**

### 2.6 Tooling

|                                           | Now                                  | Target                                       |
| ----------------------------------------- | ------------------------------------ | -------------------------------------------- |
| `ruff` floor (`[dependency-groups].lint`) | `>=0.8.0`                            | `>=0.16.4`                                   |
| `ruff` resolved in `uv.lock`              | `0.16.1`                             | `0.16.4`                                     |
| `target-version`                          | `py312`                              | unchanged — already correct                  |
| mypy `python_version`                     | `3.12`                               | unchanged — deliberately pinned to the floor |
| ruff `typing-modules`                     | `['gt4py.eve.extended_typing']`      | **removed** (§2.3)                           |
| ruff `lint.exclude`                       | `docs/**`, `examples/**`, `tests/**` | narrowed, see §3.5                           |
| `lint.ignore`                             | `E501`, `B905`, `TD003`              | `B905` removed, see §3.2                     |

CI needs no version edits: every workflow reads its matrix from
`.python-versions` via `.github/actions/get-python-versions`.

______________________________________________________________________

## 3. Ruff: version bump and rule proposal

Bump the floor to `ruff>=0.16.4` (latest release) in `[dependency-groups].lint`
and re-lock. The pre-commit hooks already run `uv`-managed ruff, so there is no
second version to keep in sync — that is what `scripts/python/update.py`
(`./scripts/run update pre-commit`) exists for, and it needs no change.

### 3.1 Rules to enable — modernization core

| Ruleset                                          | src hits | Why                                                                                      |
| ------------------------------------------------ | -------: | ---------------------------------------------------------------------------------------- |
| `UP` (pyupgrade) minus `UP040`, `UP046`, `UP047` |     1669 | The point of the task. `UP006`/`UP007`/`UP045`/`UP035` carry ~90 % of the dialect drift. |

The three PEP 695 rules are excluded, with a config comment pointing at the block
in `eve/extra_typing.py` that explains why (§4.4). This is a rule-level `ignore`,
not a blanket `UP` exclusion, so everything else keeps working.

### 3.2 Rules to enable — safety

| Ruleset                   | src hits | Why                                                                                                                                                                                                                                                                 |
| ------------------------- | -------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **un-ignore `B905`**      |      110 | `zip()` without `strict=` silently truncates to the shortest input. In an IR-transformation codebase that is a *wrong result*, not a style nit. The existing `# TODO(egparedes): remove when possible` says as much. Needs per-site judgement, so it is its own PR. |
| `PGH`                     |       23 | `PGH003`/`PGH004` reject blanket `# type: ignore` / `# noqa`. `CODING_GUIDELINES.md` §"Ignoring QA errors" already *requires* specific codes; this makes the existing rule enforceable.                                                                             |
| `PYI`                     |       53 | Correctness rules for `Protocol`/`overload`/stub-shaped definitions — `eve` and `_core.definitions` are full of them. `PYI032` (`__eq__` annotated `Any`, 15×), `PYI061`, `PYI036` catch real annotation mistakes.                                                  |
| `SLOT`                    |        1 | `SLOT001`: `eve.type_definitions.FrozenList` subclasses `tuple`; a missing `__slots__` there silently adds a `__dict__`.                                                                                                                                            |
| `W`                       |        5 | `E` is selected but `W` never was. Trailing whitespace only; free.                                                                                                                                                                                                  |
| `DTZ`                     |        1 | One naive `datetime.now()`. Free.                                                                                                                                                                                                                                   |
| `FA` (widen from `FA100`) |        0 | No new findings; keeps the family coherent now that annotations are being rewritten.                                                                                                                                                                                |

### 3.3 Rules to enable — clarity & performance

| Ruleset                                      | src hits | Why                                                                                                                                                                                                                                                      |
| -------------------------------------------- | -------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C4`                                         |      106 | `C408` (`dict()`→`{}`, 43×), `C416`, `C419`, `C420`. Faster *and* shorter. Already on the "to be considered" list in `pyproject.toml`.                                                                                                                   |
| `PERF`                                       |       26 | `PERF401`/`PERF403` (manual comprehension building, 23×) in transformation hot paths. Small, genuinely about speed.                                                                                                                                      |
| `PIE`                                        |       66 | `PIE790` (61×) removes `...` placeholders that follow a docstring. Verified **not** to conflict with the `pass` vs `...` convention in `CODING_GUIDELINES.md`, which is about *empty* bodies — the rule only fires when a docstring is already the body. |
| `FURB`                                       |       21 | Stable refurb rules only (`explicit-preview-rules = true` gates the rest): `FURB171`, `FURB110`, `FURB192`.                                                                                                                                              |
| `FLY`                                        |        3 | `"".join()` of a static list → f-string. Free.                                                                                                                                                                                                           |
| `SIM` minus `SIM102/103/105/108/114/115/117` |     ~106 | Keeps `SIM118` (`x in d.keys()`, 55×), `SIM910`, `SIM401`, `SIM101`, `SIM201`. Drops the ones that trade an `if` for a nested ternary — those hurt readability in IR-dispatch code and are why `SIM` looks noisy at first glance (193 → ~106).           |

**Total newly surfaced in `src/`: ≈ 2170, of which ≈ 1910 are auto-fixable.** (A
raw run reports 2404, but 231 of those are `I001` produced by probing with a
config that omits the repo's `isort` settings; `I001` is and stays at zero.)

Proposed configuration:

```toml
[tool.ruff.lint]
select = [
  'A', 'B', 'C4', 'CPY', 'DTZ', 'E', 'ERA', 'F', 'FA', 'FLY', 'FURB', 'I',
  'ISC', 'NPY', 'PERF', 'PGH', 'PIE', 'PYI', 'Q', 'RUF', 'SIM', 'SLOT',
  'T10', 'UP', 'W', 'YTT',
]
ignore = [
  'E501',    # [line-too-long] handled by the formatter
  'TD003',   # [missing-todo-link]
  # PEP 695 syntax is not supported by the 'eve' datamodels: see the
  # '-- PEP 695 type aliases --' block in 'gt4py/eve/extra_typing.py'.
  'UP040', 'UP046', 'UP047',
  # 'SIM' rules that replace a statement with a denser expression; rejected
  # for readability in IR dispatch code.
  'SIM102', 'SIM103', 'SIM105', 'SIM108', 'SIM114', 'SIM115', 'SIM117',
]
# 'typing-modules' is deliberately absent: after the 'extra_typing' refactor no
# module re-exports 'typing' symbols, so ruff analyses the imports natively.
```

Also add, so the docstring convention is declared even before `D` is selected:

```toml
[tool.ruff.lint.pydocstyle]
convention = 'google'
```

### 3.4 Rules explicitly rejected (record the reasoning)

| Ruleset                     |              src hits | Rejected because                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------- | --------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TC` (flake8-type-checking) |                   224 | **Actively dangerous here.** It moves imports into `if TYPE_CHECKING:`. `eve` resolves annotations *at runtime* (§2.2); every import it hides becomes a `NameError` at class creation or first instantiation. Would need `runtime-evaluated-base-classes` / `runtime-evaluated-decorators` to enumerate every datamodel base and decorator, and would still be fragile. |
| `EM`                        |                   851 | `EM101`/`EM102` forbid string literals inside `raise`. This **directly contradicts** every example in `CODING_GUIDELINES.md` §"Error messages". Guide wins.                                                                                                                                                                                                             |
| `D` (pydocstyle)            |                  4764 | Too large for this campaign and a different kind of work (writing prose, not changing dialect). Follow-up, scoped per subpackage, using the `convention = 'google'` setting added above.                                                                                                                                                                                |
| `N` (pep8-naming)           |                   707 | Would fight the DSL and IR vocabulary (`IDim`, `__externals__`, `__gtscript__`, IR node class names).                                                                                                                                                                                                                                                                   |
| `S` (bandit)                |                  1407 | A compiler legitimately runs `subprocess`, writes temp files, and `exec`s generated code. Noise.                                                                                                                                                                                                                                                                        |
| `RET`                       |                   278 | `RET504`/`RET505` are opinionated and often make dispatch chains *harder* to read. Pure churn.                                                                                                                                                                                                                                                                          |
| `TRY`, `ARG`, `FBT`, `PL`   | 815 / 625 / 401 / 584 | Pedantic; no safety payoff at this size.                                                                                                                                                                                                                                                                                                                                |
| `BLE`                       |                    28 | `extra_typing.eval_type_alias` catches broadly *on purpose* and documents why.                                                                                                                                                                                                                                                                                          |
| `TID252` (relative imports) |                    27 | `eve` uses `from .x import` deliberately and consistently. If the team wants absolute imports everywhere that is a separate style decision, not a modernization.                                                                                                                                                                                                        |
| `RSE102`                    |                   169 | `raise X()` → `raise X`. Zero-risk and auto-fixable, but 169 diff hunks for no reader benefit. Deferred, not forbidden.                                                                                                                                                                                                                                                 |
| `PTH`                       |                    42 | `os.path` → `pathlib` in build/cache paths. Semantics differ subtly (`str` vs `Path` at API boundaries, `os.fspath` coercion), so it is **not** a no-behaviour-change edit. Deferred to its own reviewed PR (§7).                                                                                                                                                       |

### 3.5 Narrowing the `tests/` exclusion

`tool.ruff.lint.exclude` currently drops `docs/**`, `examples/**` and `tests/**`
entirely, with a `TODO(egparedes)` to remove it. Measured debt there under the
*current* ruleset is 2241 findings — but it is dominated by pytest and DSL idioms
that are not defects:

| Rule                          |      Hits | Actually is                                                        |
| ----------------------------- | --------: | ------------------------------------------------------------------ |
| `F811` redefined-while-unused |       413 | pytest fixture shadowing                                           |
| `F405` / `F401`               | 285 / 273 | `from gt4py.next import *` in DSL test modules                     |
| `F841` unused-variable        |       213 | assignments inside `@field_operator` bodies (traced, not executed) |
| `F821` undefined-name         |        86 | `__externals__` / `__gtscript__` DSL globals                       |
| `NPY002` legacy random        |       159 | genuine, but a separate test-hygiene task                          |

So: **do not** simply delete the exclusion. Replace it with per-file-ignores that
encode those idioms, e.g.

```toml
[tool.ruff.lint.per-file-ignores]
'tests/**' = ['CPY001', 'F401', 'F403', 'F405', 'F811', 'F821', 'F841', 'NPY002']
'tests/**/ffront_tests/**' = ['B015', 'B018']  # DSL expressions evaluated for their trace
'examples/**' = ['CPY001', 'F401', 'F403', 'F405']
```

and tighten from there in follow-ups. The immediate win is that `UP`, `C4`,
`PIE`, `SIM`, `PGH` and `RUF` start covering 353 test modules that are currently
unlinted — including the 411 `UP` findings, which otherwise re-introduce the old
dialect through every new test.

______________________________________________________________________

## 4. Risk register

Four of the seven entries below share one failure mode: **they do not raise.**
That is what makes this campaign different from an ordinary lint sweep, and why
the conformance matrix of §5, PR 1 comes before any rewriting.

### 4.1 🔴 PEP 604 unions break `eve`'s automatic Optional converter — confirmed

`eve/datamodels/core.py:1036` dispatches on `origin_type is xtyping.Union`. On
3.12/3.13 `get_origin(int | None)` returns `types.UnionType`, a *different
object*, so the branch is skipped and the annotation falls through to
`is_actual_type(origin_type)` — `types.UnionType` is itself a class, so a
nonsensical converter is built instead of an error being raised.

Reproduced against the current tree:

```
Optional[int]    converter=<lambda>  c(None)=None  c('3')=3
int | None       FAILED TypeError: Error during coercion of given value 'None' for field 'a'.
```

**`UP007`/`UP045` cannot be auto-fixed until this is fixed.** Audit sweep for the
same shape: every `is Union`, `is xtyping.Union`, `== Union`, and
`get_origin(...) in (...)` comparison in `src/gt4py/eve/`,
`src/gt4py/next/type_system/`, and `src/gt4py/next/ffront/fbuiltins.py`.

### 4.2 🔴 `eval_forward_ref` depends on the star-import namespace — confirmed

`eval_forward_ref` with no `globalns` resolves against **the module's own
namespace** (`globalns = {**globals(), **_DEPRECATED_ALIAS_REPLACEMENTS}`), which
today contains every typing symbol only because of the star imports:

```
eval_forward_ref("Sequence[int]")                 -> collections.abc.Sequence[int]
eval_forward_ref("Sequence[int]", globalns={})    -> NameError: name 'Sequence' is not defined
```

`datamodels/core.py:1157` calls it with no `globalns`, inside
`_is_strictly_immutable_type`, wrapped in `except Exception: return False`. So
naively removing the star imports would not crash — a `frozen="strict"` datamodel
with a string annotation would quietly start reporting *not immutable*.

The fix is not to keep the star imports; it is to stop conflating "a namespace
for resolving forward references" with "this module's globals" (§5, PR 2).

### 4.3 🟡 `typing` and `typing_extensions` are not interchangeable

The current module's contract — *"definitions in `typing_extensions` take
priority over those in `typing`"* — resolves this silently today. Once call
sites choose explicitly, the choice has to be made per symbol. Measured on
3.12.13:

|                                               | Symbols                                                                                                                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Safe `from typing import` (identical objects) | 33: `Any`, `Optional`, `Union`, `Callable`, `Sequence`, `Mapping`, `Literal`, `Final`, `ClassVar`, `Self`, `Never`, `Unpack`, `TypeAlias`, `TypeGuard`, `override`, `cast`, `overload`, `get_args`, `get_origin`, … |
| **Divergent — decide per symbol**             | 8: `Protocol`, `runtime_checkable`, `TypeVar`, `TypeVarTuple`, `ParamSpec`, `NamedTuple`, `TypedDict`, `get_type_hints`                                                                                             |
| Must come from `typing_extensions` at 3.12    | 2: `TypeIs`, `is_protocol`                                                                                                                                                                                          |

Two of the divergent eight matter concretely:

- `get_type_hints` — `eve` runs every annotation through it; the
  `typing_extensions` version has different `Annotated` / PEP 695 / PEP 649
  behaviour. **Keep `typing_extensions`.**
- `Protocol` / `runtime_checkable` — `next/common.py` declares four
  `@runtime_checkable` protocols on `Field` / `Connectivity`, checked on hot
  paths. Changing the implementation under them is not a no-op.

Checked, and it narrows the problem: there is **no `TypeVar(default=...)`** in
the tree, so `TypeVar` / `TypeVarTuple` / `ParamSpec` can move to `typing`
freely. The real decision list is four symbols, not eight.

### 4.4 🔴 PEP 695 (`UP040`/`UP046`/`UP047`) is unsupported by `eve` — stays off

`eve/extended_typing.py` documents this precisely: aliases are resolved at the
annotation-dispatch funnels, not by rewriting stored annotations, and *"Not
supported: `ClassVar` behind an alias, and aliases used as runtime values (base
class, constructor, `isinstance()` argument) — the reason ruff's `UP040` stays
off."*

134 `TypeAlias` and 102 generic-syntax sites stay as they are. The `ignore`
entries in §3.3 must carry a comment pointing at that block — which PR 2 moves
into `extra_typing.py` verbatim — so the decision survives.

### 4.5 🟡 `UP006` (`Tuple` → `tuple`) has a history of breaking DSL error paths

The 1.2.1 release already did one pass of this and it surfaced latent bugs in DSL
error formatting. `tests/eve_tests` passing is **not** sufficient evidence —
validate every `UP006`/`UP035` batch against `tests/next_tests` as well.

### 4.6 🟡 Annotation evaluation differs across 3.12 → 3.14

141 of 361 source modules carry `from __future__ import annotations`,
inconsistently. On 3.14, PEP 649 makes annotations lazily evaluated by default,
while the future import forces PEP 563 *strings* — two different regimes for a
codebase that introspects annotations at runtime. With `target-version = py312`
the future import is no longer needed for *syntax* at all; it now only affects
forward references.

**Do not sweep this blindly.** Make it an explicit decision with an ADR (§7),
validated on 3.14, and keep it out of the mechanical PRs. PR 1's conformance
matrix should nonetheless cover both regimes, since that is cheap and is the
thing that would catch a divergence.

### 4.7 🟢 Stale suppressions

`src/` carries 156 `# noqa` and 369 `# type: ignore` comments. After the rewrite
many become unnecessary. `RUF100` (already selected) catches stale `noqa`, and
mypy's `warn_unused_ignores = true` (already on) catches stale `type: ignore`. No
new tooling needed — just do not let the diff grow them.

______________________________________________________________________

## 5. Implementation plan

Ship as a **stack of independently mergeable PRs**, following the repo's
`<topic>-<n>-<slug>` branch convention. Each PR: green `pre-commit run -a`, green
nox sessions for the subpackages it touches, one `CHANGELOG.md` entry.
Squash-merge, Conventional-Commit PR title.

Six PRs. PRs 1 and 2 are prerequisites, not preliminaries: PR 1 unblocks the
annotation rewrite (§4.1), and PR 2 is what makes ruff and mypy able to see it
(§2.3).

> Note for whoever picks this up: a pre-existing draft PR (#2224) already turns on
> ruff `UP`. Rebase it into PR 4 rather than starting a fresh branch.

### 5.1 How the work is split, and why not further

The split is by **how a reviewer checks the PR**, not by topic. Three PRs are
read by running the tests and spot-checking mechanical output; three need a human
to look at every hunk.

| PR                     | `src` files touched | Review mode                                           |
| ---------------------- | ------------------: | ----------------------------------------------------- |
| 1 — eve funnels        |                  ~6 | read every line; new test matrix                      |
| 2 — `extra_typing`     |                  48 | read every line of the module; imports are mechanical |
| 3 — back-compat shims  |                  ~6 | read every line; dependency + `Enum` base changes     |
| 4 — dialect + rulesets |                 233 | check the rule list, spot-check, run tests            |
| 5 — `zip(strict=)`     |                  64 | read every one of 110 sites                           |
| 6 — lint tests + docs  |      tests/ + docs/ | check the ignore list; read the prose                 |

The annotation sweep and the additional rulesets are **one PR** (PR 4). They are
the same kind of work — run `ruff --fix`, enable rules in `pyproject.toml`,
hand-fix the residue — and are reviewed the same way, so splitting them creates
two PRs with an identical review procedure. Measured, the case is stronger than
that: the sweep touches 209 files, the rulesets 157, and **133 of those are the
same files**. The union is 233 — merging costs 24 files beyond what the sweep
already touches, while splitting guarantees a 133-file rebase. Keep them as
separate *commits* inside the PR (§5.5).

The ruff bump folds into PR 4 as its **first commit**, so `ruff format --check`
can be shown clean before any rule changes. That is the whole point of isolating
it; a commit gives the same proof as a PR.

`zip(strict=)` stays separate (PR 5), and this is the one place worth spending an
extra PR. It is the only change in the campaign where a red CI run is a *finding*
rather than a mistake — `strict=True` turning a silent truncation into an
exception means a latent bug was there all along. Inside a 233-file mechanical
diff you cannot
tell whether the failure came from `strict=True` or from `UP007` corrupting an
annotation. That diagnostic separation — and a revert that does not take 1900
annotation rewrites with it — is worth one PR.

Back-compat shims stay separate (PR 3) for the same reason at smaller scale: it
removes a **runtime dependency** and changes an `Enum` base class. Six files that
should stay bisectable, not six files buried in 233.

### 5.2 PR 1 — `fix[eve]: handle modern typing spellings at every annotation funnel`

- Normalize `types.UnionType` in `_make_type_converter`, matching what
  `type_validation.py:229` already does.
- Audit the six funnels of §2.2 for the same shape, per the sweep in §4.1.
- Add a **conformance matrix** to `tests/eve_tests`: every *spelling* of a type
  crossed with every funnel, asserting equivalence. Spellings to cover:
  `Optional[X]` vs `X | None`; `Union[X, Y]` vs `X | Y`; `List[X]` vs `list[X]`;
  `TypeAlias` vs PEP 695 `type X = ...`; string annotations vs live objects; and
  modules with and without `from __future__ import annotations` (§4.6).
- Write the matrix first, watch it fail, then fix. It is the deliverable that
  outlives the campaign — the funnels are six functions, so it is cheap.
- Validate: `nox -s test_eve test_next`.

### 5.3 PR 2 — `refactor[eve]: replace extended_typing re-exports with extra_typing`

Rename **and** de-star in one PR; splitting them churns the same 56 files twice
for no review benefit.

1. Create `eve/extra_typing.py` holding only the ~46 definitions `typing` /
   `typing_extensions` do not provide — no star imports, no `__getattr__`, no
   `__dir__`, no `_DEPRECATED_TYPING_ALIASES` popping. Carry the PEP 695 comment
   block across verbatim (§4.4).
2. **Rebuild the forward-ref namespace explicitly before deleting the star
   imports**, so the change is a refactor and not a discovery (§4.2). Construct it
   from `typing` + `typing_extensions` + the deprecated alias replacements, and
   have `_ForwardRefTypingNamespace.__getattr__` consult `typing_extensions` then
   `typing` instead of `sys.modules[__name__]`. That is shorter than what is there
   now and says what it means.
   `_DEPRECATED_ALIAS_REPLACEMENTS` itself **stays**: user-written annotation
   strings may still spell `typing.List[int]`, and normalizing those to
   `list[int]` is a deliberate guarantee, not a leftover.
3. Rewrite the 56 call sites: 262 uses → `typing`, 8 → `typing_extensions`, 176
   stay. Resolve the four genuinely divergent symbols per §4.3 and record every
   decision — symbol, chosen source, one-line reason — in a table in the
   `extra_typing` module docstring. Otherwise the knowledge that used to live in
   one sentence ("`typing_extensions` wins") is simply deleted.
4. Close the `xtyping.Buffer` TODO in `storage/allocators.py` using
   `collections.abc.Buffer` (§2.4).
5. Delete `typing-modules` from `[tool.ruff.lint]` and the
   `gt4py.eve.extended_typing` mypy override.
6. Add the `typing_extensions` graduation check (§6.6).
7. **No compatibility shim.** Release 1.2.1 already made a comparable breaking
   change to `gt4py.eve`'s typing surface (dropping the deprecated alias
   re-exports) and handled it with a `CHANGELOG` entry; follow that precedent. A
   shim that re-exports from `extra_typing` would reintroduce exactly the
   `__getattr__`-shaped hole this PR removes. Announce under `### General`.

Validate: `nox -s test_eve test_cartesian test_next test_storage test_package`
plus `test_typing_exports` — the last is the only session that checks GT4Py's
typing surface from a *client* perspective, which is exactly what this PR changes.

Expected static-check dividend: mypy starts checking 446 previously-`Any`
references. Budget for it to find real errors; they are not regressions.

### 5.4 PR 3 — `refactor: drop Python 3.8-era back-compat shims`

Small and self-contained, so it stays bisectable. Everything in §2.4 not already
absorbed by PR 2:

- `cached_property` → `functools.cached_property`; drop the `cached-property`
  **runtime dependency**.
- `eve.type_definitions.StrEnum` → `enum.StrEnum`, keeping the exported name so
  downstream imports do not break. Check `__format__` as well as `__str__`: the
  `(str, Enum)` mixin and `enum.StrEnum` have differed there historically, and the
  current class overrides only `__str__`.
- `TransientMemoryMode(str, enum.Enum)` → `enum.StrEnum`.
- Remove the dead `sys.version_info >= (3, 12)` branch in the `extra_typing` tests
  (§2.5).
- Move `frozendict` from `[project].dependencies` to the `test` group.

Validate with `test_package` — it is the session that would catch a dependency
removal that was actually load-bearing.

### 5.5 PR 4 — `refactor: adopt the Python 3.12 dialect and extend the ruff ruleset`

The bulk: 233 `src` files, ≈ 4000 findings, ≈ 3400 auto-fixable. Reviewed by
checking the rule list and spot-checking, not by reading every hunk — which is
why it is one PR and not two (§5.1).

Land as **separately reviewable commits**, in this order:

1. `ruff>=0.16.4` in `[dependency-groups].lint`; `uv lock`. No rule changes yet —
   `ruff format --check` must still report 730 files formatted. If the bump does
   move the formatter, that shows up here alone instead of mixed into the sweep.
2. `--select UP006,UP035 --fix` (719 sites) → run `test_next` **and**
   `test_cartesian`, not just `test_eve` (§4.5).
3. `--select UP045 --fix` (642 sites).
4. `--select UP007 --fix --unsafe-fixes` (264 sites) — unsafe fix; read every hunk
   landing in `eve/`, `next/type_system/`, or a datamodel class body. PR 1's
   conformance matrix is the safety net here.
5. `--select UP004,UP018,UP029,UP032,UP033,UP034,UP037,UP044 --fix` (~41 sites).
6. Add `C4`, `DTZ`, `FLY`, `FURB`, `PERF`, `PGH`, `PIE`, `PYI`, `SIM`, `SLOT`, `W`
   per §3.2–3.3 with the `SIM` ignore list; `--fix` the auto-fixable, hand-fix the
   ~260 that remain.
7. Add `UP` to `select` and `UP040`/`UP046`/`UP047` to `ignore` with the §4.4
   comment; add `[tool.ruff.lint.pydocstyle] convention = 'google'`.
8. `ruff format`; drop `noqa` / `type: ignore` comments now reported unused
   (§4.7).

PR 2 and this PR touch ~260 of the same import lines — once to move `Optional`
off `xtyping`, once to rewrite it to `X | None`. That is deliberate: the
alternative makes the 1900-site rewrite depend on the `typing-modules` escape
hatch and lands both changes on the same lines in one unreviewable diff. Total
churn is not the metric; review cost per PR is.

Run the full nox matrix on 3.12, 3.13 and 3.14 before merging. This is the PR
most likely to break something that only one interpreter notices (§4.6).

### 5.6 PR 5 — `fix: make zip() truncation explicit`

- Drop `B905` from `ignore`.
- 110 sites in 64 files, each a judgement call: `strict=True` where lengths must
  match (the overwhelming majority in IR transforms), `strict=False` with a
  one-line comment where truncation is intended. **No blanket autofix.**
- The one PR in the stack that can legitimately change behaviour — by turning a
  silent wrong answer into an exception. A test that starts failing here is a bug
  found, not a regression; say so in the changelog entry, and open a separate
  issue per genuine bug rather than burying the fix in this PR.

### 5.7 PR 6 — `ci: lint tests and examples, document the Python ≥ 3.12 dialect`

Two disjoint file sets — lint config plus `tests/`, and the docs of §6 — so they
merge without interacting.

- Replace the `tests/**` / `examples/**` blanket exclusion with the
  per-file-ignores of §3.5.
- Same exclusion in the `ruff-check` pre-commit hook — it carries its own
  duplicate `exclude: "^(tests/|docs/|examples/)"` regex with the same TODO. Both
  must go, or config and hook disagree.
- Fix the ~411 `UP` findings and the residue in `tests/`.
- Notebooks under `examples/**` and `docs/user/next/workshop/**` are covered by
  ruff's `jupyter` type in the hook config; re-run and fix.
- All of §6. **Must land in the same release as PR 4**, or the guidelines describe
  a codebase that no longer exists.

______________________________________________________________________

## 6. Documentation work

Non-optional, and the part most likely to be skipped. Landed by PR 6, with the
pieces that describe a module PR 2 rewrites (§6.1, §6.3) carried in that PR.

### 6.1 `CODING_GUIDELINES.md`

Add a short **"Python dialect"** subsection under "Code Style" stating the
enforced rules, so a contributor learns them from the guide rather than from a CI
failure:

- Builtin generics (`list[int]`), not `typing.List`.
- `X | None`, not `Optional[X]`; `X | Y`, not `Union[X, Y]`.
- **`TypeAlias`, not PEP 695 `type X = ...`** — with the reason and a pointer to
  `eve/extra_typing.py`. This is the counter-intuitive one; without it,
  contributors will keep proposing `UP040`.
- Import typing symbols from `typing`; from `typing_extensions` only for what
  3.12 lacks. `eve.extra_typing` is for GT4Py's own definitions, not a
  convenience re-export.
- `zip(..., strict=...)` is mandatory.
- `# noqa` / `# type: ignore` must carry a code (now machine-enforced by `PGH`).

Fix while there — both are stale today, independent of this task:

- The ADR link points at `docs/functional/architecture/Index.md`, which **does
  not exist**. It is `docs/development/ADRs/`.
- The docstring section *permits* Sphinx cross-reference roles
  (`` :class:`Foo` ``), while `AGENTS.md` *forbids* them, and
  `eve/type_definitions.py` uses them. Pick one and make all three agree —
  `CODING_GUIDELINES.md` is authoritative per `AGENTS.md`, so either relax
  `AGENTS.md` or tighten the guide and clean up the existing uses.

### 6.2 `CONTRIBUTING.md`

- §Tools lists the QA stack; add the ruff floor rationale and a pointer to the
  new dialect subsection.
- The `nox -s` example still reads `"test_cartesian-3.12(internal, cpu)"` —
  verify against `nox --list` and fix if the session names have drifted.

### 6.3 `AGENTS.md` / `src/gt4py/next/AGENTS.md`

- Add the dialect rules — they are exactly what a model gets wrong by default:
  every one will write `type X = ...` and `Optional[X]`.
- Note that `eve.extra_typing` is not a `typing` replacement.
- Resolve the docstring-markup contradiction (§6.1).
- Keep inside the ~200-line budget the file sets for itself: one bullet each,
  linking to `CODING_GUIDELINES.md`.

### 6.4 `docs/development/`

- `onboarding.md` links to `CI/cscs-ci.md`; the file is `tools/cscs-ci.md`. Fix.
- `tools/ci-infrastructure.md`: check the QA job description still matches
  `code-quality.yml`.

### 6.5 User docs

- `docs/user/next/QuickstartGuide.md`, `docs/user/cartesian/*.rst`: any code
  sample using `Optional` / `List` teaches the old dialect. Sweep and update.
- Workshop notebooks record `"version": "3.10.x"` in their kernel metadata and
  ship stale tracebacks with `/Versions/3.10/` paths. Cosmetic but visible —
  re-run or scrub the metadata.
- `README.md` §Installation: confirm the stated Python requirement matches
  `requires-python` (`>=3.12, <3.15`, excluding 3.13.10 and 3.14.1).

### 6.6 Keeping it in sync

Three durable mechanisms, in descending order of value — these are what stop the
codebase drifting back:

1. **The conformance matrix** (PR 1). Spelling × funnel, in `tests/eve_tests`. It
   is the only one of the three that fails CI when the invariant breaks, which is
   why it is PR 1 and not an afterthought.
2. **A `typing_extensions` graduation check** (PR 2). After the refactor only two
   symbols (`TypeIs`, `is_protocol`) require `typing_extensions` at 3.12; both
   land in `typing` at 3.13. A ~30-line script, exposed as
   `./scripts/run check typing-extensions-usage` per the `scripts/README.md`
   conventions, can report for every `from typing_extensions import X` whether
   `typing.X` now exists and is the identical object at the current floor. Run it
   in `code-quality` as a non-blocking report, or on demand at each floor bump.
   That turns "we could probably move some imports at the next bump" into a
   command that says which ones.
3. **An ADR.** The decisions in §4.3, §4.4 and §4.6 belong in
   `docs/development/ADRs/`, not only in a source comment and a
   `pyproject.toml` `ignore` entry. `AGENTS.md` already mandates an ADR for
   non-trivial architectural choices, and "we do not use PEP 695 because our
   datamodels resolve annotations at runtime" is exactly that. Fold in §2.3's
   rationale for `extra_typing` and §4.3's per-symbol source table, so the next
   contributor who proposes re-adding a convenience re-export layer finds the
   answer instead of relitigating it.

______________________________________________________________________

## 7. Deliberately deferred

Not part of this campaign; each needs its own decision.

1. **`from __future__ import annotations` policy** (§4.6) — ADR + a 3.14
   validation pass first.
2. **`PTH` / pathlib migration** (42 sites) — not a no-behaviour-change edit.
3. **`D` / pydocstyle** (4764) — a prose campaign, scoped per subpackage.
4. **`NPY002` legacy random in tests** (159) — test hygiene; changes fixture
   values, needs its own review.
5. **Replacing the `black` runtime dependency** used by `eve/codegen.py` to format
   generated code with `ruff format`. Tempting — it would drop a heavy runtime
   dependency — but it changes generated-code formatting and therefore cache keys
   and golden files. **Architectural; requires an ADR.** Explicitly out of scope
   per §1.2.
6. **`RSE102`** (169) — zero-risk, zero-benefit. Fold into an unrelated PR someday
   or never.
