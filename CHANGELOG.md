# GT4Py Changelog

Notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [1.0.4] - 2024-09-20

### License

GT4Py is now licensed under the BSD license. SPDX-License-Identifier: BSD-3-Clause

### Cartesian

- Introduced a `GlobalTable` which is a data dimensions only Field. Access is provided via `.A` which also becomes a way to access data dimensions on regular Field.
- Added an error message if a non existing backend is selected.
- Allow setting compiler optimization level and flags on a per stencil basis
- Added `GT4PY_EXTRA_COMPILE_ARGS`, `GT4PY_EXTRA_LINK_ARGS` and `DACE_DEFAULT_BLOCK_SIZE` environment variables
- Fixes for the DaCe backend
- Various style modernizations

#### Deprecation

- The `cuda` backend is deprecated (enable by setting environment variable `GT4PY_GTC_CUDA_USE=1`), use `gt:gpu` or `dace:gpu` instead.

### Development

- Replaced flake8, black with ruff
- Added CI plan with GH200 GPUs

### Next

See commit history.

## [1.0.3] - 2024-02-07

### General

- Support for Python 3.11 and updated dependencies

### Testing

- Testing of Jupyter notebooks in CI

### Next

See commit history.

## [1.0.2] - 2024-01-24

### Cartesian

- Compatibility of `gt4py.next` Fields with `gt4py.cartesian` computations.
- Fixes for DaCe 0.15.1 compatibility.
- Added `log10` as native function.
- Make `scipy` optional: get `scipy` by installing `gt4py[full]` for best performance with `numpy` backend.

### Storage

- Refactored low-level storage allocation.

### Next

See commit history.

## [1.0.1] - 2023-02-20

First version including the experimental `gt4py.next` aka _Declarative GT4Py_. The `gt4py.next` package is excluded from semantic versioning.

### Cartesian

- Parametrized dtype: see option 4 of [the gtscript concept workshop](https://github.com/GridTools/concepts/blob/master/collaboration/gtscript-workshop/GTScript-Syntax-Discussion.md#gtscript-syntax-discussed-issues-20200829])

## [1.0.0] - 2022-12-21

### Added

- Remove the `Storage` classes and implement the new _"No Storage"_ concept.
- Support interfaces for calling stencils with arbitrary buffer objects (for details see [docs/arrays.rst](docs/gt4py/arrays.rst) and [PR #868](https://github.com/GridTools/gt4py/pull/868)).
- Updated documentation, logo and license headers.

### Changed

- Important `gt4py` package reorganization and modularization.
- Moved most of existing functionality into `gt4py.cartesian`.
- Moved `gtc` package into `gt4py.cartesian`.
- Moved `eve` package into `gt4py`.
- Refactored `gt4py.storage` to avoid dependencies on `gt4py.cartesian`.
- Tests split into sub-packages and reorganized as `unit_tests` and `integration_tests`.

## [0.1.0] - 2022-09-29

Last development version using old-style `Storage` class.
