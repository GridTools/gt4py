# GT4Py Changelog

Notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

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
