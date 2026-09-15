# CI infrastructure

Any test job that runs on CI is encoded in automation tools like **nox** and **pre-commit** and can be run locally instead.

## GitHub Workflows

The following workflows are currently active.

### Code Quality

A `Code Quality` workflow runs `pre-commit` to check code quality requirements through tools like **mypy** or **ruff**.

### Tests

The `Test code components (CPU)`, `Test Package Root` and `Test examples in documentation` workflows run the correspondent `nox` sessions testing different parts of GT4Py. `Test code components (CPU)` is a parameterized workflow (see more details in [_Code component test sessions_](#code-component-test-sessions). `Test Package Root` tests the root package code and `Test examples in documentation` tests the code samples in the documentation. In all cases only tests are run that do not require the presence of a GPU.

#### Code component test sessions

The idea is to run test sessions only when needed. This means that a set of tests are only run when the associated sources or the sources of a dependency change. For example, `eve` tests will not be run when only GT4Py cartesian sources are changed. The criteria to trigger tests sessions is encoded in the custom [`nox-sessions-config`](../../../nox-sessions-config.yml) configuration file, which follows a similar structure of the GitHub Actions `paths` and `paths-ignore` filters. The [`/scripts/python/nox_sessions.py`](../../../scripts/python/nox_sessions.py) script reads this file to decide which nox sessions are required for the existing changes from a base commit.

### Daily CI

There is an extra CI workflow on GitHub scheduled to run daily and testing `main` with different sets of requirements: the `highest` and the `lowest-direct` `uv` resolution strategies. Failures are accessible in [GitHub web interface](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml) and as the 'Daily CI' badge in the main [README.md](../../../README.md) file. Additionally, in case of failure a message _might_ be posted in the [#ci-notifications](https://app.slack.com/client/T0A5HP547/C0E145U65) channel of the GridTols slack, but those notifications do not work reliably.

## CSCS-CI

CI pipelines for all tests can be triggered via CSCS-CI. These automatically run from a Gitlab mirror for whitelisted users only, and have to be explicitly run by a whitelisted user via the comment "cscs-ci run default" on PRs from other users. There is currently no finegrained control over which subpackage tests are run. Neither can a subset be started manually from the comments nor can tests be skipped based on which files have been changed. Both are achievable (the latter with considerable effort), however given the current duration of the pipeline it does not seem worth doing so.

Since all tests routinely run here, this might be a better match for reintroducing test coverage in the future than GitHub workflows.

Additional information on how to change this process, such as adding whitelisted users, regenerating tokens etc can be found in [cscs-ci.md](cscs-ci.md)

## Tested operating systems

The testing workflows use a matrix strategy to run the automated tests on multiple operating systems. There is, however, a baseline 2000 hour monthly limit for the total time CI runs take. Since MacOS builds consume node hours 10 times as fast, this leaves an effective 200 hours for MacOS CI runs. MacOS is therefore only part of the Daily CI matrix, while the per-pull-request workflows run on Ubuntu only.

## Future improvements

- Reenable code coverage workflows (potentially on CSCS-CI).
- Split code quality: it might be better to run code quality tools separate for each project in the monorepo.
- Split documentation: once there is proper HTML documentation generated for the projects, it might make sense to have that run as one job per project.
- Template for tests: It would probably make sense to reuse some of the workflow descriptions for the tests.
