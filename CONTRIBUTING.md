# Contributing

Contributions to GT4Py are welcome and greatly appreciated. Proper credit will be given to contributors by adding their names to the [AUTHORS.md](AUTHORS.md) file.

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/gridtools/gt4py/issues](https://github.com/gridtools/gt4py/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Python interpreter version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the [GitHub issues](https://github.com/gridtools/gt4py/issues) for bugs. Anything tagged with `bug` and `help wanted` is open to whomever wants to implement it.

### Implement Features

Look through the [GitHub issues](https://github.com/gridtools/gt4py/issues) for features. Anything tagged with `enhancement` and `help wanted` is open to whomever wants to implement it.

### Write Documentation

GT4Py could always use more documentation, whether as part of the official GT4Py docs, in docstrings, or even on the web in blog posts and articles.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/gridtools/gt4py/issues](https://github.com/gridtools/gt4py/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is an open-source project and that contributions
  are welcome :)

## Getting Started

Ready to start contributing? We use a [fork and pull request](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) workflow for contributions. Pull requests need to pass all the automated checks as well as a review before they can be merged. To set up properly your local development environment follow these steps:

1. Fork the [GT4Py](https://github.com/gridtools/gt4py) repo on GitHub.

2. Clone your fork locally and check out the relevant branch:

   ```bash
   $ git clone git@github.com:your_name_here/gt4py.git
   $ cd gt4py
   $ git checkout main
   ```

3. Follow instructions in the [README.md](README.md) file to set up an environment for local development. For example:

   ```bash
   $ tox --devenv .venv
   $ source .venv/bin/activate
   ```

4. Create a branch for local development:

   ```bash
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally. Make sure you follow the project code style documented in [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

5. When you're done making changes, check that your code complies with the project code style and other quality assurance (QA) practices using `pre-commit`. Additionally, make sure that unit and regression tests pass for all supported Python versions by running `tox`:

   ```bash
   $ pre-commit run
   $ tox
   ```

   Read [Testing](#testing) section below for further details.

6. Commit your changes and push your branch to GitHub:

   ```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request (PR) on [GT4Py's GitHub page](https://github.com/gridtools/gt4py).

## Testing

### Quality Assurance

We use [pre-commit](https://pre-commit.com/) to run several auto-formatting and linting tools. You should always execute it locally before opening a pull request. `pre-commit` can be installed as a _git hook_ to automatically check the staged changes before committing:

```bash
# Install pre-commit as a git hook and initialized all the configured tools
pre-commit install --install-hooks
```

Alternatively, it can be executed on demand from the command line:

```bash
# Check only the staged changes
pre-commit run

# Check all the files in the repository: -a / --all-files
pre-commit run -a

# Run only some of the tools (e.g. ruff)
pre-commit run -a ruff
```

### Unit and Regression Tests

In the GT4Py project we use the [pytest](https://pytest.org/) framework for testing our code. `pytest` comes with a very convenient CLI tool to run tests. For example:

```bash
# Run tests inside `path/to/test/folder`
pytest path/to/test/folder

# Run tests stopping immediately on first error: -x / --exitfirst
pytest -x tests/

# Run tests matching the pattern: -k pattern (supports boolean operators)
pytest -k pattern tests/

# Run tests in parallel: -n NUM_OF_PROCS (or `auto`)
pytest -n auto tests/

# Run only tests that failed last time: --lf / --last-failed
pytest --lf tests/

# Run all the tests starting with the tests that failed last time:
# --ff / --failed-first
pytest --ff tests/

# Run tests with more informative output:
#   -v / --verbose          - increase verbosity
#   -l / --showlocalsflag   - show locals in tracebacks
#   -s                      - show tests outputs to stdout
pytest -v -l -s tests/
```

Check `pytest` documentation (`pytest --help`) for all the options to select and execute tests.

We recommended you to use `tox` for most development-related tasks, like running the complete test suite in different environments. `tox` runs the package installation script in properly isolated environments to run tests (or other tasks) in a reproducible way. A simple way to start with tox could be:

```bash
# List all the available task environments
tox list

# Run a specific task environment
tox run -e cartesian-py38-internal-cpu
```

Check `tox` documentation (`tox --help`) for the complete reference.

<!--
TODO: add test coverage instructions
Additionally, `tox` is configured to generate HTML test coverage reports in `tests/_reports/coverage_html/` at the end. -->

## Pull Requests (PRs) and Merge Guidelines

Before submitting a pull request, check that it meets the following criteria:

1. The pull request should include tests.
2. If the pull request adds functionality, it should be documented both in the code docstrings and in the official documentation.
3. If the pull request contains important design changes, it should contain a new ADR documenting the rationale behind the final decision.
4. The pull request should have a proper description of its intent and the main changes in the code. In general this description should be used as commit message if the pull request is approved (check point **6.** below).
5. If the pull request contains code authored by first-time contributors, check they have been added to [AUTHORS.md](AUTHORS.md) file.
6. Pick one reviewer and try to contact them directly to let them know about the pull request. If there is no feedback in 24h/48h try to contact them again or pick another reviewer.
7. Once the pull request has been approved, it should be squash-merged as soon as possible with a meaningful description of the changes. Although it is optional, we encourage the use of the [Conventional Commits][conventional-commits] specification for writing informative and automation-friendly commit messages (_commit types: `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `feature`, `style`, `test`_).

## Tools

As mentioned above, we use several tools to help us write high-quality code. New tools could be added in the future, especially if they do not add a large overhead to our workflow and they bring extra benefits to keep our codebase in shape. The most important ones which we currently rely on are:

- [ruff][ruff] for style enforcement and code linting.
- [pre-commit][pre-commit] for automating the execution of QA tools.
- [pytest][pytest] for writing readable tests, extended with:
  - [Coverage.py][coverage] and [pytest-cov][pytest-cov] for test coverage reports.
  - [pytest-xdist][pytest-xdist] for running tests in parallel.
- [tox][tox] for testing and task automation with different environments.
- [sphinx][sphinx] for generating documentation, extended with:
  - [sphinx-autodoc][sphinx-autodoc] and [sphinx-napoleon][sphinx-napoleon] for extracting API documentation from docstrings.
  - [jupytext][jupytext] for writing new user documentation with code examples.

<!-- Reference links -->

[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/#summary
[coverage]: https://coverage.readthedocs.io/
[ruff]: https://astral.sh/ruff
[jupytext]: https://jupytext.readthedocs.io/
[pre-commit]: https://pre-commit.com/
[pytest]: https://docs.pytest.org/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[pytest-xdist]: https://pytest-xdist.readthedocs.io/en/latest/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
[tox]: https://tox.wiki/en/latest/
