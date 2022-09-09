# Contributing

Contributions to GT4Py are welcome and greatly appreciated. Proper credit will be given to contributors by adding their names to the [AUTHORS.md](AUTHORS.md) file. Note that [ETH Zurich](https://ethz.ch/en.html) is the owner of the GridTools project and the GT4Py library, therefore external contributors must sign a contributor assignment agreement.


## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/gridtools/gt4py/issues](https://github.com/gridtools/gt4py/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Python interpreter version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the [GitHub issues](https://github.com/gridtools/gt4py/issues) for bugs. Anything tagged with `bug` and `help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the [GitHub issues](https://github.com/gridtools/gt4py/issues) for features. Anything tagged with `enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

GT4Py could always use more documentation, whether as part of the official GT4Py docs, in docstrings, or even on the web in blog posts and articles.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/gridtools/gt4py/issues](https://github.com/gridtools/gt4py/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is an open-source project, and that contributions
  are welcome :)


## Getting Started 

Ready to start contributing? Follow these steps:

1. Fork the [GT4Py](https://github.com/gridtools/gt4py) repo on GitHub.

2. Clone your fork locally and check out the relevant branch:

   ```bash
   $ git clone git@github.com:your_name_here/gt4py.git
   $ cd gt4py
   $ git checkout functional
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

We use [pre-commit](https://pre-commit.com/) to run several auto-formatting and linting tools. You should always execute it locally before opening a pull request. `pre-commit` can be installed as a _git hook_ to automatically check the staged changes before commiting:

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

# Run only some of the tools (e.g. flake8)
pre-commit run -a flake8
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

We recommended you to use `tox` to run the complete test suite in different environments:

```bash
# List all the available test environments
tox -a

# Run test suite in a specific environment
tox -e py310-base
```

`tox` runs the package installation script and tests reliably by using properly isolated environments. Additionally, `tox` is configured to generate HTML test coverage reports in `tests/_reports/coverage_html/` at the end.


## Pull Request and Merge Guidelines

Before submitting a pull request, check that it meets the following criteria:

1. The pull request should include tests.
2. If the pull request adds functionality, it should be documented both in the code docstrings and in the official documentation.
3. If the pull request contains important design changes, it should contain a new ADR documenting the rationale behind the final decision.
4. The pull request should have a proper description of its intent and the main changes in the code. In general this description should be used as commit message if the PR is approved (check point **6.** below).
5. Pick one reviewer and try to contact him directly to let him know about the PR. If there is no feedback in 24h/48h try to contact him again or pick another reviewer.
6. Once the PR has been approved, it should be squash-merged as soon as possible with a meaningful description of the changes. Although it is optional, we encourage the use of the [Conventional Commits][conventional-commits] specification for writing informative and automation-friendly commit messages (_commit types: `fix`, `feat`, `build`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`_).


## Releasing Process

This section documents the process of releasing new versions of GT4Py and it's only useful to core members of the development team.

Currently, GT4Py releases are published as commit tags in the main GitHub repository (although they will be soon available in TestPyPi and PyPI). To create a new release you should:

1. Make sure all the expected changes (new features, bug fixes, documentation changes, etc.) are already included in the main public branch.

2. Use `bump2version` to update the version number.

   ```bash
   $ bump2version minor # or patch
   ```

3. Update the [CHANGELOG.md](CHANGELOG.md) file to document the changes included in the new release. This process can be fully or partially automated if commit messages follow the [Conventional Commits][conventional-commits] convention as suggested in the section about [Pull Request and Merge Guidelines](#pull-request-and-merge-guidelines). 

4. Commit the changes with the following commit message:

   ```bash
   $ git commit -m 'Releasing 0.{M}.{m}.{p} version.'
   ```

5. Add a new lightweight tag like: `v0.{M}.{m}.{p}`

   ```bash
   $ git tag v0.{M}.{m}.{p}
   ```

6. Push the new commit and tag directly to the upstream repo:

   ```bash
   $ git push upstream 
   $ git push upstream v0.{M}.{m}.{p}
   ```

7. Go to the GitHub website and verify that the new tag has been successfully created.


## Tools

As mentioned above, we use several tools that help us write high-quality code. New tools could be added in the future, especially if they do not add a large overhead to our workflow and we think they bring extra benefits to keep our codebase in shape. The most important ones we currently rely on are:

   - [Black][black] for autoformatting source code.
   - [isort][isort] for autoformatting import statements.
   - [Flake8][flake8] for style enforcement and code linting.
   - [pre-commit][pre-commit] for automating the execution of QA tools.
   - [pytest][pytest] for writing readable tests, extended with:
      + [Coverage.py][coverage] and [pytest-cov][pytest-cov] for test coverage reports.
      + [pytest-xdist][pytest-xdist] for running tests in parallel.
   - [tox][tox] for testing automating with different environments.
   - [sphinx][sphinx] for generating documentation, extended with:
      + [sphinx-autodoc][sphinx-autodoc] and [sphinx-napoleon][sphinx-napoleon] for extracting API documentation from docstrings.
      + [jupytext][jupytext] for writing user documentation with code examples.


<!-- Reference links -->

[black]: https://black.readthedocs.io/en/stable/
[commitizen]: https://commitizen-tools.github.io/commitizen/
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/#summary
[coverage]: https://coverage.readthedocs.io/
[flake8]: https://flake8.pycqa.org/
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[isort]: https://pycqa.github.io/isort/
[jupytext]: https://jupytext.readthedocs.io/
[pre-commit]: https://pre-commit.com/
[pylint]: https://pylint.pycqa.org/
[pytest]: https://docs.pytest.org/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[pytest-xdist]: https://pytest-xdist.readthedocs.io/en/latest/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[tox]: https://tox.wiki/en/latest/#

