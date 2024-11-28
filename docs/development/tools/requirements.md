# Requirements

The specification of required third-party packages is scattered and partially duplicated across several configuration files used by several tools. Keeping all package requirements in sync manually is challenging and error-prone. Therefore, in this project we use [pip-tools](https://pip-tools.readthedocs.io/en/latest/) and the [cog](https://nedbatchelder.com/code/cog/) file generation tool to avoid inconsistencies.

The following files in this repository contain information about required third-party packages:

- `pyproject.toml`: GT4Py [package configuration](https://peps.python.org/pep-0621/) used by the build backend (`setuptools`). Install dependencies are specified in the _project.dependencies_ and _project.optional-dependencies_ tables.
- `requirements-dev.in`: [requirements file](https://pip.pypa.io/en/stable/reference/requirements-file-format/) used by **pip**. It contains a list of packages required only for the development of GT4Py.
- `requirements-dev.txt`: requirements file used by **pip**. It contains a completely frozen list of all packages required for installing and developing GT4Py. It is used by **pip** and **tox** to initialize the standard development and testing environments. It is automatically generated automatically from `requirements-dev.in` by **pip-compile**, when running the **tox** environment to update requirements.
- `constraints.txt`: [constraints file](https://pip.pypa.io/en/stable/user_guide/#constraints-files) used by **pip** and **tox** to initialize a subset of the standard development environment making sure that if other packages are installed, transitive dependencies are taken from the frozen package list. It is generated automatically from `requirements-dev.in` using **pip-compile**.
- `min-requirements-test.txt`: requirements file used by **pip**. It contains the minimum list of requirements to run GT4Py tests with the oldest compatible versions of all dependencies. It is generated automatically from `pyproject.toml` using **cog**.
- `min-extra-requirements-test.txt`: requirements file used by **pip**. It contains the minimum list of requirements to run GT4Py tests with the oldest compatible versions of all dependencies, additionally including all GT4Py extras. It is generated automatically from `pyproject.toml` using **cog**.
- `.pre-commit-config.yaml`: **pre-commit** configuration with settings for many linting and formatting tools. Part of its content is generated automatically from `pyproject.toml` using **cog**.

The expected workflow to update GT4Py requirements is as follows:

1. For changes in the GT4Py package dependencies, update the relevant table in `pyproject.toml`. When adding new tables to the _project.optional-dependencies_ section, make sure to add the new table as a dependency of the `all-` extra tables when possible.

2. For changes in the development tools, update the `requirements-dev.in` file. Note that required project packages already appearing in `pyproject.toml` should not be duplicated here.

3. Run the **tox** _requirements-base_ environment to update all files automatically with **pip-compile** and **cog**. Note that **pip-compile** will most likely update the versions of some unrelated tools if new versions are available in PyPI.

   ```bash
   tox r -e requirements-base
   ```

4. Check that the **mypy** mirror used by **pre-commit** (https://github.com/pre-commit/mirrors-mypy) in `.pre-commit-config.yaml` supports the same version as in `constraints.txt`, and manually update the `rev` version number.
