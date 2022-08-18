# Contributing

Contributions to GT4Py are welcome, and they are greatly appreciated. Proper credit will be given to contributors by adding their names to the [AUTHORS.md](AUTHORS.md) file. Note that ETH Zurich is the owner of the GridTools project and the GT4Py library, therefore external contributors must sign a contributor assignment agreement.


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

GT4Py could always use more documentation, whether as part of the official GT4Py docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/gridtools/gt4py/issues](https://github.com/gridtools/gt4py/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)


## Getting Started 

Ready to start contributing?

1. Fork the [GT4Py](https://github.com/gridtools/gt4py) repo on GitHub.

2. Clone your fork locally:

   ```bash
   $ git clone git@github.com:your_name_here/gt4py.git
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

5. When you're done making changes, check that your changes pass code format and analysis checks with `pre-commit`, and regression and units tests (including  other Python versions) with `tox`:

   ```bash
   $ pre-commit run
   $ tox
   ```

   [README.md](README.md) also contains more details.

6. Commit your changes and push your branch to GitHub:

   ```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request (PR) through the [GitHub](https://github.com/gridtools/gt4py) website.


## Pull Request and Merge Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, it should be documented both in the code docstrings and in the docs.
3. The pull request should contain a meaninful description of the intent of the PR and a summary of the main changes and design issues in the code for the reviewers.
4. Ask for a review ...
5. MErge using a commit message from ... [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

```bash
$ bump2version patch
$ git push
$ git push --tags
```
