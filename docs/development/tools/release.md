# Releasing Process

This document describes the process of releasing new versions of GT4Py.

Currently, GT4Py releases are published in PyPI (and TestPyPI) and also as commit tags in the main GitHub repository. To create a new release you should:

1. Make sure all the expected changes (new features, bug fixes, documentation changes, etc.) are already included in the `main` public branch.

2. Use **bump2version** to update the version number.

   ```bash
   $ bump2version minor # or patch
   ```

3. Update the [CHANGELOG.md](CHANGELOG.md) file to document the changes included in the new release. Note that this step becomes much simpler when commit messages follow the [Conventional Commits][conventional-commits] convention as encouraged in the [Pull Request and Merge Guidelines](CONTRIBUTING.md#pull-request-and-merge-guidelines) section of the contributing guidelines.

4. Commit the changes with the following message:

   ```bash
   $ git commit -m 'Releasing 0.{M}.{m}.{p} version.'
   ```

5. On the GitHub website go to _Releases_ and _Draft a new release_. Choose `v0.{M}.{m}.{p}` as tag and select a branch (usually `main`). Follow the style of the previous releases for the title (`GT4Py v0.{M}.{m}.{p}`) and description. Then _Publish release_.

6. Upload distribution package to TestPyPI and quickly test that it works properly.

7. Upload distribution package to PyPI and quickly that test it works properly.

<!-- Reference links -->

[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/#summary
