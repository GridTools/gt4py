# Releasing Process

This document describes the process of releasing new versions of GT4Py.

Currently, GT4Py releases are published in PyPI (and TestPyPI) and also as commit tags in the main GitHub repository. To create a new release you should:

1. Make sure all the expected changes (new features, bug fixes, documentation changes, etc.) are already included in the `main` public branch.

2. Use **bump2version**to update the version number.

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

8. Upload distribution package to TestPyPI and quickly test that it works properly.

9. Upload distribution package to PyPI and quickly that test it works properly.
