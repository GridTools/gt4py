# Releasing Process

This document describes the process of releasing new versions of GT4Py.

Currently, GT4Py releases are published in PyPI (and TestPyPI) and also as commit tags in the main GitHub repository. To create a new release you should:

1. Make sure all the expected changes (new features, bug fixes, documentation changes, etc.) are already included in the `main` public branch.

2. Update the [CHANGELOG.md](CHANGELOG.md) file to document the changes included in the new release. Note that this step becomes much simpler when commit messages follow the [Conventional Commits][conventional-commits] convention as encouraged in the [Pull Request and Merge Guidelines](CONTRIBUTING.md#pull-request-and-merge-guidelines) section of the contributing guidelines.

3. Commit the changes with the following message:

   ```bash
   $ git commit -m 'Releasing {M}.{m}.{p} version.'
   ```

4. On the GitHub website go to _Releases_ and _Create a new release_. Choose `v{M}.{m}.{p}` as tag and select a branch (usually `main`). Follow the style of the previous releases for the title (`GT4Py v{M}.{m}.{p}`) and description. Then _Publish release_.

5. Publishing the release will trigger a GitHub action to deploy to TestPyPI. Install the package from TestPyPi and do basic tests.

6. If tests are ok, manually trigger the deploy GitHub action selecting the release tag as target. This will publish the package to PyPI. Install the package and test if it works.

## PyPi and TestPyPi accounts

The account is called `gridtools`. Credentials can be found in the bitwarden of CSCS. For 2FA, the recovery keys are stored in bitwarden, too. In case a new developer should get access, the recovery keys can be used to setup the authentication app (for all developers who should have access).

<!-- Reference links -->

[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/#summary
