# CI infrastructure

## Workflows

The following workflows are currently active:

![workflows](workflows.drawio.svg)

### Main workflows

The `Eve / Test` and `Gt4py / Test` workflows run the automated tests for the two packages. These workflows also save the code coverage output as workflow artifacts.

The `Eve / Coverage` and `Gt4py / Coverage` workflows are triggered by a successful run of the tests. They download the coverage artifacts and publish them to codecov.io.

The `Documentation` workflow executes the Jupyter nodebook of the quick start guide to make sure it's up to date. 

The `Code Quality` workflow runs pre-commit to check code quality requirements through tools like mypy or flake8.

### When are workflows triggered

The general idea is to run workflows only when needed. In this monorepo structure, this practically means that a set of tests are only run when the associated sources or the sources of a dependency change. For example, eve tests will not be run when only GT4Py sources are changed.

## Integration with external tools

Workflows that integrate with external code quality tools (i.e. codecov.io, SonarCloud) need special treatment for security reasons. Such workflows use a secret token to interact with the external tools. Anyone having access to the secret token can interact with the external tools and can thus "hack" it by publishing spoofed code coverage results, for example. To prevent the exposure of the secret tokens, GitHub allows repository owners to record secret tokens associated with the repository. These repository secrets can then be safely accessed from CI workflows.

The repository secrets, however, are only available within the main repository, not its forks. Otherwise, someone could make a fork, create a pull request with a malicious workflow, and steal the secrets. This also makes it impossible to publish code coverage results from a workflow triggered by a pull request from a fork.

To resolve this issue, the coverage workflows are triggered in the context of the main repository after the tests run in the context of the fork using the `workflow_run` trigger. The test workflows upload the coverage results as artifacts which are then downloaded in the subsequent workflow that publishes them to codecov.io. The test workflows also save the context such as run ID or PR number in an artifact, which is then forwarded by the subsequent workflow to the external tool.

## Future improvements

- Split code quality?
- Split documentation?
- Template for tests



