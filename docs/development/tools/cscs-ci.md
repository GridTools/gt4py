# CSCS-CI Configuration

CSCS provides a way of running CI on it's machines. This is currently only available via gitlab, however there is a system in place that automatically sets up a GitLab mirror for this through a few simple steps. On Github the whole pipeline will show up as a single job, with a link to the GitLab pipeline. All the pipelines can be seen on the [pipeline page](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4455690602105886/4525297225819146/-/pipelines) of the mirror.

## Initial Setup

Follow the steps in the [CSCS-CI documentation](https://confluence.cscs.ch/x/UAXJMw). As mentioned in the documentation, you will require the help of someone who can register the project to be allowed to run CI on CSCS machines.

## Current Configuration

The configuration can be viewed and changed on the [setup page](https://cicd-ext-mw.cscs.ch/ci/setup_ci) after logging in with the credentials for "cscs-ci setup for gt4py", which can be found on the internal CSCS credentials store in the GT4Py folder.

### Pipeline File

The pipeline config for the default pipeline is at `ci/cscs-ci.yml`. Check the [syntax reference for GitLab CI](https://docs.gitlab.com/ee/ci/yaml/) if you are not familiar with it.

### Whitelisted Users

New core GT4Py developers should be added to the whitelist. This means that pipelines run on their PRs automatically and they can run them on other contributor's pipelines by commenting "cscs-ci run [pipeline name]".

### Pipeline

There can be more than one pipeline, each with it's own entry point yaml file. By default there is only one (named "default") which allows omitting the pipeline name in "cscs-ci run" comments on PRs. Each pipeline can override global settings like who is allowed to run them.

### Regenerating Notification Token

Use the `gridtoolsjenkins` github user (credentials in the internal CSCS credentials store) to follow the steps outlined on the setup page (see above). You may be able to regenerate the existing token instead of creating a new one.

### Changing Existing Pipeline Entry Point

Note that for the duration between the first and last steps, the pipeline will be broken for all other PRs

1. Change the entry point name of the pipeline on the setup page
2. Rename the yaml file, commit the change, push and open a pull request to main
3. Merge to main

If you intend to make additional changes that require testing, consider creating a separate pipeline first with the new name of the entry point file and keeping it alive until you are ready to merge to main.

## Caveats Of CSCS-CI

The GitLab pipeline is **not** triggered via a push event. This means any gitlab-ci features that rely on comparing files to the previous commit (like [`only:changes`](https://docs.gitlab.com/ee/ci/yaml/#onlychanges--exceptchanges)) will not work.
