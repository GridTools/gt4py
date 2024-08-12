# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib


file = pathlib.Path("info.txt")
content = file.read_text()

lines = content.split("\n")

ref_type = lines[0]
"""${{github.ref_type}}"""
ref = lines[1]
"""${{github.ref}}"""
sha = lines[2]
"""${{github.sha}}"""
pr_number = lines[3]
"""${{github.event.number}}"""
pr_head_ref = lines[4]
"""${{github.event.pull_request.head.ref}}"""
pr_head_sha = lines[5]
"""${{github.event.pull_request.head.sha}}"""
run_id = lines[6]
"""${{github.run_id}}"""

is_pr = bool(pr_head_ref)

commit_parent = ""
"""
The commit SHA of the parent for which you are uploading coverage. If not
present, the parent will be determined using the API of your repository
provider. When using the repository provider's API, the parent is determined
via finding the closest ancestor to the commit.
"""

override_build = run_id
"""Specify the build number"""

override_branch = ""
"""Specify the branch name"""
if is_pr:
    override_branch = pr_head_ref
elif ref_type == "branch":
    override_branch = ref

override_commit = pr_head_sha if is_pr else sha
"""Specify the commit SHA"""

override_pr = pr_number
"""Specify the pull request number"""

override_tag = ref if ref_type == "tag" else ""
"""Specify the git tag"""

github_env_file = pathlib.Path(os.environ["GITHUB_ENV"])
with open(github_env_file, "w") as file:
    file.write(f"CODECOV_COMMIT_PARENT={commit_parent}\n")
    file.write(f"CODECOV_OVERRIDE_BUILD={override_build}\n")
    file.write(f"CODECOV_OVERRIDE_BRANCH={override_branch}\n")
    file.write(f"CODECOV_OVERRIDE_COMMIT={override_commit}\n")
    file.write(f"CODECOV_OVERRIDE_PR={override_pr}\n")
    file.write(f"CODECOV_OVERRIDE_TAG={override_tag}\n")
