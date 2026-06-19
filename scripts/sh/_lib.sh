#!/usr/bin/env bash
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared functions for bash dev-scripts.  Source, don't execute.
# Sourcing must not change the caller's shell options, so each executable
# script is responsible for its own `set -euo pipefail`.
# Usage (in sibling scripts):
#   set -euo pipefail
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Logging ──────────────────────────────────────────────────────────────────

log_info()  { printf '\033[0;34m[INFO]\033[0m  %s\n' "$*"; }
log_warn()  { printf '\033[0;33m[WARN]\033[0m  %s\n' "$*" >&2; }
log_error() { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

# ── Guards ───────────────────────────────────────────────────────────────────

require_cmd() {
    command -v "$1" &>/dev/null || {
        log_error "Required command '$1' not found in PATH"
        exit 1
    }
}

require_var() {
    local varname="$1"
    if [[ -z "${!varname:-}" ]]; then
        log_error "Required environment variable '$varname' is not set"
        exit 1
    fi
}
