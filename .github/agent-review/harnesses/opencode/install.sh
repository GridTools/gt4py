#!/usr/bin/env bash
# Installs opencode, pinned by package-lock.json (npm integrity hashes).
#
# Usage: install.sh <prefix>
#
# Runs in the secretless 'prepare' job. Depends on the Linux x64 'baseline'
# platform package directly (it runs on CPUs without AVX2, too) instead of
# the '@opencode/cli' wrapper, whose postinstall script only hard-links this
# same binary and whose lockfile pulls the binaries of every platform. No
# install scripts run.

set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: install.sh <prefix>" >&2; exit 1; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prefix=$1
package=@opencode/cli-linux-x64-baseline

mkdir -p "${prefix}"
cp "${here}/package.json" "${here}/package-lock.json" "${prefix}/"
(cd "${prefix}" && npm ci --ignore-scripts --no-audit --no-fund --loglevel=error)

want=$(jq -r --arg p "${package}" '.dependencies[$p]' "${here}/package.json")
got=$("${prefix}/node_modules/${package}/bin/opencode" --version)
[ "${got}" = "opencode v${want}" ] || { echo "Version mismatch: got '${got}', want 'opencode v${want}'." >&2; exit 1; }
echo "${want}" >"${prefix}/VERSION"
