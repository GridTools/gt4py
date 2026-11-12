#!/usr/bin/env bash
# Installs Pi with its own Node.js, pinned by package-lock.json (npm integrity
# hashes) and node.pin (version and SHA-256).
#
# Usage: install.sh <prefix>
#
# Runs in the secretless 'prepare' job. Pi is pure JavaScript and WebAssembly
# without native addons; no install scripts run.

set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: install.sh <prefix>" >&2; exit 1; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prefix=$1
package=@earendil-works/pi-coding-agent

read -r node_version node_sha < <(grep -v '^#' "${here}/node.pin" | head -n 1)
[[ "${node_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid version '${node_version}' in node.pin." >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT
curl -fsSL --retry 3 -o "${tmp}/node.tar.xz" \
  "https://nodejs.org/dist/v${node_version}/node-v${node_version}-linux-x64.tar.xz"
echo "${node_sha}  ${tmp}/node.tar.xz" | sha256sum -c --quiet -
tar -xJf "${tmp}/node.tar.xz" -C "${tmp}" --strip-components=2 "node-v${node_version}-linux-x64/bin/node"
mkdir -p "${prefix}/node/bin"
install -m 0555 "${tmp}/node" "${prefix}/node/bin/node"

cp "${here}/package.json" "${here}/package-lock.json" "${prefix}/"
(cd "${prefix}" && npm ci --ignore-scripts --no-audit --no-fund --loglevel=error)

want=$(jq -r --arg p "${package}" '.dependencies[$p]' "${here}/package.json")
got=$(HOME="${tmp}" PI_OFFLINE=1 PI_TELEMETRY=0 "${prefix}/node/bin/node" \
  "${prefix}/node_modules/${package}/dist/bundle/cli.js" --version)
[ "${got}" = "${want}" ] || { echo "Version mismatch: got '${got}', want '${want}'." >&2; exit 1; }
echo "${want}" >"${prefix}/VERSION"
