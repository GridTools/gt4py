#!/usr/bin/env bash
# Installs memdoor, a single static Go binary, pinned by release.pin
# (release tag and SHA-256).
#
# Usage: install.sh <prefix>
#
# Runs in the secretless 'prepare' job.

set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: install.sh <prefix>" >&2; exit 1; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prefix=$1

read -r tag sha < <(grep -v '^#' "${here}/release.pin" | head -n 1)
[[ "${tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid tag '${tag}' in release.pin." >&2; exit 1; }
mkdir -p "${prefix}/bin"
curl -fsSL --retry 3 -o "${prefix}/bin/memdoor" \
  "https://github.com/guregodevo/memdoor/releases/download/${tag}/memdoor-linux-amd64"
echo "${sha}  ${prefix}/bin/memdoor" | sha256sum -c --quiet -
chmod 0555 "${prefix}/bin/memdoor"

# A fresh HOME, so the version check writes nowhere that matters.
got=$(HOME=$(mktemp -d) "${prefix}/bin/memdoor" --version | head -n 1)
case "${got}" in "memdoor ${tag} "* | "memdoor ${tag}") ;; *) echo "Version mismatch: '${got}', want ${tag}." >&2; exit 1 ;; esac
echo "${tag}" >"${prefix}/VERSION"
