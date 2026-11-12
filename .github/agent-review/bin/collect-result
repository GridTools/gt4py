#!/usr/bin/bash
# Copies the review out of the sandbox workspace; runs on the TAINTED host.
#
# Usage: collect-result <out-dir> <dest-dir>
#
# After 'awf' returns, everything the sandbox could write is attacker-
# controlled: <out-dir>, its parents, and the home and /tmp trees. So this
# script uses only absolute paths to /usr/bin tools (/usr is read-only in the
# sandbox), must be called with a fixed PATH, and copies only the expected
# files, only if each is a regular file with a single link, and at most
# MAX_BYTES of it. Everything else the agent wrote is ignored. The files are
# validated again in the 'publish' job, which never trusts this copy either.

set -euo pipefail
export PATH=/usr/bin:/bin LC_ALL=C

readonly MAX_BYTES=65536
readonly EXPECTED=(review.md meta.json)

warn() { echo "::warning title=Agent review result::$*"; }

[ $# -eq 2 ] || { echo "Usage: collect-result <out-dir> <dest-dir>" >&2; exit 1; }
src=$1
dest=$2

/usr/bin/mkdir -p -- "${dest}"

# Any symlink in the path (including <out-dir> itself or a parent replaced by
# the agent) would make the canonical path differ.
if ! canonical=$(/usr/bin/realpath -e -- "${src}" 2>/dev/null) || [ "${canonical}" != "${src}" ] || [ ! -d "${src}" ]; then
  warn "'${src}' is missing, not a directory, or reached through a symlink; nothing collected."
  exit 0
fi

for name in "${EXPECTED[@]}"; do
  path="${src}/${name}"
  if [ -L "${path}" ] || [ ! -f "${path}" ]; then
    warn "'${name}' is missing or not a regular file; skipped."
    continue
  fi
  # A hard link could expose a host file the agent cannot read itself.
  read -r links size < <(/usr/bin/stat -c '%h %s' -- "${path}")
  if [ "${links}" != 1 ]; then
    warn "'${name}' has ${links} hard links; skipped."
    continue
  fi
  if [ "${size}" -gt "${MAX_BYTES}" ]; then
    warn "'${name}' is ${size} bytes, over the ${MAX_BYTES}-byte limit; skipped."
    continue
  fi
  # Bounded read: the size check above is advisory, this is the limit.
  /usr/bin/head -c "$((MAX_BYTES + 1))" -- "${path}" >"${dest}/${name}.tmp"
  if [ "$(/usr/bin/stat -c %s -- "${dest}/${name}.tmp")" -gt "${MAX_BYTES}" ]; then
    /usr/bin/rm -f -- "${dest}/${name}.tmp"
    warn "'${name}' grew over the ${MAX_BYTES}-byte limit while reading; skipped."
    continue
  fi
  /usr/bin/mv -f -- "${dest}/${name}.tmp" "${dest}/${name}"
  echo "Collected '${name}' (${size} bytes)."
done
