#!/usr/bin/env bash
# Tests for bin/collect-result: only regular, single-link, size-capped files
# named review.md / meta.json may leave the sandbox workspace.

set -uo pipefail
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

collect() { /usr/bin/bash "${BIN_DIR}/collect-result" "$@"; }

setup_out() {
  mkdir -p "${TMP}/ws/out"
  echo "a review" >"${TMP}/ws/out/review.md"
  echo '{"status": "ok"}' >"${TMP}/ws/out/meta.json"
}

test_copies_expected_files() {
  setup_out
  echo "evil" >"${TMP}/ws/out/payload.sh"
  collect "${TMP}/ws/out" "${TMP}/dest" >/dev/null &&
    assert "review copied" cmp -s "${TMP}/ws/out/review.md" "${TMP}/dest/review.md" &&
    assert "meta copied" cmp -s "${TMP}/ws/out/meta.json" "${TMP}/dest/meta.json" &&
    assert "other files ignored" test ! -e "${TMP}/dest/payload.sh"
}

test_rejects_symlinked_file() {
  setup_out
  rm "${TMP}/ws/out/review.md"
  ln -s /etc/passwd "${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "not a regular file" log &&
    assert "symlink not followed" test ! -e "${TMP}/dest/review.md" &&
    assert "meta still copied" test -f "${TMP}/dest/meta.json"
}

test_rejects_symlink_to_proc_environ() {
  # A review.md symlinked at a process's environment must leak no env bytes.
  setup_out
  rm "${TMP}/ws/out/review.md"
  ln -s /proc/self/environ "${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "not a regular file" log &&
    assert "environ symlink not followed" test ! -e "${TMP}/dest/review.md"
}

test_rejects_symlinked_out_dir() {
  mkdir -p "${TMP}/elsewhere" "${TMP}/ws"
  echo "secret" >"${TMP}/elsewhere/review.md"
  ln -s "${TMP}/elsewhere" "${TMP}/ws/out"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "symlink" log &&
    assert "nothing collected" test ! -e "${TMP}/dest/review.md"
}

test_rejects_symlinked_parent() {
  mkdir -p "${TMP}/real/out"
  echo "x" >"${TMP}/real/out/review.md"
  ln -s "${TMP}/real" "${TMP}/ws"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "symlink" log &&
    assert "nothing collected" test ! -e "${TMP}/dest/review.md"
}

test_rejects_oversized_file() {
  setup_out
  head -c 70000 /dev/zero | tr '\0' 'x' >"${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "over the 65536-byte limit" log &&
    assert "oversized not copied" test ! -e "${TMP}/dest/review.md"
}

test_accepts_file_at_limit() {
  setup_out
  head -c 65536 /dev/zero | tr '\0' 'x' >"${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >/dev/null &&
    assert "copied at the limit" test "$(stat -c %s "${TMP}/dest/review.md")" -eq 65536
}

test_rejects_fifo_without_blocking() {
  setup_out
  rm "${TMP}/ws/out/review.md"
  mkfifo "${TMP}/ws/out/review.md"
  timeout 10 /usr/bin/bash "${BIN_DIR}/collect-result" "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "not a regular file" log &&
    assert "fifo not copied" test ! -e "${TMP}/dest/review.md"
}

test_rejects_directory() {
  setup_out
  rm "${TMP}/ws/out/review.md"
  mkdir "${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "not a regular file" log &&
    assert "directory not copied" test ! -e "${TMP}/dest/review.md"
}

test_rejects_hard_link() {
  setup_out
  echo "host file" >"${TMP}/hostfile"
  rm "${TMP}/ws/out/review.md"
  ln "${TMP}/hostfile" "${TMP}/ws/out/review.md"
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "hard links" log &&
    assert "hard link not copied" test ! -e "${TMP}/dest/review.md"
}

test_missing_out_dir_is_not_an_error() {
  collect "${TMP}/ws/out" "${TMP}/dest" >log && grep -q "missing" log &&
    assert "dest created and empty" test -z "$(ls -A "${TMP}/dest")"
}

test_ignores_planted_path_tools() {
  # A sandbox-writable directory first in PATH must not be used.
  setup_out
  mkdir -p "${TMP}/bin"
  printf '#!/bin/sh\ntouch %s/pwned\n' "${TMP}" >"${TMP}/bin/head"
  cp "${TMP}/bin/head" "${TMP}/bin/stat"
  chmod +x "${TMP}/bin/head" "${TMP}/bin/stat"
  PATH="${TMP}/bin:${PATH}" collect "${TMP}/ws/out" "${TMP}/dest" >/dev/null &&
    assert "planted tools not run" test ! -e "${TMP}/pwned" &&
    assert "review copied" test -f "${TMP}/dest/review.md"
}

for t in $(declare -F | awk '$3 ~ /^test_/ {print $3}'); do run_test "$t"; done
summary
