#!/usr/bin/env bash
# Tests for bin/publish-comment: sanitisation of the hostile review and the
# marker-based comment upsert (against a fake 'gh').

set -uo pipefail
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

PC="${BIN_DIR}/publish-comment"
ZWJ=$'\xe2\x80\x8d'
RUN_URL="https://github.com/egparedes/gt4py/actions/runs/1"
SHA=0123456789abcdef0123456789abcdef01234567

sanitize() { bash "${PC}" sanitize; }
lacks() { ! grep -qE -- "$1" <<<"$2"; } # pattern text
render() {
  bash "${PC}" render --harness stub --version stub-1 --model none --head-sha "${SHA}" \
    --run-url "${RUN_URL}" --review "$1"
}

test_neutralises_mentions() {
  local out
  out=$(printf 'Ping @egparedes and @GridTools/gt4py-core, mail a@b.org.\n' | sanitize) &&
    assert "no live mention left" lacks '@[A-Za-z0-9]' "${out}" &&
    assert "mention kept readable" grep -qF "@${ZWJ}egparedes" <<<"${out}"
}

test_escapes_character_references_to_at() {
  # GitHub renders all of these (with the ';') as live mentions.
  local out
  out=$(printf '&#64;egparedes &#x40;egparedes &#X40;a &#0064;b &commat;egparedes &amp;ok\n' | sanitize) &&
    assert "no numeric reference left" lacks '&#' "${out}" &&
    assert "no &commat; left" lacks '&commat;' "${out}" &&
    assert "escaped readably" grep -qF '&amp;#64;egparedes' <<<"${out}" &&
    assert "other entities untouched" grep -qF ' &amp;ok' <<<"${out}"
}

test_breaks_comment_delimiters() {
  local out
  out=$(printf '<!-- agent-review:stub -->\n<!<!---- nested -->\nhidden -->\n<!--->\n' | sanitize) &&
    assert "no comment start" lacks '<!--' "${out}" &&
    assert "no comment end" lacks '-->' "${out}"
}

test_rejects_invalid_utf8() {
  local rc=0
  printf 'bad \xff\xfe bytes\n' | sanitize >/dev/null 2>&1 || rc=$?
  assert "exit code 3 on invalid UTF-8" test "${rc}" -eq 3
}

test_drops_control_characters() {
  local out
  out=$(printf 'a\rb\x1b[31mc\x07d\te\n' | sanitize) &&
    assert "controls removed, tab kept" test "${out}" = $'ab[31mcd\te'
}

test_caps_size_on_character_boundary() {
  # 'é' is two bytes; 30001 of them cross the 60000-byte cap mid-character.
  local n
  for _ in $(seq 30001); do printf 'é'; done >big.md
  sanitize <big.md >out.md &&
    n=$(stat -c %s out.md) &&
    assert "capped at 60000 bytes (got ${n})" test "${n}" -le 60000 &&
    assert "valid UTF-8" iconv -f UTF-8 -t UTF-8 out.md -o /dev/null
}

test_render_marker_header_and_limit() {
  local first comments
  {
    echo "### Summary"
    echo "@egparedes <!-- agent-review:stub -->"
    head -c 200000 /dev/zero | tr '\0' 'x'
  } >review.md
  render review.md >body.md &&
    first=$(head -n 1 body.md) &&
    comments=$(grep -o -- "<!--" body.md | wc -l) &&
    assert "marker on the first line" test "${first}" = "<!-- agent-review:stub -->" &&
    assert "only the marker is a comment (${comments})" test "${comments}" -eq 1 &&
    assert "under GitHub's limit" test "$(stat -c %s body.md)" -le 65536 &&
    assert "truncation noted" grep -q "Truncated to 60000 bytes" body.md &&
    assert "advisory header" grep -q "Advisory review by \*\*stub\*\* stub-1" body.md &&
    assert "header names the reviewed commit" grep -qF "of commit ${SHA}," body.md
}

test_render_rejects_invalid_head_sha() {
  echo "review" >review.md
  ! bash "${PC}" render --harness stub --version stub-1 --model none --head-sha 'main' \
    --run-url "${RUN_URL}" --review review.md >/dev/null 2>&1
}

test_render_rejects_markdown_in_labels() {
  echo "review" >review.md
  ! bash "${PC}" render --harness stub --version '1 [x](https://evil)' --model none --head-sha "${SHA}" \
    --run-url "${RUN_URL}" --review review.md >/dev/null 2>&1
}

test_render_rejects_empty_review() {
  local rc=0
  printf '  \n\n' >review.md
  render review.md >/dev/null 2>&1 || rc=$?
  assert "exit code 3 on empty review" test "${rc}" -eq 3
}

# A fake 'gh' that serves a fixed comment list and logs write requests.
install_fake_gh() { # comments-tsv
  mkdir -p bin
  printf '%s' "$1" >comments.tsv
  cat >bin/gh <<'GH'
#!/usr/bin/env bash
case "$*" in
  "api --paginate "*) [ ! -e "${FAKE_DIR}/list-fails" ] || exit 1; cat "${FAKE_DIR}/comments.tsv" ;;
  "api -X "*) echo "$3 $4" >>"${FAKE_DIR}/requests.log"; cat >>"${FAKE_DIR}/requests.log"; echo "https://example/c" ;;
  *) echo "unexpected gh call: $*" >&2; exit 1 ;;
esac
GH
  chmod +x bin/gh
}

post() {
  echo "<!-- agent-review:stub -->" >body.md
  echo "text" >>body.md
  FAKE_DIR="${TMP}" PATH="${TMP}/bin:${PATH}" \
    bash "${PC}" post --repo egparedes/gt4py --pr 9 --harness stub --body-file body.md >/dev/null
}

test_post_updates_own_marked_comment() {
  install_fake_gh "$(printf '%s\n' \
    $'11\tegparedes\t<!-- agent-review:stub -->' \
    $'12\tgithub-actions[bot]\t## PR Reviewer Guide' \
    $'13\tgithub-actions[bot]\tquoted <!-- agent-review:stub -->' \
    $'14\tgithub-actions[bot]\t<!-- agent-review:opencode -->' \
    $'15\tgithub-actions[bot]\t<!-- agent-review:stub -->')"
  post &&
    assert "PATCHed comment 15" grep -q "^PATCH repos/egparedes/gt4py/issues/comments/15$" requests.log
}

test_post_creates_comment_without_marker_match() {
  install_fake_gh "$(printf '%s\n' \
    $'11\tegparedes\t<!-- agent-review:stub -->' \
    $'12\tgithub-actions[bot]\t## PR Reviewer Guide')"
  post &&
    assert "POSTed a new comment" grep -q "^POST repos/egparedes/gt4py/issues/9/comments$" requests.log &&
    assert "body sent as JSON" grep -qF '"body":"<!-- agent-review:stub -->\ntext\n"' requests.log
}

test_post_fails_without_posting_when_listing_fails() {
  install_fake_gh $'15\tgithub-actions[bot]\t<!-- agent-review:stub -->'
  touch list-fails
  ! post 2>/dev/null &&
    assert "no write request" test ! -e requests.log
}

test_post_refuses_body_without_marker() {
  install_fake_gh ""
  echo "no marker" >body.md
  ! FAKE_DIR="${TMP}" PATH="${TMP}/bin:${PATH}" \
    bash "${PC}" post --repo egparedes/gt4py --pr 9 --harness stub --body-file body.md >/dev/null 2>&1
}

for t in $(declare -F | awk '$3 ~ /^test_/ {print $3}'); do run_test "$t"; done
summary
