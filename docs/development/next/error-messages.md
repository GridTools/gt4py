# GT4Py DSL Error Messages — Authoring Guide

How to write user-facing DSL diagnostics (`DSLError` and its subclasses) in
`gt4py.next`. For the base error-message style (sentences, single-quoted code
objects, terseness) see
[`CODING_GUIDELINES.md`](../../../CODING_GUIDELINES.md#error-messages); this
guide covers the structured-diagnostic layer on top.

## Internal error vs. DSL diagnostic — pick the right tool

- **Internal / library error** (a precondition an end user can't reach, a bug,
  a misuse of an internal API) → raise a builtin (`ValueError`, `TypeError`,
  …).
- **DSL diagnostic** (the user wrote a field operator / program that GT4Py
  rejects) → raise `DSLError` or a subclass with a `SourceLocation` and, where
  you have it, structured payload (label, related spans, notes, hints). The
  user is learning the DSL's rules, not debugging GT4Py.

A `DSLError` without a usable `location` is a smell — the user gets a message
with no pointer into their code. Thread a `SourceLocation` through.

## What a good diagnostic does

1. **Carries structured data, not strings.** Put the message, primary span,
   caret label, related spans, notes, and hints in their `DSLError` fields and
   let the renderer lay them out; don't bake spans or hints into the message
   string. Rendering is a single-owner concern (see
   [Architecture reference](#architecture-reference)).
2. **Points at the smallest offending span.** Put the caret under the part
   that is wrong; describe other contributing code (e.g. the other operand of
   a type mismatch) as a `related` span, not in the headline.
3. **Says what to do, not only what is wrong.** Give a `Hint:` that names the
   supported alternative (`where(...)`, `astype(...)`, `scan_operator`,
   `&` / `|`).
4. **Keeps compiler internals out of the headline.** Say `'while' loop`, not
   `ast.While`; raw internal names are only the last-resort fallback for
   constructs that are not catalogued yet.
5. **Uses the familiar Python header.** The renderer keeps the
   `File "...", line N` line that terminals and IDEs linkify and adds a
   line-number gutter and carets beneath it — don't invent a new header.
6. **Is pinned by a test.** Every diagnostic has a bad program plus an
   assertion on its rendered text in
   `tests/next_tests/unit_tests/ffront_tests/test_diagnostic_messages.py`. An
   unpinned diagnostic rots.

## The `DSLError` data model

`DSLError` (in `gt4py.next.errors.exceptions`) is the single user-facing error
type — both the exception that flows through `raise` and the structured
diagnostic the renderer consumes. `DSLError(location, message)` still works;
the structured fields are all optional keyword arguments:

| field      | kind              | purpose                                                             |
| ---------- | ----------------- | ------------------------------------------------------------------- |
| `location` | `SourceLocation`  | primary span; where the carets go                                   |
| `message`  | `str`             | the headline sentence (`super().__init__(message)`)                 |
| `label`    | `str`             | short fragment printed right after the carets ("this has type '…'") |
| `related`  | `(loc, str)` list | secondary labeled spans (e.g. the *other* operand of a mismatch)    |
| `notes`    | `str` list        | `Note:` — facts, *why* it's an error                                |
| `hints`    | `str` list        | `Hint:` — commands, *what to do instead*                            |
| `code`     | `ClassVar[str]`   | stable category slug (`"undefined-symbol"`), set per subclass       |

Keep `notes` and `hints` distinct: a note states a fact ("GT4Py does not
implicitly convert between datatypes."), a hint gives an action ("Convert one
operand explicitly, e.g. 'astype(<expr>, float64)'."). Set `code` on the
subclass, not per call site — the category belongs to the error class and
gives tooling and documentation a stable handle.

Build the message in the subclass so call sites pass meaning, not prose:
`UndefinedSymbolError(loc, name, candidates=...)` computes its own
"Did you mean …?" hint (via `difflib.get_close_matches`); the caller only
supplies the candidate set it has at hand.

## How to add or improve a diagnostic

Pick the row that matches what you're doing.

### 1. A newly rejected Python construct

Add one entry to `_UNSUPPORTED_FEATURE_HINTS` in `ffront/dialect_parser.py`,
keyed by the `ast` node type:

```python
ast.Match: ("'match' statement", ("Use 'if'/'elif' chains or 'where' instead.",)),
```

Name the construct as the *user* spells it (`"'match' statement"`, not
`ast.Match`) and give one actionable hint naming the closest supported
alternative. `DialectParser.generic_visit` consults the table and falls back
to the qualified `ast` class name for unlisted nodes, so the table grows
incrementally and nothing regresses when CPython adds node types. Every entry
also gets the uniform note "Only a subset of Python is valid inside GT4Py
functions." Adding a construct is one dict entry plus one test — no renderer or
exception changes.

### 2. A richer message at an existing `raise errors.DSLError(...)`

Add `label=`, `related=`, `notes=`, `hints=` keyword arguments. Do **not**
encode that information into the message string — the renderer places each
field deliberately (label after the carets, related spans as their own
underline rows, notes/hints as wrapped trailing lines).

### 3. A new error category

Subclass `DSLError`, set `code`, and build the message and hints in
`__init__` from semantic arguments (as `UndefinedSymbolError` does). Export it
from `errors/__init__.py`.

### 4. Late context from a later toolchain stage

Attach context to an in-flight error without rewriting its message, using the
standard `add_note` API (PEP 678):

```python
try:
    foast_node = FieldOperatorTypeDeduction.apply(untyped_foast_node)
except errors.DSLError as err:
    err.add_note(f"While processing the definition of '{name}'.")
    raise
```

`DSLError.add_note` overrides `BaseException.add_note` to route the note into
the structured `notes` field instead of `__notes__`: the traceback machinery
(and therefore pytest and IPython/Jupyter) prints the exception via
`str(err)`, which already renders the structured notes, so writing `__notes__`
as well would duplicate them. The seam is wired at `func_to_foast`
(`ffront/func_to_foast.py`); add it at later stages as they gain useful
context.

### 5. Always: a test

Add a bad program plus an assertion on the rendered text in
`test_diagnostic_messages.py`. The test is the spec; it is what keeps message
quality from regressing.

## Style

Beyond the base style in
[`CODING_GUIDELINES.md`](../../../CODING_GUIDELINES.md#error-messages):

- **Labels are sentence fragments** — they continue the caret line, so no
  trailing period ("not defined at this point", "this has type 'bool'").
  Messages, notes, and hints stay full sentences.
- Keep the headline to one sentence; push the *why* into a `Note:` and the
  *fix* into a `Hint:` rather than growing the headline.

## Examples

Rendered output (`str(err)`). Diagnostics raised through `@field_operator`
also carry the `While processing the definition of '<name>'.` note from the
toolchain seam, omitted here for brevity.

**Undefined symbol** (`UndefinedSymbolError` with a candidate set):

```
Undeclared symbol 'tmp_feild'.
  File "/tmp/demo.py", line 11
    11 |         return tmp_feild
       |                ^^^^^^^^^ not defined at this point
  Hint: Did you mean 'tmp_field'?
```

**Unsupported construct** (a catalogue entry):

```
Unsupported Python syntax: 'while' loop.
  File "/tmp/demo.py", line 21
    21 |         while True:
       |         ^^^^^^^^^^^
  Note: Only a subset of Python is valid inside GT4Py functions.
  Hint: GT4Py functions describe operations on whole fields without explicit loops. For
    sequential dependencies along a dimension, use a 'scan_operator'.
```

**Arithmetic with a boolean mask** (`label` on the bad operand, `related` on
the other, plus a `hint`):

```
Unsupported operand type(s) for +: 'Field[[IDim], float64]' and 'Field[[IDim], bool]'.
  File "/tmp/demo.py", line 44
    44 |         return a + mask
       |                    ^^^^ '+' expects arithmetic operands, but this has type 'Field[[IDim], bool]'
       |                - the other operand has type 'Field[[IDim], float64]'
  Hint: To select values based on a boolean mask, use 'where(mask, a, b)'. To compute
    with a boolean field, convert it explicitly, e.g. 'astype(mask, int32)'.
```

## Architecture reference

- **Rendering has one owner:** `format_diagnostic_parts` in
  `errors/formatting.py`, to which both `DSLError.__str__` and the excepthook
  delegate. It draws the line-number gutter and carets, merges a same-line
  `related` span into the primary snippet as a `-` underline row, caps
  multi-line spans at three source lines, wraps notes/hints at 88 columns, and
  falls back to the bare `File "...", line N` header when source text is
  unavailable (REPL, garbage-collected notebook cell) — never crashing, never
  showing a wrong snippet. New presentation behavior belongs here, not at call
  sites.
- **The unsupported-subset catalogue** is `_UNSUPPORTED_FEATURE_HINTS` in
  `ffront/dialect_parser.py` (see recipe step 1).

## Python-version caveat

The supported floor is Python 3.10, so the diagnostics code carries a few
forward-compat shims; respect them:

- Import `Self` from `gt4py.eve.extended_typing`, not `typing` (3.11+ only).
- `DSLError.add_note` works on every Python because `DSLError` defines it;
  don't rely on `add_note` for *other* `GT4PyError`s — it is a builtin only on
  3.11+.
- The catalogue must not reference `ast` nodes added after 3.10 (e.g.
  `ast.TryStar`) unconditionally — that breaks import on 3.10.

These spots are flagged with `TODO(havogt)`.
