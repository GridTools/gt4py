================
Review Checklist
================

Use this checklist to review pull requests. If the pull request is large,
consider giving partial feedback quickly rather than taking a long time to
review comprehensively.

Large PRs
---------
- Consider giving partial feedback. The following are in order of magnitude
  changes they will request, so rather than going down the list for one file,
  prefer making a full pass for each section.

Do not allow untested "functionality" or "fixes"
------------------------------------------------

- If there is new functionality, is there a unit or at least integration test
  for it as well?

- If this fixes any issues, is there a regression test for every issue fixed?

Check for obvious omissions
---------------------------
- Any comment mentioning that something is incomplete or could be improved etc
  should be addressed before merging or split out into an issue.

Check for code quality
----------------------

- Are there any coupling issues?

 + Are any interfaces circumvented (calling underlying methods instead of
   interface methods etc)?

 + Does any part of the code require knowledge of the inner workings of another
   part?

 + Do you see any violations of encapsulation (use of ``._protected``
   attributes of another class)?

- Are there functions or methods that are too complex?

 + many possible code paths (>3)? -> suggest splitting unless trivial

 + contain more than one loop? -> suggest splitting if appropriate (Ideally
   split loops into their own functions, which should be generators where
   possible)

 + Could complexity be reduced by using a different algorithm? Think DFS <->
   BFS, recursion <-> iteration, etc


- Are there loops that could be replaced by simple comprehensions or too
  complex comprehensions?

- Consider the overall design: could it be improved significantly? Might it
  block future improvements ins some way?

Check for compliance
--------------------

- New files should never be added to the exclude list in
   ``.pre-commit-config.yaml`` -> any legitimate issues should be addressed by
   adjusting the checker config or punctual ``# noqa # <reason>`` / ``# type:
   ignore # <reason>`` comments.

- Every single comment that punctually disables a code check needs an
  associated explaining comment directly above or behind

Check for docstrings
--------------------
In general we consider that a well named simple function with type annotations does not require a docstring.

- Consider new public classes / functions: they should have a docstring if
 + Their purpose is not obvious from the name (also consider renaming)
 + Their Body is complex
 + Someone might want to use them interactively (from a shell or notebook)

- Check existing docstrings: do they need to be expanded or updated?

Check for legacy compliance
---------------------------
Only appropriate for internal PRs

- Has at least one of the changed files been removed from the pre-commit
  exclude list?

- Are there missing type annotations in changed files? -> suggest locally
  running mypy with ``--disallow-untyped-defs --disallow-incomplete-defs``. If
  the file is large, suggest fixing only part of these

Check for sphinx docs
---------------------
- Anything user relevant that changes or is added should be reflected in the
  sphinx docs.
