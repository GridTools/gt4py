==================
Codeing Guidelines
==================

In general, `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ is a good
starting point for coding style and should be applied where not superseeded
by auotmatic checks or below guidelines.

A selection of **general** recommendations from PEP8:

- prefer absolute imports (``from x import y``) over relative imports (``from
  .x import y``)

- prefer ``from a_module import AClass`` over ``import a_module``

- use ``X is None`` and ``Y is not None`` to compare to ``None``

- use ``set`` for unodered collections

- use iterators and generators to iterate over large data sets

- Block comments should refer to the code following them and should be indented
  to the same level

Module structure
----------------
Python modules should be structured in the following order:

1. Shebang line, #! /usr/bin/env python (only for executable scripts)

2. License header (``LICENSE_HEADER.txt``) and module-level comments

3. Module-level docstring

4. ``__all__ = [...]`` statement, if present

5. Imports (sorting and grouping automated by pre-commit hook)

6. Private module variables, functions and classes (names start with
   underscore)

7. Public module variables, functions and classes


Check for code quality
----------------------

- Are there any coupling issues?

 + Are any interfaces circumvented (e.g. calling underlying methods instead of
   interface methods)?

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
   BFS, recursion <-> iteration, etc.


- Are there loops that could be replaced by simple comprehensions or too
  complex comprehensions?

- Consider the overall design: could it be improved significantly? Might it
  block future improvements in some way?

Check for docstrings
--------------------
In general we consider that a well named simple function with type annotations
does not require a docstring.  When a long-form docstring is appropriate,
use `NumPy format <https://developer.lsst.io/python/numpydoc.html>`__. Prefer
python type annotations over describing parameter / return types in the
docstring.

- Consider new public classes / functions: they should have a docstring if
 + Their purpose is not obvious from the name (also consider renaming)
 + Their Body is complex
 + Someone might want to use them interactively (from a shell or notebook)

- Check existing docstrings: do they need to be expanded or updated?

- Check long-form docstrings: do they use the NumPy format?
 + Do they duplicate type annotations?
