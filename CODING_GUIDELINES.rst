==================
Coding Guidelines
==================

Code design
-----------
Before accepting changes we assess the design according to the following guidelines

0. Look at each piece of code and ask yourself: Would I point to this in a job
   interview as an example of well crafted code according to my best abilities?

1. If it isn't tested and verified, it isn't scientific. This is a scientific
   software library.

2. Separate concerns: any unit of code should have one concern and one only.
   The implementation details of any unit of code should not be known to any
   other unit of code.

3. Do not repeat yourself (DRY):
   `https://en.wikipedia.org/wiki/Don%27t_repeat_yourself`_. Always check if
   functionality exists elsewhere before implementing. Also recognize when
   multiple specific implementations of an algorithm can be replaced with a
   single generic one.

4. Do not reinvent the wheel: Instead of writing a framework to solve a problem
   consider existing solutions.

5. Names should state the intent of code objects. If the intent of a code
   expression **might** not be immediately obvious it deserves a name which
   makes it obvious.

Docstrings
----------
In general we consider that a well-named simple function with type annotations
does not require a docstring.  When a long-form docstring is appropriate,
use `NumPy format <https://developer.lsst.io/python/numpydoc.html>`__. Prefer
python type annotations over describing parameter / return types in the
docstring.

- Consider new public classes / functions: they should have a docstring if
 + Their purpose is not obvious from the name (also consider renaming)
 + Their body is complex
 + Someone might want to use them interactively (from a shell or notebook)

- Check existing docstrings: do they need to be expanded or updated?

- Check long-form docstrings: do they use the NumPy format?
 + Do they duplicate type annotations?

Code style
----------
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

- block comments should refer to the code following them and should be indented
  to the same level

Module structure
++++++++++++++++
Python modules should be structured in the following order:

1. Shebang line, ``#! /usr/bin/env python`` (only for executable scripts)

2. License header (``LICENSE_HEADER.txt``) and module-level comments

3. Module-level docstring

4. ``__all__ = [...]`` statement, if present

5. Imports (sorting and grouping automated by pre-commit hook)

6. Private module variables, functions and classes (names start with
   underscore)

7. Public module variables, functions and classes
