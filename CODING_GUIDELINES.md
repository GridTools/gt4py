# Coding Guidelines

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with some minor changes:

- [`pylint`](https://pylint.pycqa.org/) is not required. We use [`flake8`](https://flake8.pycqa.org/) with some plugins.
- We use [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://pycqa.github.io/isort/) for source code and imports formatting, which may break some of the guidelines in Section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstrings lines should still be limited to 79).
- _Power features_ (e.g. custom metaclasses, import hacks, reflection, etc.) should be avoided according to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), although the same point mentions that _"standard library classes internally using these features are allowed"_. Following the same spirit, we allow the use of power features in basic infrastructure library code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing) symbols from `typing` and `collections.abc` modules used in type annotations can be imported directly to keep common annotations concise and match standard typing practices. Following the same spirit, symbols from third-party or internal modules aiming to collect useful type definitions can also be imported directly.

------


- python file layout
- Tests best-practises: factory boy, pytest-cases, other pytest-plugins, make them parallelizable ??
- make sure you understand flake8 plugins (all the installed ones run)
- run pre-commit
- Napoleon google/numpy style docstrings (and use sections. Doctests are cool for small things but not a fully replacement of unit tests). Learn sphinx and RST, napoleon tags for beautiful but lightweight docstrins.
- import modules (except for typings)
- disable noqa: always with error code and explanation
- disable mypy: always with error code and explanation
- use tox for final tests and venv configuration
- in setting files: if somethins is in alphabetical (or any other)  order, keep it that way when adding items. If it's not, consider if it could help.
- Use RST instead of Markdown for docs ??


    + Add Google docstrings conventions
        + using only lightweight RST markups (links to other items and `literal` strings)
    + Add section for test best practices
        + factories and test input data (factory boy? pytest-cases?)
        + running tests in parallel: pytest-xdist[psutil] (`-n auto`) 
        + run only the tests that failed last time: `--lf / --last-failed` option.
        + run all the tests starting with the tests that failed last time: `--ff / --failed-first` option
        + use ` -l / --showlocalsflag` option to see the value of all local variables in tracebacks
    + Other topics
        + Comment CI ignores
        + In config files, try to keep sections and items within sections ordered logically. If there is not an evident logical order, just use alphabetical order




## Code design

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

4. Do not reinvent the wheel: instead of writing a framework to solve a problem
   consider existing solutions.

5. Names should state the intent of code objects. If the intent of a code
   expression **might** not be immediately obvious it deserves a name which
   makes it obvious.


## Docstrings

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


## Code style

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














## Development guidelines

### Code style

`Black <https://github.com/ambv/black>`__ code formatter should be
always used.

Additionally, general code style should comply with standard style
guidelines for Python programming such as
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__. 

In general, Python modules should be structured in the following order:

1. Shebang line, #! /usr/bin/env python (only for executable scripts)
2. License header (``LICENSE_HEADER.txt``) and module-level comments
3. Module-level docstring
4. ``__all__ = [...]`` statement, if present
5. Imports (alphabetically ordered within each block)

   a. Block of imports from the standard library
   b. Block of imports from general third party libraries (e.g. numpy,
      xarray)
   c. Block of imports from specific submodules of the project

6. Private module variables, functions and classes (names start with
   underscore)
7. Public module variables, functions and classes

General coding advices:

-  ``from X import Y`` import form is generally preferred over
   ``import X``
-  Absolute imports (``from library import something``) SHOULD be
   preferred over relative imports
   (``from .submodule import something``)
-  **is** and **is not** SHOULD be used when comparing to **None**
-  The **set** type SHOULD be used for unordered collections
-  **super()** MAY be used to call parent class methods
-  Iterators and generators SHOULD be used to iterate over large data
   sets efficiently
-  Block comments SHOULD reference the code following them and SHOULD be
   indented to the same level

### Tools

-  Use `Black: the uncoompromising Python code
   formatter <https://github.com/ambv/black>`__ with not more than 120
   characters per source line and 79 for docstrings

-  Follow NumPy format for docstrings with sphinx-Napoleon. Very useful
   guidelines can be found in
   `LSST <https://developer.lsst.io/python/numpydoc.html>`__ docstrings
   conventions

-  Git commit hooks with `pre-commit <https://pre-commit.com/>`__
   - runs formatting and compliance checks for you
   - will be run on all files at every pull request



=====

# Coding guidelines

- [Google style guidelines](https://google.github.io/styleguide/pyguide.html#s3.16.4-guidelines-derived-from-guidos-recommendations)
- [LSST DM style guide](https://developer.lsst.io/python/style.html)
- use from __future__ annotations
    + avoid ifTYPE_CHECKING
    + generics with builtins
- Naming style for typings
- python file layout
- Tests best-practises: factory boy, pytest-cases, other pytest-plugins, make them parallelizable ??
- make sure you understand flake8 plugins (all the installed ones run)
- run pre-commit
- Napoleon google/numpy style docstrings (and use sections. Doctests are cool for small things but not a fully replacement of unit tests). Learn sphinx and RST, napoleon tags for beautiful but lightweight docstrins.
- import modules (except for typings)
- disable noqa: always with error code and explanation
- disable mypy: always with error code and explanation
- use tox for final tests and venv configuration
- in setting files: if somethins is in alphabetical (or any other)  order, keep it that way when adding items. If it's not, consider if it could help.
- Use RST instead of Markdown for docs ??


    + Add mention and link to [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
    + Add Google docstrings conventions
        + using only lightweight RST markups (links to other items and `literal` strings)
    + Add section for test best practices
        + factories and test input data (factory boy? pytest-cases?)
        + running tests in parallel: pytest-xdist[psutil] (`-n auto`) 
        + run only the tests that failed last time: `--lf / --last-failed` option.
        + run all the tests starting with the tests that failed last time: `--ff / --failed-first` option
        + use ` -l / --showlocalsflag` option to see the value of all local variables in tracebacks
    + Other topics
        + Comment CI ignores
        + In config files, try to keep sections and items within sections ordered logically. If there is not an evident logical order, just use alphabetical order

- [ ] Quick overview presentation for development guidelines

