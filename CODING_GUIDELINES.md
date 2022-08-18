# Coding Guidelines


## Code design

Before accepting changes we assess the design according to the following guidelines:

1. Look at each piece of code and ask yourself: _Would I point to this in a job interview as an example of well crafted code according to my best abilities?_
2. If it isn't tested and verified, it isn't scientific. This is a scientifi software library.
3. Separate concerns: any unit of code should have one concern and one only. The implementation details of any unit of code should not be known to any other unit of code.
4. Do not repeat yourself ([DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)): always check if functionality already exists elsewhere in the project before implementing it. Also recognize when multiple specific implementations of an algorithm can be replaced with a single generic one.
5. Do not reinvent the wheel: instead of writing a new component to solve a problem consider existing solutions.
6. Names should state the intent of code objects. If the intent of a code expression **might** not be immediately obvious it deserves a name which makes it obvious.


## Code Style

We follow the [Google Python Style Guide][google-style-guide] with very few minor changes (mentioned below). The best way to remember something is to understand the reasons behind it, therefore make sure you go through the style guide at least once, paying special to the explanations for each decisions given in the _Pros_, _Cons_ and _Decision_ subsections.

Deviations from the [Google Python Style Guide][google-style-guide]:

- [`pylint`][pylint] is not required. We use [`flake8`][flake8] with some plugins.
- We use [`black`][black] and [`isort`][isort] for source code and imports formatting, which may break some of the guidelines in Section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstrings lines should still be limited to 79).
- _Power features_ (e.g. custom metaclasses, import hacks, reflection, etc.) should be avoided according to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), although the same point mentions that _"standard library classes internally using these features are allowed"_. Following the same spirit, we allow the use of power features in basic infrastructure library code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing) symbols from `typing` and `collections.abc` modules used in type annotations can be imported directly to keep common annotations concise and match standard typing practices. Following the same spirit, symbols can also be imported directly from third-party or internal modules which are just collections of type definitions.

### Docstrings

We generate the API documentation automatically from the docstrings using [Sphinx][sphinx] documentation generator together with some extensions like [Sphinx-autodoc][sphinx-autodoc] and [Sphinx-napoleon][sphinx-napoleon], which understands the docstrings conventions from the Google Python Style Guide (checkout [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)).

Sphinx supports [reStructuredText][sphinx-rest] (reST) markup language to add additional formatting options to the generated docs but the [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) section of the Google Python Style Guide does not define how to use markups in the docstrings. Therefore, we decided to forbid the use of reST markup in docstrings except for:

   - Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).   
   - Very basic formatting markup to improve readability of the generated documentation without obscuring the source docstring (e.g.: ``` ``literal`` ```  strings).
   

Napoleon google/numpy style docstrings (and use sections. Doctests are cool for small things but not a fully replacement of unit tests). Learn sphinx and RST, napoleon tags for beautiful but lightweight docstrins.

### Ignores
+ Comment CI ignores:  - make sure you understand flake8 plugins (all the installed ones run) always with error code and explanation
+ In config files, try to keep sections and items within sections ordered logically. If there is not an evident logical order, just use alphabetical order. - in setting files: if somethins is in alphabetical (or any other)  order, keep it that way when adding items. If it's not, consider if it could help.


### Module structure

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


Python modules should be structured in the following order:

1. Shebang line, ``#! /usr/bin/env python`` (only for executable scripts)

2. License header (``LICENSE_HEADER.txt``) and module-level comments

3. Module-level docstring

4. ``__all__ = [...]`` statement, if present

5. Imports (sorting and grouping automated by pre-commit hook)

6. Private module variables, functions and classes (names start with
   underscore)

7. Public module variables, functions and classes



## Testing 
- Tests best-practises: factory boy, pytest-cases, other pytest-plugins, make them parallelizable ??

+ factories and test input data (factory boy? pytest-cases?)
+ running tests in parallel: pytest-xdist[psutil] (`-n auto`) 
+ run only the tests that failed last time: `--lf / --last-failed` option.
+ run all the tests starting with the tests that failed last time: `--ff / --failed-first` option
+ use ` -l / --showlocalsflag` option to see the value of all local variables in tracebacks


## Tools

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


<!-- Reference links -->

[black]: https://black.readthedocs.io/en/stable/
[flake8]: https://flake8.pycqa.org/
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[isort]: https://pycqa.github.io/isort/
[pylint]: https://pylint.pycqa.org/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html