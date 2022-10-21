# Coding Guidelines


## Software design

Designing a clean architecture for a software product is a difficult task and developers can easily overlook it under pressure or when they are eager to deliver exciting new features. However, choosing the quick solution instead of a clean one will likely lead to a large amount of additional work in the future (_technical debt_).

To keep technical debt at acceptable levels, design and implementation of new features should follow best practices:

1. Make sure your code is always covered by automatic testing to ensure its correctness. Where unit tests are impractical, use integration tests.
2. Adhere to the [SOLID](https://en.wikipedia.org/wiki/SOLID) principles of software design.
3. Do not reinvent the wheel: if someone solved your problem within the project or in a third party library, consider using their solution or extending it before writing a new component for the same purpose.
4. _You ain't gonna need it_ ([YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)): do not design solutions for problems that might come up in the future, as chances are that you will never need that code. Focus on current problems and prepare for future requirements by writing clean code.
5. _Do not repeat yourself_ ([DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)): if you are writing the same code snippet in several places, extract it into a function.
6. Use meaningful names: the purpose of an object should be clear from its name. Usually, class names are nouns and function names are verbs.

Remember that important design decisions should be properly documented for future reference and to share the knowledge with other developers. We decided to use lightweight _Architecture Decision Records_ (ADRs) for this purpose. The full list of ADRs and documentation for writing new ones can be found in [docs/functional/architecture/Index.md](docs/functional/architecture/Index.md).


## Code Style

We follow the [Google Python Style Guide][google-style-guide] with a few minor changes (mentioned below). Since the best way to remember something is to understand the reasons behind it, make sure you go through the style guide at least once, paying special attention to the discussions in the _Pros_, _Cons_, and _Decision_ subsections.

We deviate from the [Google Python Style Guide][google-style-guide] only in the following points:

- We use [`flake8`][flake8] with some plugins instead of [`pylint`][pylint].
- We use [`black`][black] and [`isort`][isort] for source code and imports formatting, which may work differently than indicated by the guidelines in section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstring lines should still be limited to 79).
- According to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), direct use of _power features_ (e.g. custom metaclasses, import hacks, reflection) should be avoided, but standard library classes that internally use these power features are accepted. Following the same spirit, we allow the use of power features in infrastructure code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing), symbols from `typing` and `collections.abc` modules used in type annotations _"can be imported directly to keep common annotations concise and match standard typing practices"_. Following the same spirit, we allow symbols to be imported directly from third-party or internal modules when they only contain a collection of frequently used typying definitions.

### Docstrings

We generate the API documentation automatically from the docstrings using [Sphinx][sphinx] and some extensions such as [Sphinx-autodoc][sphinx-autodoc] and [Sphinx-napoleon][sphinx-napoleon]. These follow the Google Python Style Guide docstring conventions to automatically format the generated documentation. A complete overview can be found here: [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google).

Sphinx supports the [reStructuredText][sphinx-rest] (reST) markup language for defining additional formatting options in the generated documentation, however section [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) of the Google Python Style Guide does not specify how to use markups in docstrings. As a result, we decided to forbid reST markup in docstrings, except for the following cases:

   - Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).   
   - Very basic formatting markup to improve _readability_ of the generated documentation without obscuring the source docstring (e.g. ``` ``literal`` ```  strings, bulleted lists).
   
We highly encourage the [doctest][doctest] format for code examples in docstrings. In fact, doctest runs code examples and makes sure they are in sync with the codebase.

### Module structure

In general, you should structure new Python modules in the following way:

1. _shebang_ line: `#! /usr/bin/env python3` (only for **executable scripts**!).
2. License header (see `LICENSE_HEADER.txt`).
3. Module docstring.
4. Imports, alphabetically ordered within each block (fixed automatically by `isort`):
   1. Block of imports from the standard library.
   2. Block of imports from general third party libraries using standard shortcuts when customary (e.g. `numpy as np`).
   3. Block of imports from specific modules of the project.
5. Definition of exported symbols (optional, mainly for re-exporting symbols from other modules):
```python
__all__ = ["func_a", "CONST_B"]
   ```
6. Public constants and typing definitions.
7. Module contents organized in a convenient way for understanding how the pieces of code fit together, usually defining functions before classes.

Try to keep sections and items logically ordered, add section separator comments to make section boundaries explicit when needed. If there is not a single evident logical order, pick the order you consider best or use alphabetical order.

Consider configuration files as another type of source code and apply the same criteria, using comments when possible for better readability.

### Ignoring QA errors

You may occasionally need to disable checks from _quality assurance_  (QA) tools (e.g. linters, type checkers, etc.) on specific lines as some tool might not be able to fully understand why a certain piece of code is needed. This is usually done with special comments, e.g. `# type: ignore`. However, you should **only** ignore QA errors when you fully understand their source and rewriting your code to pass QA checks would make it less readable. Additionally, you should add a brief comment for future reference, e.g.:

   ```python
   f = lambda: 'empty'  # noqa: E731  # assign lambda expression for testing
   ```

## Testing 

Testing components is a critical part of a software development project. We follow standard practices in software development and write unit, integration, and regression tests. Note that even though [doctests][doctest] are great for documentation purposes, they lack many features and are difficult to debug. Hence, they should not be used as replacement for proper unit tests except in trivial cases.
 
TODO: add missing test conventions.
<!--
TODO: add test conventions:
TODO:    - to organize tests inside the `tests/` folder
TODO:    - to name tests
TODO:    - to use pytest features (fixtures, markers, etc.)
TODO:    - to generate mock objects and data for tests (e.g. pytest-factoryboy, pytest-cases)
TODO:    - to use pytest plugins 

Links with plugins:
https://towardsdatascience.com/pytest-plugins-to-love-%EF%B8%8F-9c71635fbe22
https://testandcode.com/116
-->

<!-- Reference links -->

[black]: https://black.readthedocs.io/en/stable/
[doctest]: https://docs.python.org/3/library/doctest.html
[flake8]: https://flake8.pycqa.org/
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[isort]: https://pycqa.github.io/isort/
[pre-commit]: https://pre-commit.com/
[pylint]: https://pylint.pycqa.org/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

