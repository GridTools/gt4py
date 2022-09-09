# Coding Guidelines


## Software design

Designing a clean architecture for a software product is a difficult task, and developers can easily overlook it under the pressure of deadlines or when they are eager to deliver exciting new features. However, choosing the quick solution instead of the clean one will likely lead to a large amount of additional work in the future (_technical debt_).

To keep technical debt at acceptable levels, follow best practices when designing and implementing new features:

1. Make sure all your code is covered by automatic testing to ensure its correctness. Where unit tests are impractical, use integration tests.
2. Adhere to the [SOLID](https://en.wikipedia.org/wiki/SOLID) principles of software design.
3. Do not reinvent the wheel: if someone else solved your problem within the project or in a third party library, consider using it or extending it before writing a new component for basically the same purpose.
4. _You ain't gonna need it_ ([YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)): don't design solutions for problems that might come up in the future, chances are you will never need that code. Focus on the current problems and prepare for future requirements by writing clean code.
5. _Do not repeat yourself_ ([DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)): if you are writing down the same code pattern in several places you should extract it into a function.
6. Use meaningful names: the purpose of an object should be clear from its name. Class names are usually nouns and function names are verbs.

Remember that important design decisions should be documented as reference for the future and to share the knowledge with all the developers. We decided to use lightweight _Architecture Decision Records_ (ADRs) for this purpose. The whole list of ADRs and documentation about when and how to write new ones can be found in [docs/functional/architecture/Index.md](docs/functional/architecture/Index.md).


## Code Style

We follow the [Google Python Style Guide][google-style-guide] with very few minor changes (mentioned below). Since the best way to remember something is to understand the reasons behind it, make sure you go through the style guide at least once, paying special attention to the discussions in the _Pros_, _Cons_ and _Decision_ subsections.

We deviate from the [Google Python Style Guide][google-style-guide] only in the following points:

- [`pylint`][pylint] is not required. We use [`flake8`][flake8] with some plugins.
- We use [`black`][black] and [`isort`][isort] for source code and imports formatting, which may work differently than indicated by the guidelines in section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstring lines should still be limited to 79).
- According to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), direct use of _power features_ (e.g. custom metaclasses, import hacks, reflection) should be avoided, but standard library classes that use these power features internally are accepted. Following the same spirit, we allow the use of power features in infrastructure code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing) symbols from `typing` and `collections.abc` modules used in type annotations _"can be imported directly to keep common annotations concise and match standard typing practices"_. Following the same spirit, symbols can also be imported directly from third-party or internal modules which are just collections of type definitions.

### Docstrings

We generate the API documentation automatically from the docstrings using [Sphinx][sphinx] together with some extensions like [Sphinx-autodoc][sphinx-autodoc] and [Sphinx-napoleon][sphinx-napoleon], which follow the Google Python Style Guide docstring conventions to automatically format the generated documentation. Check out the [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google) for a complete overview.

Sphinx supports the [reStructuredText][sphinx-rest] (reST) markup language to add additional formatting options to the generated docs, however section [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) of the Google Python Style Guide does not define how to use markups in docstrings. As a result, we decided to forbid the use of reST markup in docstrings, except for the following cases:

   - Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).   
   - Very basic formatting markup to improve _readability_ of the generated documentation without obscuring the source docstring (e.g. ``` ``literal`` ```  strings, bulleted lists).
   
Regarding code examples in docstrings, we highly encourage using the [doctest][doctest] format. This way, doctest runs your code examples and makes sure they are in sync with the codebase.

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

Try to keep sections and items within sections ordered logically, add section separator comments to make section boundaries explicit when needed. If there is not one a single evident logical order, just pick the order you consider best or use alphabetical order.

Consider configuration files as another kind of source code and apply the same criteria to keep them structured in a logical order, using comments when possible for the ease of reading.

### Ignoring QA errors

You may ocassionally need to disable checks from _quality assurance_  (QA) tools (e.g. linters, type checkers, etc.) on specific lines because the tool is not able to fully understand why that piece of code is needed. This is usually done by a special comment like `# type: ignore`. You should **only** ignore QA errors when you fully understand its cause and rewriting it to pass QA checks would actually make it less readable. Additionally, you should add a brief comment to make sure anyone else reading the code also understands what is happening there. For example:

   ```python
   f = lambda: 'empty'  # noqa: E731  # assign lambda expression for testing
   ```

## Testing 

Testing components is a critical part of a software development project. We follow standard practices in software development and write unit, integration and regression tests. Note that even though [doctests][doctest] are great for documentation purposes, they lack many features and are difficult to debug. Hence, they should not be used as replacement for proper unit tests except for trivial cases.
 
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

