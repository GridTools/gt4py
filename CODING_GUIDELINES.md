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

We follow the [Google Python Style Guide][google-style-guide] with very few minor changes (mentioned below). Since the best way to remember something is to understand the reasons behind it, make sure you go through the style guide at least once, paying special to the explanations for the final decisions given in the _Pros_, _Cons_ and _Decision_ subsections.

We deviate from the [Google Python Style Guide][google-style-guide] only in the following points:

- [`pylint`][pylint] is not required. We use [`flake8`][flake8] with some plugins.
- We use [`black`][black] and [`isort`][isort] for source code and imports formatting, which may break some of the guidelines in Section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstrings lines should still be limited to 79).
- _Power features_ (e.g. custom metaclasses, import hacks, reflection, etc.) should be avoided according to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), although it is also mentioned that _"standard library classes internally using these features are allowed"_. Following the same spirit, we allow the use of power features in basic infrastructure library code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing) symbols from `typing` and `collections.abc` modules used in type annotations _can be imported directly to keep common annotations concise and match standard typing practices_. Following the same spirit, symbols can also be imported directly from third-party or internal modules which are just collections of type definitions.

### Docstrings

We generate the API documentation automatically from the docstrings using the [Sphinx][sphinx] generator together with some extensions like [Sphinx-autodoc][sphinx-autodoc] and [Sphinx-napoleon][sphinx-napoleon], which understands and enhances the docstrings conventions from the Google Python Style Guide. Checkout [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google) for a complete overview.

Sphinx supports [reStructuredText][sphinx-rest] (reST) markup language to add additional formatting options to the generated docs, however section [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) in the Google Python Style Guide does not define how to use markups in docstrings. Therefore, we decided to forbid the use of reST markup in docstrings except for:

   - Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).   
   - Very basic formatting markup to improve _readability_ of the generated documentation without obscuring the source docstring (e.g.: ``` ``literal`` ```  strings).
   
Regarding code examples in docstrings, we highly encourage to use [doctests](https://docs.python.org/3/library/doctest.html) format to automatically test they are in sync with the code.

### Ignoring QA errors

You may ocassionally need to disable checks from _quality assurance_  (QA) tools (e.g. linters, type checkers, etc.) on specific lines because the tool is not able to fully understand why that piece of code is needed. This is usually feasible inlining a special comment like `# type: ignore`. Note that you should **only** ignore QA errors when you fully understand its cause and it is not reasonable to fix it by rewriting the offending code in a different way. Additionally, you should add a brief comment to make sure anyone else reading the code will also understand what is happening there. For example:

   ```python
   f = lambda: 'empty'  # noqa: E731  # assign lambda expression for testing
   ```

### Module structure

In general, Python modules should be structured in the following way:

1. (Only for **executable scripts**) _shebang_ line: `#! /usr/bin/env python3`.
2. License header boilerplate (check `LICENSE_HEADER.txt`).
3. Module docstring.
4. Imports, alphabetically ordered within each block:
   1. Block of imports from the standard library.
   2. Block of imports from general third party libraries using standard shortcuts if when customary (e.g. `numpy as np`).
   3. Block of imports from specific modules of the project.
5. (Optional, mainly for re-exporting symbols) Definition of exported symbols:   
```python
__all__ = ["func_a", "CONST_B"]
   ```
6. Public constants and typings definitions.
7. Module contents organized in a meaningful way for reading the module, usually defining functions before classes.

In configuration files, try to keep sections and items within sections ordered logically, adding comments to make it explicit if needed. If there is not only one evident logical order, just pick your favorite one or use alphabetical order.


## Testing 
- Tests best-practises: factory boy, pytest-cases, other pytest-plugins, make them parallelizable ??

+ factories and test input data (factory boy? pytest-cases?)
+ running tests in parallel: pytest-xdist[psutil] (`-n auto`) 
+ run only the tests that failed last time: `--lf / --last-failed` option.
+ run all the tests starting with the tests that failed last time: `--ff / --failed-first` option
+ use ` -l / --showlocalsflag` option to see the value of all local variables in tracebacks
Napoleon google/numpy style docstrings (and use sections. Doctests are cool for small things but not a fully replacement of unit tests). Learn sphinx and RST, napoleon tags for beautiful but lightweight docstrins.



## Tools

We use several coding tools to help us increasing the quality or our code. New tools can be added, specially if they do not add a large overhead to our workflow and we think they bring extra benefits to keep our codebasde in shape. The most important ones we currently rely on are:

   - [Black: the uncoompromising Python code][black]
   - [flake8][flake8]
   - [pre-commit][pre-commit]
   - [sphinx][sphinx], with the following plugins
      + [sphinx-autodoc][sphinx-autodoc]
      + [sphinx-napoleon][sphinx-napoleon]



<!-- Reference links -->

[black]: https://black.readthedocs.io/en/stable/
[flake8]: https://flake8.pycqa.org/
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[isort]: https://pycqa.github.io/isort/
[pre-commit]: https://pre-commit.com/
[pylint]: https://pylint.pycqa.org/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html


