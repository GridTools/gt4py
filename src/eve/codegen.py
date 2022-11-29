# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tools for source code generation."""


from __future__ import annotations

import abc
import collections.abc
import contextlib
import inspect
import os
import re
import string
import subprocess
import sys
import textwrap
import types

import black
import jinja2
from mako import template as mako_tpl  # type: ignore[import]

from . import exceptions, utils
from .concepts import CollectionNode, LeafNode, Node, RootNode
from .extended_typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)
from .visitors import NodeVisitor


SourceFormatter = Callable[[str], str]

SOURCE_FORMATTERS: Dict[str, SourceFormatter] = {}
"""Global dict storing registered formatters."""


class FormatterNameError(exceptions.EveRuntimeError):
    """Run-time error registering a new source code formatter."""

    ...


class FormattingError(exceptions.EveRuntimeError):
    """Run-time error applying a source code formatter."""

    ...


class TemplateDefinitionError(exceptions.EveTypeError):
    """Template definition error."""

    ...


class TemplateRenderingError(exceptions.EveRuntimeError):
    """Run-time error rendering a template."""

    ...


def register_formatter(language: str) -> Callable[[SourceFormatter], SourceFormatter]:
    """Register source code formatters for specific languages (decorator)."""

    def _decorator(formatter: SourceFormatter) -> SourceFormatter:
        if language in SOURCE_FORMATTERS:
            raise FormatterNameError(f"Another formatter for language '{language}' already exists")

        assert callable(formatter)
        SOURCE_FORMATTERS[language] = formatter

        return formatter

    return _decorator


@register_formatter("python")
def format_python_source(
    source: str,
    *,
    line_length: int = 100,
    python_versions: Optional[Set[str]] = None,
    string_normalization: bool = True,
) -> str:
    """Format Python source code using black formatter."""
    python_versions = python_versions or {f"{sys.version_info.major}{sys.version_info.minor}"}
    target_versions = set(
        black.TargetVersion[f"PY{v.replace('.', '')}"]  # type: ignore[attr-defined]
        for v in python_versions
    )

    formatted_source = black.format_str(
        source,
        mode=black.FileMode(
            line_length=line_length,
            target_versions=target_versions,
            string_normalization=string_normalization,
        ),
    )
    assert isinstance(formatted_source, str)

    return formatted_source


def _get_clang_format() -> Optional[str]:
    """Return the clang-format executable, or None if not available."""
    executable = os.getenv("CLANG_FORMAT_EXECUTABLE", "clang-format")
    try:
        assert isinstance(executable, str)
        if subprocess.run([executable, "--version"], capture_output=True).returncode != 0:
            return None
    except Exception:
        return None

    return executable


_CLANG_FORMAT_EXECUTABLE = _get_clang_format()


if _CLANG_FORMAT_EXECUTABLE is not None:

    @register_formatter("cpp")
    def format_cpp_source(
        source: str,
        *,
        style: Optional[str] = None,
        fallback_style: Optional[str] = None,
        sort_includes: bool = False,
    ) -> str:
        """Format C++ source code using clang-format."""
        assert isinstance(_CLANG_FORMAT_EXECUTABLE, str)
        args = [_CLANG_FORMAT_EXECUTABLE]
        if style:
            args.append(f"--style={style}")
        if fallback_style:
            args.append(f"--fallback-style={style}")
        if sort_includes:
            args.append("--sort-includes")

        try:
            # use a timeout as clang-format used to deadlock on some sources
            formatted_source = subprocess.run(
                args, check=True, input=source, capture_output=True, text=True, timeout=3
            ).stdout
        except subprocess.TimeoutExpired:
            return source

        assert isinstance(formatted_source, str)
        return formatted_source


def format_source(language: str, source: str, *, skip_errors: bool = True, **kwargs: Any) -> str:
    """Format source code if a formatter exists for the specific language."""
    formatter = SOURCE_FORMATTERS.get(language, None)
    try:
        if formatter:
            return formatter(source, **kwargs)
        else:
            raise FormattingError(f"Missing formatter for '{language}' language")
    except Exception as e:
        if skip_errors:
            return source
        else:
            raise FormattingError(
                f"Something went wrong when trying to format '{language}' source code"
            ) from e


class Name:
    """Text formatter with different case styles for symbol names in source code."""

    words: List[str]

    @classmethod
    def from_string(cls, name: str, case_style: utils.CaseStyleConverter.CASE_STYLE) -> Name:
        return cls(utils.CaseStyleConverter.split(name, case_style))

    def __init__(self, words: utils.AnyWordsIterable) -> None:
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        words = [*words]
        if not all(isinstance(item, str) for item in words):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        self.words = words

    def as_case(self, case_style: utils.CaseStyleConverter.CASE_STYLE) -> str:
        return utils.CaseStyleConverter.join(self.words, case_style)


AnyTextSequence = Union[Sequence[str], "TextBlock"]


class TextBlock:
    """A block of source code represented as a sequence of text lines.

    Check the provided context manager creator method (:meth:`indented`)
    for simple `indent - append - dedent` workflows.

    Args:
        indent_level: Initial indentation level.
        indent_size: Number of characters per indentation level.
        indent_char: Character used in the indentation.
        end_line: Character or string used as new-line separator.

    """

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ) -> None:
        if not isinstance(indent_char, str) or len(indent_char) != 1:
            raise ValueError("'indent_char' must be a single-character string")
        if not isinstance(end_line, str):
            raise ValueError("'end_line' must be a string")

        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines: List[str] = []

    def append(self, new_line: str, *, update_indent: int = 0) -> TextBlock:
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        self.lines.append(self.indent_str + new_line)

        return self

    def extend(self, new_lines: AnyTextSequence, *, dedent: bool = False) -> TextBlock:
        assert isinstance(new_lines, (collections.abc.Sequence, TextBlock))

        if dedent:
            if isinstance(new_lines, TextBlock):
                new_lines = textwrap.dedent(new_lines.text).splitlines()
            else:
                new_lines = textwrap.dedent("\n".join(new_lines)).splitlines()

        elif isinstance(new_lines, TextBlock):
            new_lines = new_lines.lines

        for line in new_lines:
            self.append(line)

        return self

    def empty_line(self, count: int = 1) -> TextBlock:
        self.lines.extend([""] * count)
        return self

    def indent(self, steps: int = 1) -> TextBlock:
        self.indent_level += steps
        return self

    def dedent(self, steps: int = 1) -> TextBlock:
        assert self.indent_level >= steps
        self.indent_level -= steps
        return self

    @contextlib.contextmanager
    def indented(self, steps: int = 1) -> Iterator[TextBlock]:
        """Context manager creator for temporary indentation of sources.

        This context manager simplifies the usage of indent/dedent in
        common `indent - append - dedent` workflows.

        Examples:
            >>> block = TextBlock();
            >>> block.append('first line')  # doctest: +ELLIPSIS
            <...>
            >>> with block.indented():
            ...     block.append('second line');  # doctest: +ELLIPSIS
            <...>
            >>> block.append('third line')  # doctest: +ELLIPSIS
            <...>
            >>> print(block.text)
            first line
                second line
            third line

        """
        self.indent(steps)
        yield self
        self.dedent(steps)

    @property
    def text(self) -> str:
        """Single string with the whole block contents."""
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    @property
    def indent_str(self) -> str:
        """Indentation string for new lines (in the current state)."""
        return self.indent_char * (self.indent_level * self.indent_size)

    def __iadd__(self, source_line: Union[str, AnyTextSequence]) -> TextBlock:
        if isinstance(source_line, str):
            return self.append(source_line)
        else:
            return self.extend(source_line)

    def __len__(self) -> int:
        return len(self.lines)

    def __str__(self) -> str:
        return self.text


TemplateT = TypeVar("TemplateT", bound="Template")


@runtime_checkable
class Template(Protocol):
    """Protocol (abstract base class) defining the Template interface.

    Direct subclassess of this base class only need to implement the
    abstract methods to adapt different template engines to this
    interface.

    Raises:
        TemplateDefinitionError: If the template definition contains errors.

    """

    def render(self, mapping: Optional[Mapping[str, str]] = None, **kwargs: Any) -> str:
        """Render the template.

        Args:
            mapping: A mapping whose keys match the template placeholders.

            ``**kwargs``: Placeholder values provided as keyword arguments (they will
                take precedence over ``mapping`` values for duplicated keys.

        Returns:
            The rendered string.

        Raises:
            TemplateRenderingError: If the template rendering fails.

        """
        if not mapping:
            mapping = {}
        if kwargs:
            mapping = {**mapping, **kwargs}

        return self.render_values(**mapping)

    @abc.abstractmethod
    def __init__(self, definition: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def render_values(self, **kwargs: Any) -> str:
        """Render the template.

        Args:
            ``**kwargs``: Placeholder values provided as keyword arguments.

        Returns:
            The rendered string.

        Raises:
            TemplateRenderingError: If the template rendering fails.

        """
        pass


class BaseTemplate(Template):
    """Helper class to add source location info of template definitions."""

    definition: Any
    definition_loc: Optional[Tuple[str, int]]

    def __init__(self) -> None:
        self.definition_loc = None
        frame = inspect.currentframe()
        try:
            if frame is not None and frame.f_back is not None and frame.f_back.f_back is not None:
                (filename, lineno, _, _, _) = inspect.getframeinfo(frame.f_back.f_back)
                self.definition_loc = (filename, lineno)
        except Exception:
            pass
        finally:
            del frame

    def __str__(self) -> str:
        result = f"<{type(self).__qualname__}: '{self.definition}'>"
        if self.definition_loc:
            result += f" created at {self.definition_loc[0]}:{self.definition_loc[1]}"
        return result


class FormatTemplate(BaseTemplate):
    """Template adapter to render regular strings as fully-featured f-strings."""

    definition: str

    def __init__(self, definition: str, **kwargs: Any) -> None:
        super().__init__()
        self.definition = f'(f"""{definition}""")'

    def render_values(self, **kwargs: Any) -> str:
        try:
            result = eval(self.definition, {}, kwargs or {})
            assert isinstance(result, str)
            return result
        except Exception as e:
            message = f"<{type(self).__name__}: '{self.definition}'>"
            if self.definition_loc:
                message += f" (created at {self.definition_loc[0]}:{self.definition_loc[1]})"
            message += " rendering error."

            raise TemplateRenderingError(message, template=self) from e


class StringTemplate(BaseTemplate):
    """Template adapter for `string.Template`."""

    definition: string.Template

    def __init__(self, definition: Union[str, string.Template], **kwargs: Any) -> None:
        super().__init__()
        if isinstance(definition, str):
            definition = string.Template(definition)
        assert isinstance(definition, string.Template)
        self.definition = definition

    def render_values(self, **kwargs: Any) -> str:
        try:
            return self.definition.substitute(**kwargs)
        except Exception as e:
            message = f"<{type(self).__name__}>"
            if self.definition_loc:
                message += f" (created at {self.definition_loc[0]}:{self.definition_loc[1]})"
            try:
                loc_info = re.search(r"line (\d+), col (\d+)", str(e))
                message += f" rendering error at template line: {loc_info[1]}, column {loc_info[2]}."  # type: ignore
            except Exception:
                message += " rendering error."

            raise TemplateRenderingError(message, template=self) from e


class JinjaTemplate(BaseTemplate):
    """Template adapter for `jinja2.Template`."""

    definition: jinja2.Template

    __jinja_env__ = jinja2.Environment(undefined=jinja2.StrictUndefined)

    def __init__(self, definition: Union[str, jinja2.Template], **kwargs: Any) -> None:
        super().__init__()
        try:
            if isinstance(definition, str):
                definition = self.__jinja_env__.from_string(definition)
            assert isinstance(definition, jinja2.Template)
            self.definition = definition
        except Exception as e:
            message = "Error in JinjaTemplate"
            if self.definition_loc:
                message += f" created at {self.definition_loc[0]}:{self.definition_loc[1]}"
                try:
                    if hasattr(e, "lineno"):
                        message += f" (error likely around line: {e.lineno})"  # type: ignore  # assume Jinja exception
                except Exception:
                    message = f"{message}:\n---\n{definition}\n---\n"

            raise TemplateDefinitionError(message, definition=definition) from e

    def render_values(self, **kwargs: Any) -> str:
        try:
            return self.definition.render(**kwargs)
        except Exception as e:
            message = f"<{type(self).__name__}>"
            if self.definition_loc:
                message += f" (created at {self.definition_loc[0]}:{self.definition_loc[1]})"
            try:
                message += f" rendering error at template line: {e.lineno}."  # type: ignore  # assume Jinja exception
            except Exception:
                message += " rendering error."

            raise TemplateRenderingError(message, template=self) from e


class MakoTemplate(BaseTemplate):
    """Template adapter for `mako.template.Template`."""

    definition: mako_tpl.Template

    def __init__(self, definition: mako_tpl.Template, **kwargs: Any) -> None:
        super().__init__()
        try:
            if isinstance(definition, str):
                definition = mako_tpl.Template(definition)
            assert isinstance(definition, mako_tpl.Template)
            self.definition = definition
        except Exception as e:
            message = "Error in MakoTemplate"
            if self.definition_loc:
                message += f" created at {self.definition_loc[0]}:{self.definition_loc[1]}"
                try:
                    message += f" (error likely around line {e.lineno}, column: {getattr(e, 'pos', '?')})"  # type: ignore  # assume Mako exception
                except Exception:
                    message = f"{message}:\n---\n{definition}\n---\n"

            raise TemplateDefinitionError(message, definition=definition) from e

    def render_values(self, **kwargs: Any) -> str:
        try:
            result = self.definition.render(**kwargs)
            assert isinstance(result, str)
            return result
        except Exception as e:
            message = f"<{type(self).__name__}>"
            if self.definition_loc:
                message += f" (created at {self.definition_loc[0]}:{self.definition_loc[1]})"
            try:
                message += f" rendering error at template line: {e.lineno}, column: {getattr(e, 'pos', '?')}"  # type: ignore  # assume Mako exception
            except Exception:
                message += " rendering error."
            raise TemplateRenderingError(message, template=self) from e


class TemplatedGenerator(NodeVisitor):
    """A code generator visitor using :class:`TextTemplate`.

    The order followed to choose a `dump()` function for node values is the following:

        1. A ``self.visit_NODE_CLASS_NAME()`` method where `NODE_CLASS_NAME`
           matches ``type(node).__name__``.
        2. A ``self.visit_NODE_BASE_CLASS_NAME()`` method where
           `NODE_BASE_CLASS_NAME` matches ``base.__name__``, and `base` is
           one of the node base classes (evaluated following the order
           given in ``type(node).__mro__``).

        If `node` is an instance of :class:`eve.Node` and a `visit_` method has
        not been found, `TemplatedGenerator` will look for an appropriate
        :class:`Template` definition:

        3. A ``NODE_CLASS_NAME`` class variable of type :class:`Template`,
           where ``NODE_CLASS_NAME`` matches ``type(node).__name__``.
        4. A ``NODE_BASE_CLASS_NAME`` class variable of type :class:`Template`,
           where ``NODE_BASE_CLASS_NAME`` matches ``base.__name__``, and
           `base` is one of the node base classes (evaluated following the
           order given in ``type(node).__mro__``).

        In any other case (templates cannot be used for instances of arbitrary types),
        steps 3 and 4 will be substituted by a call to the :meth:`self.generic_dump()`
        method.

    The following keys are passed to template instances at rendering:

        * ``**node_fields``: all the node children and implementation fields by name.
        * ``_impl``: a ``dict`` instance with the results of visiting all
          the node implementation fields.
        * ``_children``: a ``dict`` instance with the results of visiting all
          the node children.
        * ``_this_node``: the actual node instance (before visiting children).
        * ``_this_generator``: the current generator instance.
        * ``_this_module``: the generator's module instance.
        * ``**kwargs``: the keyword arguments received by the visiting method.

    Visitor methods can trigger regular template rendering for the same node class
    by explicitly calling :meth:`generic_visit()` (typically at the end), which will
    continue the search rendering algorithm at step 3. Thus, a common pattern to deal
    with complicated nodes is to define both a visitor method and a template for
    the same class, where the visitor method preprocess the node data and calls
    :meth:`generic_visit()` at the end with additional keyword arguments which will
    be forwarded to the node template.

    """

    __templates__: ClassVar[Mapping[str, Template]]

    @classmethod
    def __init_subclass__(cls, *, inherit_templates: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__templates__" in cls.__dict__:
            raise TypeError(f"Invalid '__templates__' member in class {cls}")

        templates: Dict[str, Template] = {}
        if inherit_templates:
            for templated_gen_class in reversed(cls.__mro__[1:]):
                if (
                    issubclass(templated_gen_class, TemplatedGenerator)
                    and templated_gen_class is not TemplatedGenerator
                ):
                    templates.update(templated_gen_class.__templates__)

        templates.update(
            {
                key: value
                for key, value in cls.__dict__.items()
                if isinstance(value, Template) and not key.startswith("_") and not key.endswith("_")
            }
        )

        cls.__templates__ = types.MappingProxyType(templates)

    @overload
    @classmethod
    def apply(cls, root: LeafNode, **kwargs: Any) -> str:
        ...

    @overload
    @classmethod
    def apply(  # noqa: F811  # redefinition of symbol
        cls, root: CollectionNode, **kwargs: Any
    ) -> Collection[str]:
        ...

    @classmethod
    def apply(  # noqa: F811  # redefinition of symbol
        cls, root: RootNode, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        """Public method to build a class instance and visit an IR node.

        Args:
            root: An IR node.
            node_templates (optiona): see :class:`NodeDumper`.
            dump_function (optiona): see :class:`NodeDumper`.
            ``**kwargs`` (optional): custom extra parameters forwarded to `visit_NODE_TYPE_NAME()`.

        Returns:
            String (or collection of strings) with the dumped version of the root IR node.

        """
        return cls().visit(root, **kwargs)

    @classmethod
    def generic_dump(cls, node: RootNode, **kwargs: Any) -> str:
        """Class-specific ``dump()`` function for primitive types.

        This class could be redefined in the subclasses.
        """
        return str(node)

    def generic_visit(self, node: RootNode, **kwargs: Any) -> Union[str, Collection[str]]:
        if isinstance(node, Node):
            template, key = self.get_template(node)
            if template:
                try:
                    return self.render_template(
                        template,
                        node,
                        self.transform_children(node, **kwargs),
                        self.transform_annexed_items(node, **kwargs),
                        **kwargs,
                    )
                except TemplateRenderingError as e:
                    # Raise a new exception with extra information keeping the original cause
                    raise TemplateRenderingError(
                        f"Error in '{key}' template when rendering node '{node}'.\n"
                        + getattr(e, "message", str(e)),
                        **e.info,
                        node=node,
                    ) from e.__cause__

        if isinstance(node, (list, tuple, collections.abc.Set)) or (
            isinstance(node, collections.abc.Sequence) and not isinstance(node, (str, bytes))
        ):
            return [self.visit(value, **kwargs) for value in node]
        elif isinstance(node, (dict, collections.abc.Mapping)):
            return {key: self.visit(value, **kwargs) for key, value in node.items()}

        return self.generic_dump(node, **kwargs)

    def get_template(self, node: RootNode) -> Tuple[Optional[Template], Optional[str]]:
        """Get a template for a node instance (see class documentation)."""
        template: Optional[Template] = None
        template_key = None
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                template_key = node_class.__name__
                template = self.__templates__.get(template_key, None)
                if template is not None or node_class is Node:
                    break

        return template, None if template is None else template_key

    def render_template(
        self,
        template: Template,
        node: Node,
        transformed_children: Mapping[str, Any],
        transformed_annexed_items: Mapping[str, Any],
        **kwargs: Any,
    ) -> str:
        """Render a template using node instance data (see class documentation)."""
        return template.render(
            **{**transformed_children, **transformed_annexed_items, **kwargs},
            _children=transformed_children,
            _impl=transformed_annexed_items,
            _this_node=node,
            _this_generator=self,
            _this_module=sys.modules[type(self).__module__],
        )

    def transform_children(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.iter_children_items()}  # type: ignore[misc]

    def transform_annexed_items(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.annex.items()}
