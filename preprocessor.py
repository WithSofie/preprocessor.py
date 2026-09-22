#!/usr/bin/env python3
"""Flatten local ``from module import *`` imports into one Python file.

The source tree remains ordinary Python so editors, language servers, type checkers,
and test runners can inspect the original modules. During the build step, local
wildcard imports act as include directives and are recursively replaced by the
referenced source files.

Only the Python standard library is used. The preprocessor never imports or
executes project modules while building the output.
"""

from __future__ import annotations

import argparse
import ast
import io
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
import tokenize
from typing import Iterable, Sequence


PROGRAM = "preprocessor"
VERSION = "0.2.0"
_ENCODING_COOKIE_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


class PreprocessorError(Exception):
    """Base class for user-facing preprocessing failures."""


class ResolutionError(PreprocessorError):
    """Raised when strict local-module resolution fails."""


@dataclass(frozen=True)
class IncludeDirective:
    """A top-level wildcard import and its resolved local target, if any."""

    node: ast.ImportFrom
    display_name: str
    target: Path | None


@dataclass
class ModuleInfo:
    """Parsed information for one source module."""

    path: Path
    source: str
    tree: ast.Module
    lines: list[str]
    includes: list[IncludeDirective] = field(default_factory=list)
    future_imports: list[ast.ImportFrom] = field(default_factory=list)
    main_guards: list[ast.If] = field(default_factory=list)
    inline_comment_unsafe_lines: set[int] = field(default_factory=set)


@dataclass
class BuildResult:
    """Final generated source plus build metadata."""

    source: str
    included_paths: list[Path]
    unresolved_wildcards: list[tuple[Path, str]]
    cycles: list[tuple[Path, Path]]


def _is_wildcard_import(node: ast.ImportFrom) -> bool:
    return len(node.names) == 1 and node.names[0].name == "*"


def _is_main_guard(node: ast.stmt) -> bool:
    """Return True for the conventional ``if __name__ == '__main__':`` guard."""
    if not isinstance(node, ast.If):
        return False

    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False

    left = test.left
    right = test.comparators[0]

    def is_name(value: ast.expr) -> bool:
        return isinstance(value, ast.Name) and value.id == "__name__"

    def is_main(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) and value.value == "__main__"

    return (is_name(left) and is_main(right)) or (is_main(left) and is_name(right))


def _import_display_name(node: ast.ImportFrom) -> str:
    dots = "." * node.level
    module = node.module or ""
    return f"{dots}{module}"


def _read_python_source(path: Path) -> str:
    """Read Python using its declared PEP 263 encoding."""
    try:
        with tokenize.open(path) as handle:
            return handle.read()
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PreprocessorError(f"cannot read {path}: {exc}") from exc


def _inline_comment_unsafe_lines(source: str) -> set[int]:
    """Return 1-based physical lines where an inline comment may change code.

    Multi-line lexical tokens (most importantly triple-quoted strings) cannot
    safely receive a source-map comment on their interior physical lines. An
    explicit backslash continuation also cannot have anything appended after
    the backslash. Those lines are therefore left untouched in inline-map mode.
    """
    unsafe: set[int] = set()

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            start_line = token.start[0]
            end_line = token.end[0]
            if end_line > start_line:
                unsafe.update(range(start_line, end_line))
    except tokenize.TokenError:
        # ``ast.parse`` is the authoritative syntax check and runs immediately
        # afterwards. If tokenization cannot finish here, simply avoid deriving
        # extra unsafe ranges rather than masking the eventual syntax error.
        pass

    for number, line in enumerate(source.splitlines(), start=1):
        if line.rstrip().endswith("\\"):
            unsafe.add(number)

    return unsafe


def _parse_python(path: Path, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=str(path), type_comments=True)
    except SyntaxError as exc:
        location = f"{path}:{exc.lineno or '?'}:{exc.offset or '?'}"
        message = exc.msg or "invalid Python syntax"
        raise PreprocessorError(f"syntax error at {location}: {message}") from exc


def _module_candidates(base: Path) -> tuple[Path, Path]:
    return base.with_suffix(".py"), base / "__init__.py"


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


class ModuleResolver:
    """Resolve local wildcard imports without importing or executing modules."""

    def __init__(self, root: Path, search_paths: Sequence[Path] = ()) -> None:
        roots = [root, *search_paths]
        self.roots: list[Path] = []
        seen: set[Path] = set()
        for item in roots:
            resolved = item.expanduser().resolve()
            if resolved not in seen:
                self.roots.append(resolved)
                seen.add(resolved)

    def resolve(self, importer: Path, node: ast.ImportFrom) -> Path | None:
        if node.module == "__future__":
            return None

        module_parts = node.module.split(".") if node.module else []

        if node.level:
            base = importer.parent
            for _ in range(node.level - 1):
                base = base.parent
            target_base = base.joinpath(*module_parts) if module_parts else base
            return _first_existing(_module_candidates(target_base))

        if not module_parts:
            return None

        for root in self.roots:
            target_base = root.joinpath(*module_parts)
            found = _first_existing(_module_candidates(target_base))
            if found is not None:
                return found
        return None


class ProjectAnalyzer:
    """Parse the reachable include graph before rendering it."""

    def __init__(
        self,
        resolver: ModuleResolver,
        *,
        strict: bool = False,
    ) -> None:
        self.resolver = resolver
        self.strict = strict
        self.modules: dict[Path, ModuleInfo] = {}
        self.discovery_order: list[Path] = []
        self.unresolved_wildcards: list[tuple[Path, str]] = []

    def analyze(self, root_file: Path) -> None:
        self._load(root_file.resolve())

    def _load(self, path: Path) -> ModuleInfo:
        path = path.resolve()
        existing = self.modules.get(path)
        if existing is not None:
            return existing

        if not path.is_file():
            raise PreprocessorError(f"source file does not exist: {path}")

        source = _read_python_source(path)
        tree = _parse_python(path, source)
        info = ModuleInfo(
            path=path,
            source=source,
            tree=tree,
            lines=source.splitlines(keepends=True),
            inline_comment_unsafe_lines=_inline_comment_unsafe_lines(source),
        )

        # Register before recursing so circular dependency graphs terminate.
        self.modules[path] = info
        self.discovery_order.append(path)

        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                if statement.module == "__future__":
                    info.future_imports.append(statement)
                    continue

                if _is_wildcard_import(statement):
                    display_name = _import_display_name(statement)
                    target = self.resolver.resolve(path, statement)
                    directive = IncludeDirective(statement, display_name, target)
                    info.includes.append(directive)

                    if target is None:
                        self.unresolved_wildcards.append((path, display_name))
                        if self.strict:
                            raise ResolutionError(
                                f"cannot resolve wildcard import {display_name!r} "
                                f"from {path}"
                            )
                    else:
                        self._load(target)

            if _is_main_guard(statement):
                info.main_guards.append(statement)

        return info


@dataclass(frozen=True)
class _LineReplacement:
    start: int
    end: int
    text: str


class Renderer:
    """Render an analyzed module graph into one source file."""

    def __init__(
        self,
        analyzer: ProjectAnalyzer,
        root_file: Path,
        *,
        markers: bool = True,
        keep_main_guards: bool = False,
        inline_source_map: bool = False,
        source_marker: bool = False,
    ) -> None:
        self.analyzer = analyzer
        self.root_file = root_file.resolve()
        self.markers = markers
        self.keep_main_guards = keep_main_guards
        self.inline_source_map = inline_source_map
        self.source_marker = source_marker
        self.included: set[Path] = set()
        self.active: list[Path] = []
        self.render_order: list[Path] = []
        self.cycles: list[tuple[Path, Path]] = []

    def render(self) -> BuildResult:
        root_info = self.analyzer.modules[self.root_file]
        shebang = self._root_shebang(root_info)
        future_block = self._future_block()

        body = self._render_module(self.root_file, is_root=True, future_block=future_block)

        header_lines: list[str] = []
        if shebang:
            header_lines.append(shebang.rstrip("\r\n") + "\n")
        header_lines.append("# -*- coding: utf-8 -*-\n")
        header_lines.append(
            f"# Generated by {PROGRAM} {VERSION}; edit the source modules, not this file.\n"
        )
        header_lines.append("\n")

        generated = "".join(header_lines) + body
        return BuildResult(
            source=generated,
            included_paths=self.render_order.copy(),
            unresolved_wildcards=self.analyzer.unresolved_wildcards.copy(),
            cycles=self.cycles.copy(),
        )

    def _render_module(
        self,
        path: Path,
        *,
        is_root: bool,
        future_block: str = "",
    ) -> str:
        path = path.resolve()

        if path in self.active:
            parent = self.active[-1]
            self.cycles.append((parent, path))
            return self._skip_marker(path, "cycle")

        if path in self.included:
            return self._skip_marker(path, "already included")

        self.included.add(path)
        self.active.append(path)
        self.render_order.append(path)

        info = self.analyzer.modules[path]
        replacements: list[_LineReplacement] = []

        # Shebangs and source-encoding cookies from inputs are not copied into
        # the combined body. The output gets one UTF-8 declaration at the top.
        replacements.extend(self._metadata_replacements(info, is_root=is_root))

        for node in info.future_imports:
            replacements.append(
                _LineReplacement(node.lineno - 1, node.end_lineno or node.lineno, "")
            )

        if not is_root and not self.keep_main_guards:
            for node in info.main_guards:
                replacements.append(
                    _LineReplacement(node.lineno - 1, node.end_lineno or node.lineno, "")
                )

        for directive in info.includes:
            if directive.target is None:
                continue
            replacement = self._render_module(directive.target, is_root=False)
            replacements.append(
                _LineReplacement(
                    directive.node.lineno - 1,
                    directive.node.end_lineno or directive.node.lineno,
                    replacement,
                )
            )

        if is_root and future_block:
            insertion_line = self._future_insertion_line(info)
            replacements.append(_LineReplacement(insertion_line, insertion_line, future_block))

        rendered = self._apply_replacements(info.lines, replacements, info.path)
        self.active.pop()

        if is_root or not self.markers:
            return rendered

        label = self._display_path(path)
        return (
            f"# >>> {PROGRAM}: begin {label}\n"
            f"{rendered}"
            f"# <<< {PROGRAM}: end {label}\n"
        )

    def _apply_replacements(
        self,
        lines: list[str],
        replacements: list[_LineReplacement],
        path: Path,
    ) -> str:
        ordered = sorted(replacements, key=lambda item: (item.start, item.end))
        normalized: list[_LineReplacement] = []

        for replacement in ordered:
            if not (0 <= replacement.start <= replacement.end <= len(lines)):
                raise PreprocessorError(
                    f"internal replacement range is outside {path}: "
                    f"{replacement.start}:{replacement.end}"
                )

            if normalized and replacement.start == normalized[-1].start:
                previous = normalized.pop()
                if previous.start == previous.end and replacement.start < replacement.end:
                    normalized.append(
                        _LineReplacement(
                            replacement.start,
                            replacement.end,
                            previous.text + replacement.text,
                        )
                    )
                    continue
                if replacement.start == replacement.end and previous.start < previous.end:
                    normalized.append(
                        _LineReplacement(
                            previous.start,
                            previous.end,
                            replacement.text + previous.text,
                        )
                    )
                    continue
                raise PreprocessorError(
                    f"internal overlapping replacements in {path} at line "
                    f"{replacement.start + 1}"
                )

            if normalized and replacement.start < normalized[-1].end:
                raise PreprocessorError(
                    f"internal overlapping replacements in {path} near line "
                    f"{replacement.start + 1}"
                )

            normalized.append(replacement)

        output: list[str] = []
        cursor = 0
        info = self.analyzer.modules[path.resolve()]

        for replacement in normalized:
            output.append(
                self._render_source_segment(
                    info,
                    cursor,
                    replacement.start,
                )
            )
            output.append(replacement.text)
            cursor = replacement.end

        output.append(self._render_source_segment(info, cursor, len(lines)))
        return "".join(output)

    def _render_source_segment(
        self,
        info: ModuleInfo,
        start: int,
        end: int,
    ) -> str:
        """Render an untouched original-source segment with optional tracing."""
        if start >= end:
            return ""

        segment = info.lines[start:end]
        if self.inline_source_map:
            rendered: list[str] = []
            label = self._display_path(info.path)
            for offset, line in enumerate(segment, start=start + 1):
                rendered.append(
                    self._append_inline_source_comment(
                        line,
                        label,
                        offset,
                        unsafe=offset in info.inline_comment_unsafe_lines,
                    )
                )
            return "".join(rendered)

        if self.source_marker:
            label = self._display_path(info.path)
            return (
                f"# >>> {PROGRAM}: source {label}:{start + 1}\n"
                + "".join(segment)
            )

        return "".join(segment)

    @staticmethod
    def _append_inline_source_comment(
        line: str,
        label: str,
        line_number: int,
        *,
        unsafe: bool,
    ) -> str:
        """Append ``# file.py N`` without changing unsafe physical lines."""
        if unsafe:
            return line

        if line.endswith("\r\n"):
            content, ending = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            content, ending = line[:-1], line[-1:]
        else:
            content, ending = line, ""

        # Keep genuinely blank lines blank. They carry no executable source and
        # annotating them only adds noise to generated artifacts.
        if not content.strip():
            return line

        return f"{content} # {label} {line_number}{ending}"

    def _metadata_replacements(
        self,
        info: ModuleInfo,
        *,
        is_root: bool,
    ) -> list[_LineReplacement]:
        replacements: list[_LineReplacement] = []
        for index, line in enumerate(info.lines[:2]):
            stripped = line.lstrip()
            if index == 0 and stripped.startswith("#!"):
                replacements.append(_LineReplacement(index, index + 1, ""))
                continue
            if _ENCODING_COOKIE_RE.search(line):
                replacements.append(_LineReplacement(index, index + 1, ""))

        # Included-file shebangs only have special meaning on line 1, but we
        # remove them above for both root and dependencies. ``is_root`` is kept
        # in the signature to make that policy explicit and easy to extend.
        _ = is_root
        return replacements

    def _root_shebang(self, info: ModuleInfo) -> str | None:
        if info.lines and info.lines[0].startswith("#!"):
            return info.lines[0]
        return None

    def _future_block(self) -> str:
        statements: list[tuple[str, Path, int]] = []
        seen: set[str] = set()

        for path in self.analyzer.discovery_order:
            info = self.analyzer.modules[path]
            for node in info.future_imports:
                statement = ast.unparse(node).strip()
                if statement not in seen:
                    statements.append((statement, path, node.lineno))
                    seen.add(statement)

        if not statements:
            return ""

        lines: list[str] = []
        for statement, path, line_number in statements:
            label = self._display_path(path)
            if self.inline_source_map:
                lines.append(f"{statement} # {label} {line_number}\n")
            elif self.source_marker:
                lines.append(
                    f"# >>> {PROGRAM}: source {label}:{line_number}\n"
                    f"{statement}\n"
                )
            else:
                lines.append(f"{statement}\n")
        lines.append("\n")
        return "".join(lines)

    def _future_insertion_line(self, info: ModuleInfo) -> int:
        """Return a zero-based line boundary after the root module docstring."""
        if not info.tree.body:
            return len(info.lines)

        first = info.tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first.end_lineno or first.lineno

        # A future statement may follow comments/blank lines. Inserting at the
        # beginning of the source body is valid because generated header lines
        # are comments, not statements.
        return 0

    def _display_path(self, path: Path) -> str:
        for root in self.analyzer.resolver.roots:
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    def _skip_marker(self, path: Path, reason: str) -> str:
        if not self.markers:
            return ""
        return f"# --- {PROGRAM}: {reason}: {self._display_path(path)} ---\n"


def _validate_generated_source(source: str, output_name: str) -> None:
    try:
        compile(source, output_name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        line = exc.lineno or "?"
        column = exc.offset or "?"
        message = exc.msg or "invalid generated syntax"
        raise PreprocessorError(
            f"generated file is not valid Python at {output_name}:{line}:{column}: {message}"
        ) from exc


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.flat.py")


def _write_output(path: Path, source: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise PreprocessorError(f"cannot write {path}: {exc}") from exc


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocessor.py",
        description=(
            "Recursively flatten local 'from module import *' statements into "
            "one Python source file."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="entry Python file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: <input>.flat.py)",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        help="root directory for absolute local imports (default: input directory)",
    )
    parser.add_argument(
        "-I",
        "--search-path",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="additional local import search directory; repeatable",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any wildcard import cannot be resolved as a local file",
    )
    parser.add_argument(
        "--keep-main-guards",
        action="store_true",
        help="keep __name__ == '__main__' blocks from included modules",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="omit begin/end and duplicate/cycle comments in generated output",
    )
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument(
        "--inline-source-map",
        action="store_true",
        help=(
            "append '# file.py LINE' to safe original source lines in the "
            "generated output"
        ),
    )
    tracing.add_argument(
        "--source-marker",
        action="store_true",
        help=(
            "insert '# >>> preprocessor: source file.py:LINE' before each "
            "contiguous original-source segment"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="analyze, render, and syntax-check without writing the output file",
    )
    parser.add_argument(
        "--list-deps",
        action="store_true",
        help="print included source files in render order",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print resolution and cycle information",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    return parser


def build(args: argparse.Namespace) -> tuple[BuildResult, Path]:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise PreprocessorError(f"input file does not exist: {input_path}")
    if input_path.suffix != ".py":
        raise PreprocessorError(f"input file must end in .py: {input_path}")

    root = (args.root or input_path.parent).expanduser().resolve()
    if not root.is_dir():
        raise PreprocessorError(f"root directory does not exist: {root}")

    search_paths = [path.expanduser().resolve() for path in args.search_path]
    for path in search_paths:
        if not path.is_dir():
            raise PreprocessorError(f"search path does not exist: {path}")

    resolver = ModuleResolver(root, search_paths)
    analyzer = ProjectAnalyzer(resolver, strict=args.strict)
    analyzer.analyze(input_path)

    renderer = Renderer(
        analyzer,
        input_path,
        markers=not args.no_markers,
        keep_main_guards=args.keep_main_guards,
        inline_source_map=args.inline_source_map,
        source_marker=args.source_marker,
    )
    result = renderer.render()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(input_path).resolve()
    )

    if output_path in result.included_paths:
        raise PreprocessorError(
            "output path would overwrite a source file: "
            f"{output_path}; choose a different -o/--output path"
        )

    _validate_generated_source(result.source, str(output_path))
    return result, output_path


def _print_report(result: BuildResult, output_path: Path, args: argparse.Namespace) -> None:
    if args.list_deps:
        for path in result.included_paths:
            print(path)

    if args.verbose:
        for path, module in result.unresolved_wildcards:
            print(
                f"{PROGRAM}: kept unresolved/external wildcard import "
                f"{module!r} in {path}",
                file=sys.stderr,
            )
        for parent, target in result.cycles:
            print(
                f"{PROGRAM}: cycle edge skipped: {parent} -> {target}",
                file=sys.stderr,
            )

    if args.check_only:
        print(
            f"{PROGRAM}: OK; {len(result.included_paths)} source file(s) would be "
            f"combined into {output_path}",
            file=sys.stderr,
        )
    else:
        print(
            f"{PROGRAM}: wrote {output_path} from {len(result.included_paths)} source file(s)",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    try:
        result, output_path = build(args)
        if not args.check_only:
            _write_output(output_path, result.source)
        _print_report(result, output_path, args)
        return 0
    except PreprocessorError as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
