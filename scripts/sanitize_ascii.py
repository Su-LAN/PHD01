#!/usr/bin/env python3
"""
Repo-wide ASCII sanitization for comments and docstrings.

- Scans .py and .ipynb files
- In .py: sanitizes module/class/function docstrings and # comments only
  (uses ast + tokenize to precisely target non-executable text)
- In .ipynb: sanitizes code cell docstrings (triple-quoted) and comment lines

Heuristics: only lines that contain typical mojibake/suspect characters are
sanitized; normal English/Chinese content is left untouched.

Usage:
  python scripts/sanitize_ascii.py --apply          # in-place update
  python scripts/sanitize_ascii.py --dry-run        # show planned changes

Notes:
  - Creates no backups by default; use your VCS to review/undo.
  - Requires Python 3.8+ (for ast end_lineno).
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tokenize
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple, Dict


SUSPECT_CHARS = set(
    # Common mojibake fragments observed in this repo (not exhaustive)
    list("瀹鏋鍥闂缁灉浠鈹鉁鈿锛鐨妗璇鍗杩閫閲绱鐢") + ["�"]
)


def has_mojibake(text: str) -> bool:
    return any(ch in SUSPECT_CHARS for ch in text)


def to_ascii(text: str) -> str:
    # Normalize to ASCII, dropping non-ASCII; also collapse long non-ASCII lines
    try:
        norm = unicodedata.normalize("NFKD", text)
        asc = norm.encode("ascii", "ignore").decode("ascii")
        # Replace empty decorative lines with ASCII separators if they were long
        stripped = text.strip()
        if stripped and not asc.strip() and len(stripped) >= 5:
            return "=" * min(80, len(stripped))
        return asc
    except Exception:
        return "".join(ch if ord(ch) < 128 else "" for ch in text)


def sanitize_py(path: Path, apply: bool) -> Tuple[int, int]:
    """Return (lines_changed, occurrences)"""
    src = path.read_text(encoding="utf-8", errors="replace")

    # Collect docstring spans via AST (module, classes, functions)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (0, 0)

    doc_spans: List[Tuple[int, int]] = []  # 1-based inclusive line spans

    def add_doc(node):
        if not getattr(node, "body", None):
            return
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
            # Python 3.8+ has end_lineno
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", None)
            if start and end:
                doc_spans.append((start, end))

    add_doc(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_doc(node)

    # Tokenize to locate comments
    comment_by_line: Dict[int, List[Tuple[int, str]]] = {}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        toks = []
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            (line, col) = tok.start
            comment_by_line.setdefault(line, []).append((col, tok.string))

    lines = src.splitlines(True)  # keepends
    changed = 0
    occurrences = 0

    # Sanitize docstring content (not the quote markers)
    spans_by_line = set()
    for (a, b) in doc_spans:
        for ln in range(a, b + 1):
            spans_by_line.add(ln)

    for ln in range(1, len(lines) + 1):
        orig = lines[ln - 1]
        new = orig

        # Comments
        if ln in comment_by_line:
            # Only sanitize the comment tail
            for (col, text) in comment_by_line[ln]:
                if has_mojibake(text):
                    prefix = new[:col]
                    sanitized = to_ascii(text)
                    new = prefix + sanitized + ("" if new.endswith("\n") else "")
                    occurrences += 1

        # Docstring content lines: heuristically sanitize if mojibake present
        if ln in spans_by_line and has_mojibake(orig):
            new = to_ascii(orig)
            occurrences += 1

        if new != orig:
            lines[ln - 1] = new
            changed += 1

    if changed and apply:
        path.write_text("".join(lines), encoding="utf-8")

    return changed, occurrences


def sanitize_ipynb(path: Path, apply: bool) -> Tuple[int, int]:
    # Some ipynb files may carry a UTF-8 BOM; try utf-8 first, then utf-8-sig
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        data = json.loads(raw)
    if "cells" not in data:
        return (0, 0)

    changed = 0
    occurrences = 0

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if not isinstance(src, list):
            continue

        # Detect docstring blocks inside the cell
        in_doc = False
        quote_type = None
        for i, line in enumerate(src):
            stripped = line.strip()

            # Toggle docstring multi-line mode
            if not in_doc and (stripped in ('"""', "'''")):
                in_doc = True
                quote_type = stripped
                continue
            elif in_doc and stripped == quote_type:
                in_doc = False
                quote_type = None
                continue

            # Single-line docstring like """ ... """
            if not in_doc and ((stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6) or (stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6)):
                inner = stripped[3:-3]
                if has_mojibake(inner):
                    new_inner = to_ascii(inner)
                    src[i] = line.replace(inner, new_inner)
                    changed += 1
                    occurrences += 1
                continue

            # Inside docstring: sanitize if mojibake
            if in_doc and has_mojibake(line):
                src[i] = to_ascii(line) + ("" if line.endswith("\n") else "")
                changed += 1
                occurrences += 1
                continue

            # Comment line: sanitize mojibake
            lstr = line.lstrip()
            if lstr.startswith('#') and has_mojibake(line):
                src[i] = to_ascii(line)
                changed += 1
                occurrences += 1

        cell["source"] = src

    if changed and apply:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    return changed, occurrences


def main(argv: List[str]) -> int:
    apply = "--apply" in argv and "--dry-run" not in argv
    root = Path.cwd()

    targets: List[Path] = []
    for ext in (".py", ".ipynb"):
        targets.extend(p for p in root.rglob(f"*{ext}") if ".venv" not in p.parts)

    total_changed = 0
    total_occ = 0
    for p in targets:
        if p.suffix == ".py":
            changed, occ = sanitize_py(p, apply)
        else:
            changed, occ = sanitize_ipynb(p, apply)
        if changed:
            action = "UPDATED" if apply else "WOULD-UPDATE"
            print(f"{action} {p} (lines={changed}, occurrences={occ})")
            total_changed += changed
            total_occ += occ

    print(f"\nDone. Files affected lines={total_changed}, occurrences={total_occ}. Mode={'apply' if apply else 'dry-run'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
