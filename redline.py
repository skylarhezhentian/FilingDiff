"""
redline.py
----------
Inline word-level diff rendering, similar to the inFilings/Verity
"redline" style: deletions show with strikethrough on a light-red
background, additions show on a light-green background, and unchanged
words appear plain. Numbers that change between old and new naturally
fall out as "<del>old</del> <add>new</add>" pairs.

Why this beats the block before/after format:
  - Easier to scan: only the changed words pop visually.
  - Numeric updates (e.g. "$5.0 million" -> "$3.1 million") land right
    next to each other so the reader can read the delta inline.
  - Pure deletions don't dominate the page; they appear as small
    strikethrough spans within otherwise intact paragraphs.

The output is a string of HTML safe to render inside Streamlit with
`unsafe_allow_html=True`.
"""

from __future__ import annotations

import difflib
import re
from html import escape

_ADD_STYLE = (
    "background:#DCFCE7;color:#166534;border-radius:3px;"
    "padding:0 2px;"
)
_DEL_STYLE = (
    "background:#FEE2E2;color:#7F1D1D;text-decoration:line-through;"
    "border-radius:3px;padding:0 2px;"
)

# Token regex: keep punctuation attached to words so the diff doesn't
# spray "(", ")", "." into individual changes. We tokenise on whitespace
# but preserve currency / percent / parenthesized numerics as one unit.
_TOKEN_RE = re.compile(r"\S+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _wrap(span_class: str, text: str) -> str:
    style = _ADD_STYLE if span_class == "add" else _DEL_STYLE
    return f'<span style="{style}">{escape(text)}</span>'


def redline_html(old: str, new: str) -> str:
    """Return inline word-level redline HTML comparing old -> new.

    Unchanged spans are escaped plain text.  Inserts are green, deletes
    are struck-through red.  Replacements render as deleted-old followed
    by added-new so the reader sees both values side by side.
    """
    if not old:
        return _wrap("add", new) if new else ""
    if not new:
        return _wrap("del", old)

    old_toks = _tokens(old)
    new_toks = _tokens(new)
    sm = difflib.SequenceMatcher(a=old_toks, b=new_toks, autojunk=False)

    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(escape(" ".join(new_toks[j1:j2])))
        elif tag == "replace":
            parts.append(_wrap("del", " ".join(old_toks[i1:i2])))
            parts.append(_wrap("add", " ".join(new_toks[j1:j2])))
        elif tag == "delete":
            parts.append(_wrap("del", " ".join(old_toks[i1:i2])))
        elif tag == "insert":
            parts.append(_wrap("add", " ".join(new_toks[j1:j2])))

    # Rejoin with single spaces; collapse any double spaces that crept in.
    out = " ".join(p for p in parts if p)
    out = re.sub(r"\s+", " ", out)
    return out


def added_html(new: str) -> str:
    """Render a fully-new paragraph as a single 'added' span."""
    return _wrap("add", new)


def deleted_html(old: str) -> str:
    """Render a fully-removed paragraph as a single 'deleted' span."""
    return _wrap("del", old)
