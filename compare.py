"""
compare.py
----------
Paragraph-level comparison between two filings.

Given two lists of `Paragraph` objects (old, new), we want to know:
  - Which paragraphs are NEW in the latest filing?
  - Which paragraphs were DELETED?
  - Which paragraphs are CHANGED (with a similarity score so we can label
    the change as Small / Medium / Big)?

We do this with rapidfuzz's token_set_ratio, which is robust to small
re-orderings and punctuation changes that are very common between
consecutive SEC filings.

Note: this is intentionally simple and O(N*M). Real filings have a few
thousand paragraphs after we drop tables, which still runs in well under
a second on a laptop. If you want to scale up, use rapidfuzz.process.cdist.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from extract import Paragraph


# Similarity thresholds (token_set_ratio is 0-100).
# Above HIGH we consider paragraphs effectively the same.
# Between MED and HIGH it's a "Small Change" (typo/word swap).
# Between LOW and MED it's a "Medium Change".
# Below LOW it's a "Big Change" (or we treat it as new/deleted).
SIM_HIGH = 92
SIM_MED = 78
SIM_LOW = 55


@dataclass
class ParagraphChange:
    """One unit of difference between two filings."""

    change_type: str  # "New" | "Deleted" | "Small Change" | "Medium Change" | "Big Change"
    section: str
    old_text: str  # "" for New
    new_text: str  # "" for Deleted
    similarity: float  # 0-100; 100 = identical, 0 = unrelated

    @property
    def display_text(self) -> str:
        """Best text to show as the 'changed excerpt' in the UI."""
        return self.new_text or self.old_text


def _label_for_similarity(sim: float) -> str:
    if sim >= SIM_HIGH:
        return "Unchanged"
    if sim >= SIM_MED:
        return "Small Change"
    if sim >= SIM_LOW:
        return "Medium Change"
    return "Big Change"


def diff_paragraphs(
    old_paras: list[Paragraph], new_paras: list[Paragraph]
) -> list[ParagraphChange]:
    """Compare two paragraph lists and return only the *changed* units.

    Algorithm:
      - For each new paragraph, find its best match in the old paragraphs
        by token_set_ratio. If the best match is >= SIM_HIGH, consider it
        unchanged. Otherwise, classify as Small/Medium/Big Change *unless*
        the similarity is so low that it's basically new (< SIM_LOW and
        the old paragraph it matched is short or unrelated -> "New").
      - After we walk new paragraphs, any old paragraph that was never
        the best match for any new paragraph with sim >= SIM_LOW is
        considered "Deleted".
    """
    changes: list[ParagraphChange] = []
    if not new_paras and not old_paras:
        return changes

    old_texts = [p.text for p in old_paras]
    matched_old_indices: set[int] = set()

    # Pre-compute lookup. rapidfuzz.process.extractOne is fast enough here.
    for new_p in new_paras:
        if not old_texts:
            changes.append(
                ParagraphChange("New", new_p.section, "", new_p.text, 0.0)
            )
            continue

        match = process.extractOne(
            new_p.text, old_texts, scorer=fuzz.token_set_ratio
        )
        # match is (best_text, score, index) or None
        if match is None:
            changes.append(
                ParagraphChange("New", new_p.section, "", new_p.text, 0.0)
            )
            continue

        best_text, score, idx = match
        if score >= SIM_HIGH:
            # Effectively unchanged -> we don't emit it.
            matched_old_indices.add(idx)
            continue

        if score < SIM_LOW:
            # Too dissimilar to call it a change; treat as truly New.
            changes.append(
                ParagraphChange("New", new_p.section, "", new_p.text, float(score))
            )
            continue

        # Small / Medium / Big change.
        matched_old_indices.add(idx)
        changes.append(
            ParagraphChange(
                change_type=_label_for_similarity(score),
                section=new_p.section,
                old_text=best_text,
                new_text=new_p.text,
                similarity=float(score),
            )
        )

    # Deleted paragraphs: old paragraphs no new paragraph claimed.
    # We are *very* conservative here. Most "deletions" are boilerplate
    # re-flow between filings (a paragraph moves, gets re-split, etc.)
    # and they swamp the output. We only emit a deletion if it is long
    # AND looks substantive (contains a topic-relevant keyword).
    for i, old_p in enumerate(old_paras):
        if i in matched_old_indices:
            continue
        if len(old_p.text) < 350:
            continue
        if not _looks_substantive(old_p.text):
            continue
        changes.append(
            ParagraphChange(
                change_type="Deleted",
                section=old_p.section,
                old_text=old_p.text,
                new_text="",
                similarity=0.0,
            )
        )

    return changes


# Tiny keyword check used to suppress noisy deletions. Kept local so we
# don't import analysis.py (which would create a cycle).
_DELETE_KEEP_WORDS = (
    "material weakness", "going concern", "covenant", "default",
    "impairment", "restructuring", "investigation", "subpoena",
    "lawsuit", "litigation", "guidance", "outlook", "customer concentration",
    "backlog", "liquidity", "revenue", "segment", "tariff", "dividend",
    "repurchase", "remediation", "ceo", "cfo", "resign", "appointed",
)


def _looks_substantive(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _DELETE_KEEP_WORDS)


# ---------------------------------------------------------------------------
# Numeric change extraction
#
# Given an old/new paragraph pair, find spans where numbers changed and
# return them as (old_chunk, new_chunk) tuples. The UI shows these as
# inFilings-style "X => Y" arrows above the redline excerpt, so an
# analyst can read the headline numeric deltas without parsing the prose.
# ---------------------------------------------------------------------------

import difflib as _difflib
import re as _re


# Match a "numeric token": dollar amount with optional unit, percent,
# ratio like "4.5 to 1.0", or a bare decimal. Used to compress redline
# replace-spans down to just the value that actually moved.
_NUMERIC_TOKEN_RE = _re.compile(
    r"""
    (?:\$\s*\(?-?[\d,]+(?:\.\d+)?\)?\s*(?:million|billion|thousand)?)   # $X.X million
  | (?:\(?-?[\d.]+\)?\s*%)                                              # 17.6%
  | (?:[\d.]+\s*(?:to|:)\s*[\d.]+)                                      # 4.5 to 1.0
  | (?:-?[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand))               # 84.4 million
  | (?:-?[\d]+\.[\d]+)                                                  # 4.5 (bare decimal)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)


def _first_numeric_phrase(text: str) -> str | None:
    """Pull just the numeric phrase out of a longer replace-span chunk."""
    m = _NUMERIC_TOKEN_RE.search(text)
    return m.group(0).strip() if m else None


def numeric_changes(old: str, new: str) -> list[tuple[str, str]]:
    """Return list of (old_value, new_value) text pairs where a number moved.

    Output is compact: each side is just the numeric phrase (e.g.
    "$5.2 million", "4.5 to 1.0", "91%") with surrounding context
    stripped, so the UI can show them as a row of "X => Y" chips.
    """
    if not old or not new:
        return []
    old_toks = old.split()
    new_toks = new.split()
    sm = _difflib.SequenceMatcher(a=old_toks, b=new_toks, autojunk=False)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        old_chunk = " ".join(old_toks[i1:i2])
        new_chunk = " ".join(new_toks[j1:j2])
        old_val = _first_numeric_phrase(old_chunk)
        new_val = _first_numeric_phrase(new_chunk)
        if not (old_val and new_val):
            continue
        if old_val == new_val:
            continue
        key = (old_val, new_val)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs[:8]
