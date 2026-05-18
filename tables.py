"""
tables.py
---------
Extract revenue disaggregation tables from a filing's HTML.

Two structured views we care about:
  - Product / service breakdown (e.g. Apple's iPhone / Mac / iPad /
    Wearables / Services rows)
  - Geographic breakdown (Americas / Europe / Greater China / Japan / RoAP)

Strategy: walk every <table>, score it by how many of the expected row
labels appear in the first column, keep the rows we found per table,
then pick the best table per breakdown type. For each row we pull the
first two numeric cells -- typically "current period" and "prior period"
because SEC tables present comparatives side by side.

If we don't find a matching table, we return an empty list and the UI
shows "Not detected in this basic demo" instead of fabricating values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Label vocab
# ---------------------------------------------------------------------------

PRODUCT_LABELS = [
    "iPhone", "Mac", "iPad",
    "Wearables, Home and Accessories", "Wearables",
    "Services", "Products",
    # Generic fallbacks that other filers may use.
    "Software", "Hardware", "Subscription", "Licensing", "Advertising",
]

GEOGRAPHIC_LABELS = [
    "Americas", "Europe", "Greater China", "China",
    "Japan", "Rest of Asia Pacific", "Asia Pacific",
    "United States", "International",
    "North America", "EMEA", "APAC", "Latin America",
]

# Anchors that signal a revenue / income-statement-style table.
_TABLE_ANCHORS = ("net sales", "revenue", "net revenue", "total net sales")


@dataclass
class BreakdownRow:
    label: str
    latest_str: Optional[str]  # display string, e.g. "$39,668"
    prior_str: Optional[str]
    delta_pct: Optional[float]  # signed %, or None


# ---------------------------------------------------------------------------
# Cell parsing helpers
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(
    r"\(?\$?\s*\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?",
)


def _cell_text(c) -> str:
    return c.get_text(" ", strip=True).replace("\xa0", " ")


def _extract_number_display(text: str) -> Optional[str]:
    """Return the first numeric string in a cell, preserving format."""
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    s = m.group(0).strip()
    # Drop cells that are obviously bare punctuation or single-digit footnote refs.
    if len(re.sub(r"[^\d]", "", s)) < 2:
        return None
    return s


def _parse_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    neg = "(" in text and ")" in text
    n = re.sub(r"[\$\(\),\s]", "", text)
    try:
        v = float(n)
        return -v if neg else v
    except ValueError:
        return None


def _row_cells(tr) -> list[str]:
    return [_cell_text(c) for c in tr.find_all(["td", "th"])]


def _looks_like_revenue_table(table) -> bool:
    text = table.get_text(" ", strip=True).lower()
    return any(a in text for a in _TABLE_ANCHORS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_breakdown(html: str, labels: list[str]) -> list[BreakdownRow]:
    """Find the best table containing the given row labels and parse it.

    For each label we extract the *first* matching row, then pull the
    first two numeric cells as (latest, prior). The "best" table is the
    one with the most label matches.
    """
    soup = BeautifulSoup(html, "lxml")
    best_rows: list[BreakdownRow] = []
    best_score = 0

    for table in soup.find_all("table"):
        if not _looks_like_revenue_table(table):
            continue

        found: dict[str, BreakdownRow] = {}
        for tr in table.find_all("tr"):
            cells = [c for c in _row_cells(tr) if c]
            if len(cells) < 2:
                continue
            label_cell = cells[0]
            # Throw away cells that are pure currency signs or punctuation.
            value_cells = [c for c in cells[1:] if re.search(r"\d", c)]
            if not value_cells:
                continue

            for target in labels:
                if target.lower() not in label_cell.lower():
                    continue
                # Only the first match per label per table wins.
                if target in found:
                    break
                latest_str = _extract_number_display(value_cells[0]) if value_cells else None
                prior_str = (
                    _extract_number_display(value_cells[1])
                    if len(value_cells) > 1
                    else None
                )
                latest_num = _parse_number(latest_str)
                prior_num = _parse_number(prior_str)
                delta_pct = None
                if (
                    latest_num is not None
                    and prior_num is not None
                    and prior_num != 0
                ):
                    delta_pct = (latest_num - prior_num) / abs(prior_num) * 100.0
                found[target] = BreakdownRow(
                    label=target,
                    latest_str=latest_str,
                    prior_str=prior_str,
                    delta_pct=delta_pct,
                )
                break

        # Keep the table with the most label hits.
        if len(found) > best_score:
            best_score = len(found)
            best_rows = list(found.values())

    return best_rows


def extract_product_breakdown(html: str) -> list[BreakdownRow]:
    rows = extract_breakdown(html, PRODUCT_LABELS)
    # Deduplicate: keep the more specific label when we have overlap
    # (e.g. "Wearables" and "Wearables, Home and Accessories").
    by_norm: dict[str, BreakdownRow] = {}
    for r in rows:
        key = r.label.split(",")[0].strip().lower()
        existing = by_norm.get(key)
        if existing is None or len(r.label) > len(existing.label):
            by_norm[key] = r
    return list(by_norm.values())


def extract_geographic_breakdown(html: str) -> list[BreakdownRow]:
    rows = extract_breakdown(html, GEOGRAPHIC_LABELS)
    # Same dedupe trick for "China" vs "Greater China", etc.
    by_norm: dict[str, BreakdownRow] = {}
    for r in rows:
        key = r.label.lower().replace("greater ", "")
        existing = by_norm.get(key)
        if existing is None or len(r.label) > len(existing.label):
            by_norm[key] = r
    return list(by_norm.values())
