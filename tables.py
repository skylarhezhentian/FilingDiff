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
from collections import defaultdict
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


_PERIOD_PATTERNS = [
    ("three_months", re.compile(r"three[\s-]months?\s+(?:ended|period)", re.IGNORECASE)),
    ("six_months", re.compile(r"six[\s-]months?\s+(?:ended|period)", re.IGNORECASE)),
    ("nine_months", re.compile(r"nine[\s-]months?\s+(?:ended|period)", re.IGNORECASE)),
    ("year", re.compile(r"(?:twelve[\s-]months?|year\s+ended|fiscal\s+year|full[\s-]year)", re.IGNORECASE)),
    ("quarter", re.compile(r"quarter\s+ended", re.IGNORECASE)),
]
_CHANGE_HEADER_RE = re.compile(r"\bchange\b|^\s*\$\s+change\s*$|^\s*%\s*(?:change)?\s*$", re.IGNORECASE)
_DATE_HEADER_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[a-z]*\s+\d{1,2},?\s+\d{4}",
    re.IGNORECASE,
)


def _expand_row_cells(tr) -> list[tuple[str, object]]:
    """Walk a row and expand colspan into a flat list of (text, cell) tuples.

    The cell object is the original BeautifulSoup tag (we keep it so a
    later phase can read attributes). Each cell repeats `colspan` times
    so column indices line up across header and data rows.
    """
    out: list[tuple[str, object]] = []
    for c in tr.find_all(["td", "th"]):
        try:
            cs = int(c.get("colspan", "1") or "1")
        except ValueError:
            cs = 1
        text = _cell_text(c)
        for _ in range(max(1, cs)):
            out.append((text, c))
    return out


def _classify_columns(table) -> list[dict]:
    """Classify each column of `table` by aggregating its header cells.

    Returns a list of {period_type, is_change, date, raw} dicts, one per
    column. Headers are detected by looking at consecutive rows from the
    top that either contain no digits or have all-th cells; the first
    row with numeric data terminates header scanning.
    """
    header_rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        all_th = all(c.name == "th" for c in cells)
        has_number = any(re.search(r"\d", _cell_text(c)) for c in cells)
        has_date_in_header = any(_DATE_HEADER_RE.search(_cell_text(c)) for c in cells)
        if all_th or not has_number or has_date_in_header:
            header_rows.append(tr)
        else:
            break

    col_text: dict[int, list[str]] = defaultdict(list)
    for tr in header_rows:
        for idx, (text, _c) in enumerate(_expand_row_cells(tr)):
            if text:
                col_text[idx].append(text)

    if not col_text:
        return []

    out: list[dict] = []
    max_col = max(col_text.keys())
    for i in range(max_col + 1):
        joined = " | ".join(col_text.get(i, [])).strip()
        joined_l = joined.lower()
        period_type = None
        for pt_name, pt_re in _PERIOD_PATTERNS:
            if pt_re.search(joined):
                period_type = pt_name
                break
        is_change = bool(_CHANGE_HEADER_RE.search(joined))
        date_m = _DATE_HEADER_RE.search(joined)
        out.append({
            "period_type": period_type,
            "is_change": is_change,
            "date": date_m.group(0) if date_m else None,
            "raw": joined_l,
        })
    return out


def _identify_periods(col_info: list[dict]) -> list[dict]:
    """Group adjacent columns into period spans.

    A period span is a contiguous run of columns that share the same
    period_type (and date, when both are known). Change / % columns and
    unclassified columns break the run.
    """
    spans: list[dict] = []
    current: Optional[dict] = None
    for idx, info in enumerate(col_info):
        if not info["period_type"] or info["is_change"]:
            if current:
                spans.append(current)
                current = None
            continue
        same_period = (
            current
            and current["type"] == info["period_type"]
            and (current["date"] == info["date"] or info["date"] is None or current["date"] is None)
        )
        if same_period:
            current["cols"].append(idx)
            if info["date"] and not current["date"]:
                current["date"] = info["date"]
        else:
            if current:
                spans.append(current)
            current = {"type": info["period_type"], "date": info["date"], "cols": [idx]}
    if current:
        spans.append(current)
    return spans


def _pick_comparable_spans(spans: list[dict]) -> Optional[tuple[dict, dict]]:
    """Return (latest_span, prior_span) of the same period type, or None."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_type[s["type"]].append(s)
    for pt in ("three_months", "quarter", "six_months", "nine_months", "year"):
        cohort = by_type.get(pt, [])
        if len(cohort) < 2:
            continue
        # If both dates are known, the larger (later) date is latest.
        def _sortkey(span):
            return span["date"] or ""
        ordered = sorted(cohort, key=_sortkey, reverse=True)
        # Fall back to column order when dates are missing.
        if all(s["date"] for s in ordered):
            return ordered[0], ordered[1]
        ordered_by_col = sorted(cohort, key=lambda s: s["cols"][0])
        return ordered_by_col[0], ordered_by_col[1]
    return None


def _pick_value_cell(cells: list[tuple[str, object]], span: dict) -> Optional[str]:
    """Pick the numeric cell within a period span on a data row.

    A period span typically covers 2-4 cells: empty / $ spacer / number /
    optional empty. We pick the first cell within the span whose text
    contains digits beyond a footnote reference.
    """
    for idx in span["cols"]:
        if idx >= len(cells):
            continue
        text = cells[idx][0]
        n = _extract_number_display(text)
        if n is not None:
            return n
    return None


# Sanity gate: a "YoY" beyond this magnitude is almost certainly a
# mis-paired pair of cells. Drop the delta and surface the raw values
# only.
_MAX_PLAUSIBLE_YOY_PCT = 500.0


def _safe_delta(latest_num: Optional[float], prior_num: Optional[float]) -> Optional[float]:
    if latest_num is None or prior_num is None or prior_num == 0:
        return None
    delta = (latest_num - prior_num) / abs(prior_num) * 100.0
    if abs(delta) > _MAX_PLAUSIBLE_YOY_PCT:
        return None
    return delta


def extract_breakdown(html: str, labels: list[str]) -> list[BreakdownRow]:
    """Find the best table containing the given row labels and parse it.

    The parser is *header-aware*: it inspects the column headers, picks
    two columns of the same period type ("three months ended X" vs
    "three months ended Y"), and only computes YoY between those two
    paired values. If the column headers don't clearly identify two
    comparable periods, ``prior_str`` and ``delta_pct`` are left blank
    for that table -- the UI then notes that prior-period values were
    not confidently parsed.

    A sanity gate also drops any YoY computation whose magnitude is
    implausibly large (>500%), which catches the remaining cases where
    cell pairing is wrong despite a confident header read.
    """
    soup = BeautifulSoup(html, "lxml")
    best_rows: list[BreakdownRow] = []
    best_score = 0

    for table in soup.find_all("table"):
        if not _looks_like_revenue_table(table):
            continue

        col_info = _classify_columns(table)
        spans = _identify_periods(col_info)
        paired = _pick_comparable_spans(spans)

        found: dict[str, BreakdownRow] = {}
        for tr in table.find_all("tr"):
            expanded = _expand_row_cells(tr)
            texts = [t for t, _ in expanded]
            non_empty = [t for t in texts if t]
            if len(non_empty) < 2:
                continue
            label_cell = non_empty[0]

            for target in labels:
                if target.lower() not in label_cell.lower():
                    continue
                if target in found:
                    break

                latest_str: Optional[str] = None
                prior_str: Optional[str] = None
                if paired:
                    latest_span, prior_span = paired
                    latest_str = _pick_value_cell(expanded, latest_span)
                    prior_str = _pick_value_cell(expanded, prior_span)
                else:
                    # Fallback: take the first numeric cell only. We do
                    # NOT use the second numeric cell here because we
                    # cannot prove the two columns are comparable.
                    numeric_cells = [t for t in texts[1:] if _extract_number_display(t)]
                    if numeric_cells:
                        latest_str = _extract_number_display(numeric_cells[0])

                latest_num = _parse_number(latest_str)
                prior_num = _parse_number(prior_str)
                delta_pct = _safe_delta(latest_num, prior_num)
                found[target] = BreakdownRow(
                    label=target,
                    latest_str=latest_str,
                    prior_str=prior_str,
                    delta_pct=delta_pct,
                )
                break

        if len(found) > best_score:
            best_score = len(found)
            best_rows = list(found.values())

    return best_rows


def breakdown_has_confident_prior(rows: list[BreakdownRow]) -> bool:
    """Return True iff a majority of rows have a parsed prior value.

    The UI uses this to decide whether to surface "prior-period values
    were not confidently parsed" guidance under the table.
    """
    if not rows:
        return False
    with_prior = sum(1 for r in rows if r.prior_str)
    return with_prior >= max(1, len(rows) // 2 + 1)


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
