"""
extract.py
----------
Download a filing's primary document and turn it into a clean list of
paragraph strings that the comparison and analysis modules can consume.

The MVP goal is "readable, comparable text" -- not perfect section parsing.
We do three things:
  1. Pull the primary .htm/.html document via edgar.http_get.
  2. Strip tags, scripts, style, and known-noisy elements with BeautifulSoup.
  3. Split on double newlines / large gaps to recover paragraph units.

We also do a *very* light section-heading sniff so each paragraph carries a
best-effort `section` label (e.g. "Risk Factors", "Liquidity"). This helps
the report group changes by section without requiring real XBRL parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

from edgar import Filing, http_get


# Headings we try to detect heuristically. Each entry is (canonical_name, regex).
# The regex is matched against an uppercased, whitespace-collapsed heading line.
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Risk Factors", re.compile(r"\bRISK FACTORS\b")),
    ("Legal Proceedings", re.compile(r"\bLEGAL PROCEEDINGS\b")),
    ("Management Discussion", re.compile(r"\bMANAGEMENT'S DISCUSSION\b|\bMD&A\b")),
    ("Liquidity", re.compile(r"\bLIQUIDITY\b")),
    ("Capital Resources", re.compile(r"\bCAPITAL RESOURCES\b")),
    ("Controls and Procedures", re.compile(r"\bCONTROLS AND PROCEDURES\b")),
    ("Revenue", re.compile(r"\bREVENUE\b(?! RECOGNITION)|\bNET SALES\b")),
    ("Revenue Recognition", re.compile(r"\bREVENUE RECOGNITION\b")),
    ("Segment Information", re.compile(r"\bSEGMENT\b")),
    ("Debt", re.compile(r"\bLONG-TERM DEBT\b|\bDEBT\b")),
    ("Related Party", re.compile(r"\bRELATED PART(Y|IES)\b")),
    ("Subsequent Events", re.compile(r"\bSUBSEQUENT EVENTS\b")),
    ("Income Taxes", re.compile(r"\bINCOME TAXES\b")),
    ("Stock Repurchases", re.compile(r"\bSHARE REPURCHASE\b|\bSTOCK REPURCHASE\b")),
    ("Going Concern", re.compile(r"\bGOING CONCERN\b")),
    ("Item 1.01", re.compile(r"\bITEM 1\.01\b")),  # 8-K event items
    ("Item 2.02", re.compile(r"\bITEM 2\.02\b")),
    ("Item 5.02", re.compile(r"\bITEM 5\.02\b")),
    ("Item 7.01", re.compile(r"\bITEM 7\.01\b")),
    ("Item 8.01", re.compile(r"\bITEM 8\.01\b")),
]


@dataclass
class Paragraph:
    """One paragraph of filing text plus a best-guess section label."""

    text: str
    section: str  # may be "Unknown" if we couldn't classify

    def __len__(self) -> int:
        return len(self.text)


def fetch_filing_text(filing: Filing) -> str:
    """Download the primary document and return the raw HTML (or text)."""
    resp = http_get(filing.primary_doc_url)
    # SEC sometimes serves filings as text/plain; BeautifulSoup handles both.
    return resp.text


def _looks_like_heading(line: str) -> bool:
    """Heuristic: short, mostly uppercase or title-cased, no terminal punct."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if s.endswith(":"):
        return True
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.7


def _classify_section(heading: str, current: str) -> str:
    """Map a heading line to a canonical section, or keep current section."""
    h = re.sub(r"\s+", " ", heading.upper())
    for name, pattern in _SECTION_PATTERNS:
        if pattern.search(h):
            return name
    return current


def html_to_paragraphs(html: str, min_chars: int = 60) -> list[Paragraph]:
    """Turn raw filing HTML into a clean list of Paragraph objects.

    Strategy:
      - Drop <script>, <style>, <table> (tables are noisy for paragraph diff).
      - Use BeautifulSoup's get_text with newlines as block separators.
      - Collapse whitespace inside each line.
      - Walk lines; if a line looks like a heading, update the section
        label; if it looks like a real paragraph, emit a Paragraph.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Tables are a big source of churn between filings (formatting tweaks,
    # spacing). Drop them for the MVP -- a more advanced version could
    # extract financial tables explicitly.
    for tag in soup(["table"]):
        tag.decompose()

    raw = soup.get_text(separator="\n")
    # Collapse whitespace within each candidate line.
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.split("\n")]

    paragraphs: list[Paragraph] = []
    current_section = "Unknown"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if len(text) >= min_chars:
            paragraphs.append(Paragraph(text=text, section=current_section))
        buffer.clear()

    for ln in lines:
        if not ln:
            flush()
            continue
        if _looks_like_heading(ln):
            flush()
            current_section = _classify_section(ln, current_section)
            continue
        buffer.append(ln)

    flush()
    return paragraphs


def extract_paragraphs(filing: Filing) -> list[Paragraph]:
    """Convenience: fetch + parse a Filing in one call."""
    html = fetch_filing_text(filing)
    return html_to_paragraphs(html)


def extract_filing(filing: Filing) -> tuple[list[Paragraph], str]:
    """Fetch a filing once and return (paragraphs, raw_html).

    The raw HTML is needed by `tables.py` to extract financial tables;
    `html_to_paragraphs` deliberately strips tables before producing
    paragraphs so the table data lives only in the raw HTML stream.
    """
    html = fetch_filing_text(filing)
    return html_to_paragraphs(html), html
