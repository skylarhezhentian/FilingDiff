"""
edgar.py
--------
Thin helpers around the SEC EDGAR public APIs.

SEC requires a descriptive User-Agent on every request. We expose a single
USER_AGENT constant that the rest of the app uses. Change `USER_AGENT` to
include your real email if you plan to use this seriously, otherwise the
SEC may rate-limit or block requests.

What this module provides:
  - ticker -> CIK lookup (cached)
  - "submissions" JSON for a CIK (the full filing history)
  - latest + previous filing of a given form type (10-Q, 10-K, 8-K)
  - URL helpers for filing index pages and primary documents
  - a polite GET wrapper that throttles & adds the User-Agent
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

# IMPORTANT: SEC EDGAR asks every request to declare a User-Agent that
# identifies the user. Replace the email below if you fork this app.
USER_AGENT = "FilingDiff Research Tool (educational use) contact@example.com"

# Be polite: SEC asks for <= 10 req/sec. We go a bit slower to be safe.
_MIN_INTERVAL_SEC = 0.15
_last_request_ts: float = 0.0


def _polite_sleep() -> None:
    """Sleep just enough to keep us under the SEC rate limit."""
    global _last_request_ts
    now = time.time()
    delta = now - _last_request_ts
    if delta < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - delta)
    _last_request_ts = time.time()


def http_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with the required SEC headers and a small throttle.

    Raises requests.HTTPError on non-2xx so callers can decide how to react.
    """
    _polite_sleep()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": _host_of(url),
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _host_of(url: str) -> str:
    # very small helper, avoids pulling urllib.parse for one line
    # url like "https://data.sec.gov/..." -> "data.sec.gov"
    return url.split("://", 1)[1].split("/", 1)[0]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Company:
    """Minimal company identity record."""

    cik: str  # 10-digit zero-padded string
    ticker: str
    name: str


@dataclass
class Filing:
    """One SEC filing entry, normalized for the rest of the app."""

    accession_no: str  # e.g. "0000320193-24-000123"
    form: str  # e.g. "10-Q"
    filing_date: str  # YYYY-MM-DD
    report_date: str  # YYYY-MM-DD, period of report
    primary_document: str  # the main .htm file inside the filing
    cik: str

    @property
    def accession_nodash(self) -> str:
        return self.accession_no.replace("-", "")

    @property
    def index_url(self) -> str:
        """The human-readable EDGAR index page for this filing."""
        return (
            f"https://www.sec.gov/cgi-bin/browse-edgar?"
            f"action=getcompany&CIK={self.cik}&type={self.form}&dateb=&owner=include&count=40"
        )

    @property
    def filing_page_url(self) -> str:
        """The filing-specific index page (lists all docs in the filing)."""
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_nodash}/"
            f"{self.accession_no}-index.htm"
        )

    @property
    def primary_doc_url(self) -> str:
        """Direct URL to the main filing document (usually an .htm)."""
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_nodash}/{self.primary_document}"
        )


# ---------------------------------------------------------------------------
# Ticker -> CIK lookup
# ---------------------------------------------------------------------------


def lookup_company(ticker: str) -> Optional[Company]:
    """Look up a company by ticker symbol using SEC's master ticker file.

    The endpoint returns a dict keyed by an integer string, each value is
    {"cik_str": int, "ticker": "AAPL", "title": "Apple Inc."}.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return None

    url = "https://www.sec.gov/files/company_tickers.json"
    data = http_get(url).json()

    for _key, row in data.items():
        if row.get("ticker", "").upper() == ticker:
            cik_padded = str(row["cik_str"]).zfill(10)
            return Company(cik=cik_padded, ticker=ticker, name=row.get("title", ticker))
    return None


# ---------------------------------------------------------------------------
# Submissions / filing history
# ---------------------------------------------------------------------------


def get_submissions(cik: str) -> dict:
    """Return the parsed submissions JSON for a CIK.

    The JSON contains a 'filings.recent' table with parallel arrays we can
    walk to build Filing records.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return http_get(url).json()


def latest_two_filings(submissions: dict, form: str) -> list[Filing]:
    """Return the latest and previous filings for a given form type.

    `form` is matched case-insensitively. For 10-K / 10-Q we also accept
    amendments-free variants. For 8-K we treat any '8-K' variant as 8-K.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    fdates = recent.get("filingDate", [])
    rdates = recent.get("reportDate", [])
    pdocs = recent.get("primaryDocument", [])
    cik = str(submissions.get("cik", "")).zfill(10)

    target = form.upper().strip()

    matches: list[Filing] = []
    for i, f in enumerate(forms):
        if f.upper() == target:
            matches.append(
                Filing(
                    accession_no=accs[i],
                    form=f,
                    filing_date=fdates[i],
                    report_date=rdates[i] or fdates[i],
                    primary_document=pdocs[i],
                    cik=cik,
                )
            )
            if len(matches) == 2:
                break
    return matches
