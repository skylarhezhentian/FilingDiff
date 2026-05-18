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
USER_AGENT = "FilingDiff Research Tool (educational use) skylarhtian@gmail.com"

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


# ---------------------------------------------------------------------------
# Filing document discovery (cover + exhibits)
# ---------------------------------------------------------------------------


@dataclass
class FilingDocument:
    """One file inside a filing (cover doc, exhibit, supporting doc)."""

    name: str  # filename inside the accession folder, e.g. "spire-ex99_1.htm"
    doc_type: str  # SEC document type label, e.g. "8-K", "EX-99.1", "EX-10.1"
    url: str  # absolute URL to fetch the document


def _archive_dir_url(cik: str, accession_nodash: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_nodash}"
    )


def get_filing_documents(cik: str, accession_no: str) -> list[FilingDocument]:
    """Return all documents in a filing by reading the EDGAR filing-index HTML.

    The bare ``index.json`` SEC exposes for an accession only carries
    icon-style ``type`` values like ``text.gif`` -- it does not include
    the SEC document type label (``EX-99.1`` etc.). The filing-index
    HTML page DOES have a real Type column, so we parse it instead.

    Returns an empty list on any fetch / parse error so the caller can
    fall back to using the primary document only.
    """
    accession_nodash = accession_no.replace("-", "")
    dir_url = _archive_dir_url(cik, accession_nodash)
    index_html_url = f"{dir_url}/{accession_no}-index.html"
    try:
        html = http_get(index_html_url).text
    except Exception:  # noqa: BLE001
        return []

    # Lazy-import BeautifulSoup so callers that don't need this path
    # don't pay the import cost.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    docs: list[FilingDocument] = []
    seen_names: set[str] = set()
    for table in soup.find_all("table"):
        header_cells = [c.get_text(strip=True).lower() for c in table.find_all("th")]
        if "type" not in header_cells or "document" not in header_cells:
            continue
        # Resolve column indices from the header row.
        col_idx = {name: i for i, name in enumerate(header_cells)}
        i_doc = col_idx["document"]
        i_type = col_idx["type"]
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) <= max(i_doc, i_type):
                continue
            doc_cell = cells[i_doc]
            type_cell = cells[i_type]
            doc_type = type_cell.get_text(strip=True)
            link = doc_cell.find("a")
            if not link:
                continue
            href = link.get("href") or ""
            # The link can be relative ("/Archives/...") or an iXBRL
            # viewer wrapper ("/ix?doc=/Archives/..."); strip the
            # wrapper and resolve to the raw file URL.
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            if href.startswith("/"):
                url = f"https://www.sec.gov{href}"
            else:
                url = href
            # Derive the filename for skip/dedup checks.
            name = url.rsplit("/", 1)[-1]
            if not name or name in seen_names:
                continue
            # Skip auto-generated index files.
            if name.endswith(("-index.htm", "-index.html", "-index-headers.html")):
                continue
            seen_names.add(name)
            docs.append(FilingDocument(name=name, doc_type=doc_type, url=url))
    return docs


# Exhibit types worth parsing for 8-K event intelligence, in priority
# order. Anything not matched here is skipped (XBRL schemas, cover-page
# interactive data, etc.).
EIGHT_K_EXHIBIT_PRIORITY: list[tuple[str, str]] = [
    ("EX-99.1", "Earnings release / press release / investor update"),
    ("EX-99.2", "Presentation or supplemental information"),
    ("EX-99",   "Additional exhibit content"),
    ("EX-10",   "Material agreement"),
    ("EX-2",    "Acquisition / merger agreement"),
    ("EX-4",    "Securities / debt instrument"),
    ("EX-1",    "Underwriting agreement"),
]


def select_8k_exhibits(docs: list[FilingDocument]) -> list[FilingDocument]:
    """Pick the substantive exhibits from an 8-K's document list, ordered by priority.

    Matching is prefix-based on the SEC document-type label so
    ``EX-99.1`` and ``EX-99.2`` both match the ``EX-99`` family but the
    more specific prefix wins. Each prefix may match multiple files
    (e.g. EX-10.1 and EX-10.2). Only documents whose name looks like
    parseable HTML/text are returned -- raw XBRL .xml or image files
    are excluded.
    """
    skip_ext = (".xml", ".xsd", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip")
    keepers: list[tuple[int, FilingDocument]] = []
    seen_urls: set[str] = set()
    for rank, (prefix, _label) in enumerate(EIGHT_K_EXHIBIT_PRIORITY):
        for d in docs:
            t = d.doc_type.upper()
            if not (t == prefix or t.startswith(prefix + ".") or t.startswith(prefix + "-")):
                continue
            if d.url in seen_urls:
                continue
            if d.name.lower().endswith(skip_ext):
                continue
            seen_urls.add(d.url)
            keepers.append((rank, d))
    keepers.sort(key=lambda x: x[0])
    return [d for _r, d in keepers]


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
