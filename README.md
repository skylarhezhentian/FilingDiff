# FilingDiff

This is a SEC filing insight Streamlit app for public equity research.

Enter a ticker. The app fetches the company's filing history, the SEC
XBRL companyfacts API, and the latest filing's HTML, then renders an
analyst-style report in three tabs: **10-Q**, **10-K**, **8-K**.

## What's in the report

Each tab presents (for 10-Q / 10-K):

1. **Header** — company name, ticker, CIK, filing dates, SEC links,
   overall change signal, key disclosure topics, filing timeliness.
2. **Executive Summary** — one specific paragraph stitched from XBRL
   deltas, parsed table deltas, and topic activity.
3. **Key Financials** — metric cards driven by SEC XBRL companyfacts
   (Revenue, Gross Profit, Operating Income, Net Income, Diluted EPS,
   ...). Each card shows current vs prior YoY.
4. **Segment / Product / Geographic Performance** — parsed directly
   from the filing's disaggregated revenue tables.
5. **Liquidity & Capital Allocation** — balance-sheet positions and
   capital-return cash flows (repurchases, dividends, capex, OCF).
6. **Risk / Legal / Controls** — compact red-flag findings + topic
   sentiment cards for these areas.
7. **Topic Sentiment Cards** — remaining active topics labeled
   Positive / Slightly Positive / Neutral / Slightly Negative /
   Negative / Mixed.

8-K tabs show the header, executive summary, Risk / Legal / Controls
findings, and topic sentiment cards. XBRL-driven sections are omitted
because 8-Ks don't carry quarterly financials.

Paragraph-level diffs are **not** the primary output. Every numbered
section above has an optional `Show paragraph evidence` expander that
reveals up to 5 relevant redlined excerpts as supporting evidence.

## Honest limits

- If an XBRL tag isn't filed by the issuer, the matching metric card
  is omitted — never fabricated.
- If a disaggregated-revenue table can't be matched against expected
  labels, the section says "Not detected in this basic demo".
- The 8-K comparison is inherently noisy because event types differ
  between filings — the tab carries a banner reminding users.
- Section parsing is heuristic. The "Section: ..." chip on evidence
  expanders is best-effort.

## Project layout

```
FilingDiff/
  app.py        Streamlit UI + tab orchestration
  edgar.py      SEC EDGAR helpers (ticker -> CIK, submissions, filings)
  extract.py    HTML -> paragraphs + raw HTML retained for tables.py
  tables.py     Disaggregated-revenue table parser (product / geographic)
  xbrl.py       SEC XBRL companyfacts fetcher + YoY comparator
  compare.py    Paragraph diff + numeric-arrow extraction
  redline.py    Inline word-level redline rendering (evidence expanders)
  analysis.py   Topic detection, sentiment, ranking, executive summary
  requirements.txt
  README.md
```

## Local setup

```bash
cd FilingDiff
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, enter a ticker like `AAPL` in the
sidebar, and click **Run analysis**.

> SEC requires a descriptive `User-Agent` on every request. Open
> `edgar.py` and replace the placeholder email in `USER_AGENT` with
> your own before extended use.

## Caching

- `cached_lookup_company` — ticker -> CIK, 1-hour TTL.
- `cached_submissions` — filing history, 30-minute TTL.
- `cached_company_facts` — XBRL companyfacts, 1-hour TTL.
- `cached_filing` — primary doc HTML + paragraphs, 1-hour TTL.

## Disclaimer

This output is AI-generated and rule-based. It may contain errors and
is for educational/research workflow purposes only, not investment
advice. Verify all conclusions against the original SEC filings.
