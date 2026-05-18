"""
app.py
------
FilingDiff: a polished SEC filing insight report.

Run locally with:

    pip install -r requirements.txt
    streamlit run app.py

Per-tab layout (10-Q / 10-K / 8-K):

  1. Header (company, ticker, form, dates, SEC links, signal pill,
     key topic chips).
  2. Executive summary -- a single specific paragraph stitched from
     XBRL deltas, parsed table deltas, and topic activity.
  3. Key Financials -- metric cards driven by SEC XBRL companyfacts
     (Revenue, Gross Profit, Operating Income, Net Income, Diluted EPS,
     Operating Cash Flow, ...). Cards show current vs prior YoY.
  4. Segment / Product / Geographic Performance -- parsed from the
     filing's revenue-disaggregation tables.
  5. Liquidity & Capital Allocation -- balance-sheet XBRL metrics
     (cash, investments, debt, equity) plus repurchases / dividends.
  6. Risk / Legal / Controls -- compact findings drawn from changed
     paragraphs that touch those topics, including any red-flag
     phrases that surfaced in the diff.
  7. Topic Sentiment Cards -- one card per active topic, with a
     Positive / Slightly Positive / Neutral / Slightly Negative /
     Negative / Mixed sentiment.

Paragraph-level diffs are *not* displayed by default. Every numbered
section above has an optional "Show paragraph evidence" expander that
reveals up to 5 relevant redlined excerpts -- supporting evidence
only, never the primary output.

If a metric / table / topic isn't detected, we say so explicitly. We
never fabricate a number.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from edgar import (
    Company,
    Filing,
    get_submissions,
    latest_two_filings,
    lookup_company,
)
from extract import extract_filing, Paragraph
from compare import diff_paragraphs, numeric_changes
from analysis import (
    RankedChange,
    RedFlagHit,
    ReportBundle,
    TopicCard,
    TopicEvidence,
    WatchItem,
    build_report,
    filter_changes_by_topics,
)
from xbrl import (
    BALANCE_METRICS,
    CASHFLOW_METRICS,
    INCOME_METRICS,
    XBRLMetric,
    fetch_metrics,
    format_delta,
    format_value,
    get_company_facts,
)
from tables import (
    BreakdownRow,
    extract_geographic_breakdown,
    extract_product_breakdown,
)
from narrative import (
    DriverNote,
    FallbackSegmentNote,
    GuidanceNote,
    extract_drivers,
    extract_fallback_segments,
    extract_guidance,
    segment_commentary,
)
from eightk import (
    EightKEvent,
    Highlight,
    KeyMetric,
    extract_8k_event,
)
from redline import redline_html, added_html, deleted_html


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FilingDiff — SEC Filing Insight Report",
    page_icon="📑",
    layout="wide",
)


st.markdown(
    """
    <style>
      .fd-signal {
        display:inline-block; padding:4px 10px; border-radius:999px;
        font-weight:600; font-size:0.85rem; margin-left:6px;
      }
      .fd-positive { background:#DCFCE7; color:#166534; }
      .fd-slightly-positive { background:#ECFCCB; color:#365314; }
      .fd-neutral  { background:#E5E7EB; color:#1F2937; }
      .fd-mixed    { background:#FEF3C7; color:#92400E; }
      .fd-slightly-negative { background:#FFE4E6; color:#9F1239; }
      .fd-negative { background:#FECACA; color:#7F1D1D; }

      .fd-card {
        border:1px solid #E5E7EB; border-radius:10px;
        padding:12px 14px; margin-bottom:10px; background:#FFFFFF;
      }
      .fd-card-title {
        font-weight:600; color:#111827; font-size:1rem; margin-bottom:4px;
      }
      .fd-card-sub {
        color:#6B7280; font-size:0.82rem; margin-bottom:6px;
      }
      .fd-tag {
        display:inline-block; font-size:0.72rem; font-weight:600;
        padding:1px 7px; border-radius:999px; margin-right:6px;
        color:#374151; background:#F3F4F6;
      }
      .fd-tag-new { background:#DCFCE7; color:#166534; }
      .fd-tag-del { background:#FEE2E2; color:#7F1D1D; }
      .fd-tag-big { background:#FEF3C7; color:#92400E; }
      .fd-tag-med { background:#E0E7FF; color:#3730A3; }
      .fd-tag-small { background:#F3F4F6; color:#374151; }
      .fd-table {
        border-collapse:collapse; width:100%; font-size:0.9rem;
      }
      .fd-table th, .fd-table td {
        padding:6px 10px; border-bottom:1px solid #F3F4F6; text-align:right;
      }
      .fd-table th:first-child, .fd-table td:first-child { text-align:left; }
      .fd-table th { background:#F9FAFB; color:#374151; font-weight:600; }
      .fd-table td.delta-up   { color:#166534; font-weight:600; }
      .fd-table td.delta-down { color:#7F1D1D; font-weight:600; }
      .fd-finding {
        border-left:3px solid #DC2626; background:#FEF2F2;
        padding:8px 12px; border-radius:6px; margin-bottom:6px;
      }
      .fd-evidence-redline {
        padding:8px 0; border-bottom:1px dashed #F3F4F6;
      }
      .fd-evidence-redline:last-child { border-bottom:none; }
      .fd-arrows {
        margin:6px 0 4px 0; padding:6px 10px;
        background:#F8FAFC; border-radius:6px; border:1px solid #E2E8F0;
        font-size:0.85rem;
      }
      .fd-arrow {
        display:inline-block; margin-right:14px; white-space:nowrap;
      }
      .fd-arrow-old { color:#7F1D1D; text-decoration:line-through; }
      .fd-arrow-new { color:#166534; font-weight:600; }
      .fd-arrow-glyph { color:#6B7280; font-weight:700; }

      .fd-why {
        margin-top:8px; padding:6px 10px;
        background:#FFFBEB; border-left:3px solid #F59E0B;
        font-size:0.85rem; color:#374151; border-radius:4px;
      }
      .fd-numrow {
        margin-top:8px; display:flex; flex-wrap:wrap; gap:6px;
      }
      .fd-numchip {
        background:#EFF6FF; color:#1E3A8A; font-size:0.82rem;
        padding:3px 8px; border-radius:999px; white-space:nowrap;
      }
      .fd-driver {
        border-left:3px solid #2563EB; background:#EFF6FF;
        padding:10px 14px; border-radius:6px; margin-bottom:10px;
      }
      .fd-driver-cat {
        font-weight:600; color:#1E3A8A; font-size:0.85rem;
        margin-bottom:4px; display:flex; justify-content:space-between;
        align-items:baseline;
      }
      .fd-driver-conf {
        font-weight:500; font-size:0.74rem; padding:2px 8px;
        border-radius:999px; white-space:nowrap;
      }
      .fd-driver-conf-high   { color:#065F46; background:#D1FAE5; }
      .fd-driver-conf-medium { color:#92400E; background:#FEF3C7; }
      .fd-driver-conf-low    { color:#374151; background:#E5E7EB; }
      .fd-driver-headline {
        color:#111827; font-size:0.95rem; font-weight:600;
        margin:4px 0 6px 0;
      }
      .fd-driver-bullets {
        margin:4px 0 0 0; padding-left:18px;
        color:#374151; font-size:0.88rem;
      }
      .fd-driver-bullets li { margin-bottom:3px; }
      .fd-driver-bullets li strong { color:#1E3A8A; }
      .fd-guidance {
        border-left:3px solid #059669; background:#ECFDF5;
        padding:8px 12px; border-radius:6px; margin-bottom:8px;
      }
      .fd-guidance-cat {
        font-weight:600; color:#065F46; font-size:0.85rem;
        margin-bottom:4px;
      }
      .fd-segcomm {
        background:#F9FAFB; border-radius:6px; padding:6px 10px;
        margin-top:4px; font-size:0.88rem; color:#374151;
      }
      .fd-8k-header {
        background:#F0F9FF; border:1px solid #BAE6FD;
        border-radius:6px; padding:12px 14px; margin-bottom:10px;
      }
      .fd-8k-event-type {
        font-weight:700; color:#075985; font-size:1.05rem;
      }
      .fd-8k-meta {
        display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;
      }
      .fd-8k-chip {
        background:#E0F2FE; color:#075985; font-size:0.78rem;
        padding:3px 9px; border-radius:999px; white-space:nowrap;
      }
      .fd-8k-mat-high   { background:#FECACA; color:#7F1D1D; }
      .fd-8k-mat-medium { background:#FEF3C7; color:#92400E; }
      .fd-8k-mat-low    { background:#E5E7EB; color:#374151; }
      .fd-8k-metric {
        background:#FFFFFF; border:1px solid #E5E7EB;
        border-radius:6px; padding:8px 12px; margin-bottom:6px;
      }
      .fd-8k-metric-label {
        font-size:0.78rem; color:#6B7280; text-transform:uppercase;
        letter-spacing:0.05em;
      }
      .fd-8k-metric-value {
        font-size:1.1rem; font-weight:600; color:#111827;
        margin-top:2px;
      }
      .fd-8k-metric-delta {
        font-size:0.85rem; color:#6B7280; margin-left:6px;
      }
      .fd-8k-highlight {
        border-left:3px solid #0EA5E9; background:#F0F9FF;
        padding:8px 12px; border-radius:6px; margin-bottom:8px;
      }
      .fd-8k-highlight-title {
        font-weight:600; color:#075985; font-size:0.88rem;
        margin-bottom:3px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached wrappers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_lookup_company(ticker: str) -> Optional[dict]:
    c = lookup_company(ticker)
    if c is None:
        return None
    return {"cik": c.cik, "ticker": c.ticker, "name": c.name}


@st.cache_data(show_spinner=False, ttl=30 * 60)
def cached_submissions(cik: str) -> dict:
    return get_submissions(cik)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_company_facts(cik: str) -> Optional[dict]:
    return get_company_facts(cik)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_filing(
    cik: str, accession_no: str, primary_document: str, form: str,
    filing_date: str, report_date: str,
) -> tuple[list[dict], str]:
    """Return (paragraph_dicts, raw_html) for a single filing."""
    f = Filing(
        accession_no=accession_no, form=form, filing_date=filing_date,
        report_date=report_date, primary_document=primary_document, cik=cik,
    )
    paras, html = extract_filing(f)
    return [{"text": p.text, "section": p.section} for p in paras], html


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_filing_documents(cik: str, accession_no: str) -> list[dict]:
    """List every file inside a filing's accession folder, with SEC type label."""
    from edgar import get_filing_documents
    docs = get_filing_documents(cik, accession_no)
    return [
        {"name": d.name, "doc_type": d.doc_type, "url": d.url}
        for d in docs
    ]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_exhibit_paragraphs(url: str) -> tuple[list[dict], int]:
    """Fetch and parse one exhibit document into (paragraph_dicts, char_count).

    Uses the press-release extractor (table-preserving, bullet-aware)
    because most 8-K exhibits are earnings releases or marketing-style
    business updates whose data lives in compact inline tables and
    bullet lists rather than long prose paragraphs.

    Returns ([], 0) on any fetch / parse error so the caller can surface
    a "could not be parsed" status without crashing the tab.
    """
    from edgar import http_get
    from extract import html_to_press_release_text
    try:
        resp = http_get(url)
        html = resp.text
    except Exception:  # noqa: BLE001
        return [], 0
    if not html:
        return [], 0
    try:
        paras = html_to_press_release_text(html)
    except Exception:  # noqa: BLE001
        return [], len(html)
    return (
        [{"text": p.text, "section": p.section} for p in paras],
        len(html),
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📑 FilingDiff")
    st.caption("SEC filing insight reports for equity research.")
    ticker_input = st.text_input(
        "Ticker", value="AAPL",
        help="Public US equity ticker, e.g. AAPL, MSFT, NVDA.",
    ).strip().upper()
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)
    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- SEC EDGAR submissions + filing HTML\n"
        "- SEC XBRL companyfacts API\n"
        "- Disaggregated revenue tables parsed from the filing HTML\n"
        "\n"
        "Paragraph-level diffs are kept as optional evidence inside expanders."
    )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _signal_pill(label: str) -> str:
    cls_map = {
        "Positive": "fd-positive",
        "Slightly Positive": "fd-slightly-positive",
        "Neutral": "fd-neutral",
        "Mixed": "fd-mixed",
        "Slightly Negative": "fd-slightly-negative",
        "Negative": "fd-negative",
    }
    return f'<span class="fd-signal {cls_map.get(label, "fd-neutral")}">{label}</span>'


def _change_tag(change_type: str) -> str:
    cls = {
        "New": "fd-tag-new",
        "Deleted": "fd-tag-del",
        "Big Change": "fd-tag-big",
        "Medium Change": "fd-tag-med",
        "Small Change": "fd-tag-small",
    }.get(change_type, "")
    return f'<span class="fd-tag {cls}">{change_type}</span>'


def _render_metrics_grid(metrics: list[XBRLMetric]) -> None:
    """Render a row of `st.metric` cards. Skip metrics with no values."""
    visible = [m for m in metrics if m.latest_val is not None]
    if not visible:
        st.info(
            "Not detected in this basic demo: SEC XBRL companyfacts did not "
            "return values for the requested tags."
        )
        return
    for i in range(0, len(visible), 4):
        row = visible[i : i + 4]
        cols = st.columns(len(row))
        for col, m in zip(cols, row):
            with col:
                help_lines = [f"XBRL tag: `{m.tag}`"]
                if m.latest_end:
                    help_lines.append(f"Latest period end: {m.latest_end}")
                if m.prior_val is not None and m.prior_end:
                    help_lines.append(
                        f"Prior: {format_value(m.prior_val, m.unit)} ({m.prior_end})"
                    )
                st.metric(
                    label=m.label,
                    value=format_value(m.latest_val, m.unit),
                    delta=format_delta(m.delta_pct),
                    help="\n\n".join(help_lines),
                )


def _render_breakdown_table(rows: list[BreakdownRow], title: str) -> None:
    if not rows:
        st.markdown(
            f"_**{title}:** Not detected in this basic demo. The filing may "
            "use a non-standard label set; try opening the SEC document directly._"
        )
        return

    # Confidence gate: if more than half the rows have no parsed prior,
    # we render the latest column only and surface a caveat. This is the
    # conservative path -- better to say "not confidently parsed" than
    # to compute YoY from mis-paired cells.
    from tables import breakdown_has_confident_prior  # local to avoid cycle
    confident = breakdown_has_confident_prior(rows)

    if confident:
        html = [
            f'<table class="fd-table"><thead><tr>'
            f'<th>{title}</th><th>Latest</th><th>Prior</th><th>YoY</th></tr></thead><tbody>'
        ]
        for r in rows:
            if r.delta_pct is None:
                delta_cell = "—"
                cls = ""
            else:
                delta_cell = f"{r.delta_pct:+.1f}%"
                cls = "delta-up" if r.delta_pct >= 0 else "delta-down"
            html.append(
                f"<tr><td>{r.label}</td>"
                f"<td>{r.latest_str or '—'}</td>"
                f"<td>{r.prior_str or '—'}</td>"
                f'<td class="{cls}">{delta_cell}</td></tr>'
            )
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)
    else:
        html = [
            f'<table class="fd-table"><thead><tr>'
            f'<th>{title}</th><th>Latest</th></tr></thead><tbody>'
        ]
        for r in rows:
            html.append(
                f"<tr><td>{r.label}</td>"
                f"<td>{r.latest_str or '—'}</td></tr>"
            )
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)
        st.caption(
            "Prior-period values were not confidently parsed for this "
            "table — YoY suppressed to avoid mispaired comparisons. See "
            "the source filing for the as-disclosed comparatives."
        )


def _render_evidence(ranked: list[RankedChange], wanted_topics: set[str]) -> None:
    """Render an evidence expander for a section (collapsed by default)."""
    items = filter_changes_by_topics(ranked, wanted_topics, limit=5)
    with st.expander("Show source evidence", expanded=False):
        if not items:
            st.write(
                "_The relevant filing prose did not change materially "
                "versus the prior comparable filing._"
            )
            return
        for rc in items:
            _render_evidence_change(rc)


def _render_evidence_change(rc: RankedChange) -> None:
    ch = rc.change
    arrows_html = ""
    if ch.old_text and ch.new_text:
        pairs = numeric_changes(ch.old_text, ch.new_text)
        if pairs:
            items = "".join(
                f'<span class="fd-arrow">'
                f'<span class="fd-arrow-old">{old}</span> '
                f'<span class="fd-arrow-glyph">=&gt;</span> '
                f'<span class="fd-arrow-new">{new}</span>'
                f"</span>"
                for old, new in pairs
            )
            arrows_html = f'<div class="fd-arrows">{items}</div>'

    if ch.change_type == "New":
        body = added_html(ch.new_text)
    elif ch.change_type == "Deleted":
        body = deleted_html(ch.old_text)
    else:
        body = redline_html(ch.old_text, ch.new_text)

    section_chip = (
        f'<span class="fd-tag">Section: {ch.section}</span>'
        if ch.section != "Unknown"
        else ""
    )
    st.markdown(
        f'<div class="fd-evidence-redline">'
        f"{_change_tag(ch.change_type)}{section_chip}"
        f"{arrows_html}"
        f'<div style="margin-top:6px">{body}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_topic_card(card: TopicCard) -> None:
    cls = {
        "Positive": "fd-positive",
        "Slightly Positive": "fd-slightly-positive",
        "Neutral": "fd-neutral",
        "Mixed": "fd-mixed",
        "Slightly Negative": "fd-slightly-negative",
        "Negative": "fd-negative",
    }.get(card.sentiment, "fd-neutral")
    # Render the numbers as a compact row of pill-style chips, when any.
    nums_html = ""
    if card.numbers:
        chips = []
        for n in card.numbers[:5]:
            label = n.label
            value = n.value
            delta = f" {n.delta}" if n.delta else ""
            chips.append(
                f'<span class="fd-numchip"><b>{label}:</b> {value}{delta}</span>'
            )
        nums_html = '<div class="fd-numrow">' + " ".join(chips) + "</div>"

    why_html = (
        f'<div class="fd-why"><b>Why it matters:</b> {card.why_it_matters}</div>'
        if card.why_it_matters
        else ""
    )
    st.markdown(
        f"""
        <div class="fd-card">
          <div class="fd-card-title">{card.topic}
            <span class="fd-signal {cls}">{card.sentiment}</span>
          </div>
          <div>{card.narrative}</div>
          {nums_html}
          {why_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cautionary_topic_card(card: TopicCard) -> None:
    """Render a risk-severe topic card with evidence breakdown.

    Used for Risk Factors / Legal Proceedings / Controls and
    Procedures. Shows What changed / Why negative / Evidence type /
    Confidence so a negative label is never unexplained. Cards that
    were downgraded from Negative to Neutral due to insufficient
    evidence render with an honest "no material new ... language
    detected" line.
    """
    cls = {
        "Positive": "fd-positive",
        "Slightly Positive": "fd-slightly-positive",
        "Neutral": "fd-neutral",
        "Mixed": "fd-mixed",
        "Slightly Negative": "fd-slightly-negative",
        "Negative": "fd-negative",
    }.get(card.sentiment, "fd-neutral")

    # Build the evidence block (if any).
    evidence_html = ""
    if card.downgraded_from:
        evidence_html = (
            '<ul class="fd-driver-bullets">'
            f'<li><strong>Status:</strong> downgraded from {card.downgraded_from}; '
            f"no specific change found.</li>"
            '<li><strong>Confidence:</strong> low (no event or material '
            "language change to anchor the label).</li>"
            "</ul>"
        )
    elif card.evidence:
        ev = card.evidence
        type_label = {
            "event": "Actual event",
            "cautionary": "Cautionary language",
            "boilerplate": "Boilerplate / definition only",
            "insufficient": "Insufficient evidence",
        }.get(ev.evidence_type, ev.evidence_type)
        evidence_html = (
            '<ul class="fd-driver-bullets">'
            f"<li><strong>What changed:</strong> {ev.what_changed}</li>"
            f"<li><strong>Why the sentiment is negative:</strong> {ev.why_negative}</li>"
            f"<li><strong>Evidence type:</strong> {type_label}</li>"
            f"<li><strong>Confidence:</strong> {ev.confidence}</li>"
            "</ul>"
        )

    why_html = (
        f'<div class="fd-why"><b>Why it matters:</b> {card.why_it_matters}</div>'
        if card.why_it_matters else ""
    )
    st.markdown(
        f"""
        <div class="fd-card">
          <div class="fd-card-title">{card.topic}
            <span class="fd-signal {cls}">{card.sentiment}</span>
          </div>
          <div>{card.narrative}</div>
          {evidence_html}
          {why_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if card.evidence and card.evidence.excerpt:
        with st.expander(f"Source evidence for {card.topic}", expanded=False):
            st.markdown(f"> \"{card.evidence.excerpt}\"")


def _render_drivers(drivers: list[DriverNote]) -> None:
    """Render Business Drivers as structured analyst cards.

    Each card carries: category label, one-sentence directional headline,
    Driver / Financial impact / Investment read bullets, and a confidence
    chip. Raw filing text lives only inside the "Source evidence"
    expander.
    """
    if not drivers:
        st.write(
            "_No explicit business-driver language detected in the filing._"
        )
        return

    conf_labels = {
        "high": "High confidence",
        "medium": "Medium confidence",
        "low": "Low confidence",
    }

    for d in drivers:
        conf_cls = f"fd-driver-conf-{d.confidence}" if d.confidence in conf_labels else "fd-driver-conf-low"
        conf_text = conf_labels.get(d.confidence, "")
        bullets: list[str] = []
        if d.driver:
            bullets.append(f"<li><strong>Driver:</strong> {d.driver}</li>")
        if d.financial_impact:
            bullets.append(f"<li><strong>Financial impact:</strong> {d.financial_impact}</li>")
        if d.investment_read:
            bullets.append(f"<li><strong>Investment read:</strong> {d.investment_read}</li>")
        bullets_html = "".join(bullets)
        st.markdown(
            f"""
            <div class="fd-driver">
              <div class="fd-driver-cat">
                <span>{d.category}</span>
                <span class="fd-driver-conf {conf_cls}">{conf_text}</span>
              </div>
              <div class="fd-driver-headline">{d.headline}</div>
              {f'<ul class="fd-driver-bullets">{bullets_html}</ul>' if bullets_html else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if any(d.excerpts for d in drivers):
        with st.expander("Source evidence", expanded=False):
            for d in drivers:
                if not d.excerpts:
                    continue
                st.markdown(f"**{d.category}**")
                for ex in d.excerpts:
                    st.markdown(f"> \"{ex}\"")


def _render_guidance(notes: list[GuidanceNote]) -> None:
    """Render Guidance / Outlook as polished bullets; raw quotes in expander."""
    if not notes:
        st.write("**No specific quantitative guidance detected in the filing.**")
        return
    for n in notes[:8]:
        st.markdown(
            f"""
            <div class="fd-guidance">
              <div class="fd-guidance-cat">{n.metric_hint}</div>
              <div>{n.headline or n.excerpt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Show evidence", expanded=False):
        for n in notes[:8]:
            st.markdown(f"**{n.metric_hint}** — \"{n.excerpt}\"")


def _render_fallback_segments(notes: list[FallbackSegmentNote]) -> None:
    """Render the MD&A fallback when structured tables didn't return rows."""
    if not notes:
        return
    st.caption(
        "Structured table parsing did not return matched rows. The "
        "headlines below were synthesised from MD&A text."
    )
    for n in notes[:8]:
        st.markdown(
            f"""
            <div class="fd-driver">
              <div class="fd-driver-cat">{n.name}</div>
              <div>{n.headline}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Show evidence", expanded=False):
        for n in notes[:8]:
            st.markdown(f"**{n.name}** — \"{n.excerpt}\"")


def _render_breakdown_with_commentary(
    title: str, rows: list[BreakdownRow], commentary: dict[str, str]
) -> None:
    """Show a breakdown table; for each row with commentary, quote it underneath."""
    _render_breakdown_table(rows, title)
    if not rows or not commentary:
        return
    bullets = []
    for r in rows:
        # Try exact label, then a prefix match (for 'Wearables, Home and Accessories' etc.).
        quote = commentary.get(r.label)
        if not quote:
            for k, v in commentary.items():
                if k.lower().startswith(r.label.lower()[:8]) or r.label.lower().startswith(k.lower()[:8]):
                    quote = v
                    break
        if quote:
            bullets.append((r.label, quote))
    if bullets:
        st.markdown("**Per-line color (from MD&A):**")
        for label, quote in bullets:
            st.markdown(
                f'<div class="fd-segcomm"><b>{label}:</b> "{quote}"</div>',
                unsafe_allow_html=True,
            )


def _render_severe_red_flags(items: list[WatchItem]) -> None:
    """Render the **Severe Red Flags** subsection only.

    Severe red flags mean actual events (going concern, covenant
    breach, default, material weakness, investigation, impairment
    charge, litigation update). Topic sentiment is rendered separately
    by ``_render_cautionary_topic_cards``.

    Resolved items and non-findings (negated / boilerplate / weak) are
    rendered by their own helpers so each subsection stands alone.
    """
    true_flags = [w for w in items if w.classification == "true"]
    ongoing = [w for w in items if w.classification == "ongoing"]

    if not true_flags and not ongoing:
        st.success(
            "No severe red-flag events detected in changed disclosures."
        )
        return

    for h in true_flags:
        change_chip = (
            f" — {h.change_type}" if h.change_type else " — present in filing"
        )
        st.markdown(
            f"""
            <div class="fd-finding">
              <b>⚠️ True red flag: {h.phrase.title()}</b>
              <span style="color:#6B7280">{change_chip} · Section: {h.section}</span>
              <div style="margin-top:4px;font-size:0.82rem;color:#374151">
                <i>Why it matters:</i> {h.why_it_matters}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for h in ongoing:
        st.markdown(
            f"""
            <div class="fd-finding" style="border-left-color:#D97706;background:#FEF3C7">
              <b>Ongoing known issue: {h.phrase.title()}</b>
              <span style="color:#6B7280"> — present in filing · Section: {h.section}</span>
              <div style="margin-top:4px;font-size:0.82rem;color:#374151">
                <i>Why it matters:</i> {h.why_it_matters}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Show evidence", expanded=False):
        for h in true_flags + ongoing:
            st.markdown(f"**{h.phrase.title()}** — \"{h.excerpt}\"")


def _render_resolved_risks(items: list[WatchItem]) -> None:
    """Render the **Resolved / Removed Prior Risks** subsection."""
    resolved = [w for w in items if w.classification == "resolved"]
    if not resolved:
        return
    st.markdown(
        ", ".join(sorted({w.phrase.title() for w in resolved}))
    )


def _render_non_findings(items: list[WatchItem]) -> None:
    """Render the collapsed non-findings panel (negated / boilerplate / weak)."""
    negated = [w for w in items if w.classification == "negated"]
    boilerplate = [w for w in items if w.classification == "boilerplate"]
    weak = [w for w in items if w.classification == "weak"]
    buried = negated + boilerplate + weak
    if not buried:
        return
    with st.expander(
        f"Show non-findings ({len(buried)}): negated · boilerplate · weak evidence",
        expanded=False,
    ):
        for w in negated:
            st.markdown(
                f"- *Mentioned but negated:* **{w.phrase.title()}** — the filing "
                "mentions this phrase but explicitly negates it (e.g. \"None\" / \"Not applicable\")."
            )
        for w in boilerplate:
            st.markdown(
                f"- *Boilerplate / definition only:* **{w.phrase.title()}** — "
                "appears inside a non-GAAP definition, a recurring SEC header "
                "with \"None\", or a hypothetical contractual clause."
            )
        for w in weak:
            st.markdown(
                f"- *Weak evidence:* **{w.phrase.title()}** — keyword present "
                "but no event-confirming context (e.g. \"notice of default\", "
                "\"restructuring charges of $X million\"). Read the source."
            )


# ---------------------------------------------------------------------------
# 8-K event-intelligence tab
# ---------------------------------------------------------------------------


def _render_8k_header(ev: EightKEvent) -> None:
    """Top banner: event type + item / exhibit / materiality chips."""
    mat_cls = f"fd-8k-mat-{ev.materiality}"
    mat_label = {"high": "High materiality", "medium": "Medium materiality", "low": "Low materiality"}.get(
        ev.materiality, ev.materiality.title()
    )
    chips: list[str] = []
    if ev.item_numbers:
        for i in ev.item_numbers:
            chips.append(f'<span class="fd-8k-chip">Item {i}</span>')
    else:
        chips.append('<span class="fd-8k-chip fd-8k-mat-low">No item detected</span>')
    if ev.exhibits:
        for e in ev.exhibits:
            chips.append(f'<span class="fd-8k-chip">Exhibit {e}</span>')
    chips.append(f'<span class="fd-8k-chip {mat_cls}">{mat_label}</span>')

    st.markdown(
        f"""
        <div class="fd-8k-header">
          <div class="fd-8k-event-type">{ev.event_type}</div>
          <div style="margin-top:4px;color:#0C4A6E;font-size:0.92rem;">
            <b>Overall read:</b> {ev.overall_read}
          </div>
          <div class="fd-8k-meta">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ev.item_descriptions:
        with st.expander("Item detail", expanded=False):
            for d in ev.item_descriptions:
                st.markdown(f"- {d}")


def _render_8k_metrics(metrics: list[KeyMetric]) -> None:
    if not metrics:
        st.write(
            "_No press-release-style headline metrics detected. For an "
            "Item 2.02 filing the press release often lives in a separate "
            "Exhibit 99.1 document; open the SEC link above to view the full "
            "earnings release._"
        )
        return
    n_cols = min(3, len(metrics))
    cols = st.columns(n_cols)
    for i, m in enumerate(metrics):
        with cols[i % n_cols]:
            delta_html = (
                f'<span class="fd-8k-metric-delta">{m.delta}</span>'
                if m.delta else ""
            )
            st.markdown(
                f"""
                <div class="fd-8k-metric">
                  <div class="fd-8k-metric-label">{m.label}</div>
                  <div class="fd-8k-metric-value">{m.value}{delta_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_8k_highlights(highlights: list[Highlight]) -> None:
    if not highlights:
        st.write(
            "_No customer / contract highlights detected in the primary "
            "document. Press-release exhibits typically carry these in a "
            "bullet list — see the source filing._"
        )
        return
    for h in highlights:
        st.markdown(
            f"""
            <div class="fd-8k-highlight">
              <div class="fd-8k-highlight-title">{h.title}</div>
              <div>{h.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_8k_tab(
    company: Company,
    form: str,
    filings: list[Filing],
    note: Optional[str] = None,
) -> None:
    """Dedicated 8-K event-intelligence tab.

    Unlike the 10-Q / 10-K layout, the 8-K tab is built around the
    single event the company disclosed, not period-over-period
    comparatives. For an Item 2.02 earnings 8-K the substantive content
    lives in Exhibit 99.1 -- the cover document is just a wrapper that
    declares the items and references the exhibit -- so this tab
    downloads and parses the relevant exhibits before running the
    metric / highlight extractors.
    """
    from edgar import select_8k_exhibits, FilingDocument
    from eightk import ExhibitParseStatus

    if not filings:
        st.warning(
            f"No recent 8-K filings found for {company.name} ({company.ticker})."
        )
        return

    latest = filings[0]

    st.subheader(f"{company.name} ({company.ticker}) — 8-K")
    st.markdown(
        f"**Latest 8-K**  \n"
        f"Filed: `{latest.filing_date}` · Reports period: `{latest.report_date}`  \n"
        f"[Open on SEC EDGAR]({latest.primary_doc_url})"
    )
    st.caption(f"CIK: {company.cik}")
    if note:
        st.info(note)

    with st.spinner("Downloading 8-K cover document…"):
        try:
            cover_paras_raw, cover_html = cached_filing(
                company.cik, latest.accession_no, latest.primary_document,
                latest.form, latest.filing_date, latest.report_date,
            )
            primary_parsed = True
            primary_char_count = len(cover_html or "")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to fetch or parse 8-K cover: {exc}")
            return

    cover_paras = [Paragraph(**p) for p in cover_paras_raw]

    # --- Discover exhibits and download the substantive ones -----------
    with st.spinner("Looking for exhibits (EX-99.1, EX-99.2, EX-10, ...)…"):
        try:
            docs_raw = cached_filing_documents(company.cik, latest.accession_no)
        except Exception:  # noqa: BLE001
            docs_raw = []
    docs = [FilingDocument(**d) for d in docs_raw]
    targeted_exhibits = select_8k_exhibits(docs)

    exhibit_paragraphs: list[Paragraph] = []
    exhibits_status: list[ExhibitParseStatus] = []
    if targeted_exhibits:
        with st.spinner(f"Downloading {len(targeted_exhibits)} exhibit(s)…"):
            for d in targeted_exhibits:
                paras_raw, char_count = cached_exhibit_paragraphs(d.url)
                parsed = bool(paras_raw)
                exhibits_status.append(
                    ExhibitParseStatus(
                        doc_type=d.doc_type,
                        name=d.name,
                        url=d.url,
                        parsed=parsed,
                        char_count=char_count,
                        note=None if parsed else "fetched but produced no paragraphs",
                    )
                )
                if parsed:
                    exhibit_paragraphs.extend(Paragraph(**p) for p in paras_raw)

    # Guidance is best detected over the union of cover + exhibits.
    guidance = extract_guidance(cover_paras + exhibit_paragraphs)
    event = extract_8k_event(
        cover_paras,
        exhibit_paragraphs=exhibit_paragraphs,
        guidance_notes=guidance,
        issuer_name=company.name,
        exhibits_status=exhibits_status,
        primary_parsed=primary_parsed,
        primary_char_count=primary_char_count,
    )

    st.divider()

    # ----- A. Header -----------------------------------------------------
    _render_8k_header(event)

    # ----- B. Event Classification (item-level detail) -------------------
    st.markdown("### Event Classification")
    cols = st.columns(2)
    with cols[0]:
        if event.item_numbers:
            for d in event.item_descriptions:
                st.markdown(f"- {d}")
        else:
            st.markdown("- _No item numbers detected in cover document_")
    with cols[1]:
        if event.exhibits:
            st.markdown("**Exhibits referenced:** " + ", ".join(
                f"Exhibit {e}" for e in event.exhibits
            ))
        parsed_exhibits = [s for s in exhibits_status if s.parsed]
        if parsed_exhibits:
            st.markdown(
                "**Exhibits parsed:** "
                + ", ".join(f"{s.doc_type} ({s.char_count:,} chars)" for s in parsed_exhibits)
            )

    # ----- C. Event Summary ----------------------------------------------
    st.markdown("### Event Summary")
    st.write(event.event_summary)

    # ----- D. Key Financial Metrics --------------------------------------
    if "2.02" in event.item_numbers or event.key_metrics:
        st.markdown("### Key Financial Metrics")
        if event.key_metrics:
            st.caption(
                "Extracted from press-release language in Exhibit 99.1. "
                "Verify against the GAAP / non-GAAP reconciliation in the source."
            )
            _render_8k_metrics(event.key_metrics)
        elif event.exhibit_99_1_detected and not any(
            s.parsed and s.doc_type.upper().startswith("EX-99.1") for s in exhibits_status
        ):
            st.warning(
                "Exhibit 99.1 was detected but could not be parsed. The press "
                "release content is therefore unavailable to this analysis. "
                "Open the source filing directly for the as-released metrics."
            )
        else:
            st.write(
                "_Press-release-style headline metrics were not detected. "
                "Open the source filing for the as-released financial table._"
            )

    # ----- E. Guidance / Outlook -----------------------------------------
    st.markdown("### Guidance / Outlook")
    if event.guidance_lines:
        for g in event.guidance_lines:
            st.markdown(
                f"""
                <div class="fd-guidance">
                  <div>{g}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write(
            "_No specific forward-looking guidance with a quantitative "
            "anchor detected in the parsed documents._"
        )

    # ----- F. Business Highlights ----------------------------------------
    st.markdown("### Business Highlights")
    _render_8k_highlights(event.highlights)

    # ----- G. Investment Read --------------------------------------------
    st.markdown("### Investment Read")
    st.write(event.investment_read)

    # ----- H. Source Evidence --------------------------------------------
    if event.raw_excerpts:
        with st.expander("Source evidence", expanded=False):
            for ex in event.raw_excerpts:
                st.markdown(f"> {ex}")

    # --- Debug expander -------------------------------------------------
    with st.expander("Parsing diagnostics", expanded=False):
        st.markdown(
            f"- **Primary 8-K document parsed:** "
            f"{'yes' if event.primary_parsed else 'no'}"
            f" ({event.primary_char_count:,} chars)"
        )
        st.markdown(
            f"- **Exhibit 99.1 detected on cover:** "
            f"{'yes' if event.exhibit_99_1_detected else 'no'}"
        )
        if exhibits_status:
            for s in exhibits_status:
                st.markdown(
                    f"- **{s.doc_type}** (`{s.name}`)  \n"
                    f"  URL: {s.url}  \n"
                    f"  Parsed: {'yes' if s.parsed else 'no'} "
                    f"· chars: {s.char_count:,}"
                    + (f"  \n  Note: {s.note}" if s.note else "")
                )
        else:
            st.markdown(
                "- No exhibits in the targeted set (EX-99 / EX-10 / EX-2 / "
                "EX-4 / EX-1) were listed in the filing index."
            )
        # Optional: full document list for inspection.
        if docs:
            other_docs = [d for d in docs if d not in targeted_exhibits]
            if other_docs:
                st.caption("Other files in the filing folder (not parsed):")
                for d in other_docs[:30]:
                    st.markdown(
                        f"  - `{d.name}` — type `{d.doc_type or 'unknown'}`"
                    )

    st.divider()
    st.caption(
        "This output is AI-generated and rule-based. It may contain errors "
        "and is for educational/research workflow purposes only, not "
        "investment advice. Verify all conclusions against the original SEC "
        "filings."
    )


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------


def render_filing_tab(
    company: Company,
    form: str,
    filings: list[Filing],
    facts: Optional[dict],
    note: Optional[str] = None,
) -> None:
    if len(filings) < 2:
        st.warning(
            f"Could not find two consecutive {form} filings for "
            f"{company.name} ({company.ticker}). Found {len(filings)}."
        )
        return

    latest, prev = filings[0], filings[1]

    # ----- 1. Header -------------------------------------------------------
    st.subheader(f"{company.name} ({company.ticker}) — {form}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"**Latest filing**  \n"
            f"Filed: `{latest.filing_date}` · Reports period: `{latest.report_date}`  \n"
            f"[Open on SEC EDGAR]({latest.primary_doc_url})"
        )
    with cols[1]:
        st.markdown(
            f"**Previous filing**  \n"
            f"Filed: `{prev.filing_date}` · Reports period: `{prev.report_date}`  \n"
            f"[Open on SEC EDGAR]({prev.primary_doc_url})"
        )
    st.caption(f"CIK: {company.cik}")
    if note:
        st.info(note)

    # ----- Run the pipeline ------------------------------------------------
    with st.spinner(f"Downloading and analyzing {form} filings…"):
        try:
            latest_paras_raw, latest_html = cached_filing(
                company.cik, latest.accession_no, latest.primary_document,
                latest.form, latest.filing_date, latest.report_date,
            )
            prev_paras_raw, _ = cached_filing(
                company.cik, prev.accession_no, prev.primary_document,
                prev.form, prev.filing_date, prev.report_date,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to fetch or parse filings: {exc}")
            return

    latest_paras = [Paragraph(**p) for p in latest_paras_raw]
    prev_paras = [Paragraph(**p) for p in prev_paras_raw]
    changes = diff_paragraphs(prev_paras, latest_paras)

    # Pull XBRL metrics. For 8-K we expect nothing useful and degrade.
    if facts and form.upper() in ("10-Q", "10-K"):
        income = fetch_metrics(facts, form, INCOME_METRICS, kind="flow")
        cashflow = fetch_metrics(facts, form, CASHFLOW_METRICS, kind="flow")
        balance = fetch_metrics(facts, form, BALANCE_METRICS, kind="stock")
    else:
        income, cashflow, balance = [], [], []

    # Parse disaggregated revenue tables from the latest filing.
    if form.upper() in ("10-Q", "10-K"):
        product_rows = extract_product_breakdown(latest_html)
        geographic_rows = extract_geographic_breakdown(latest_html)
    else:
        product_rows, geographic_rows = [], []

    # Narrative extractions from the latest filing's prose.
    drivers = extract_drivers(latest_paras)
    guidance = extract_guidance(latest_paras)
    segment_quotes = segment_commentary(
        latest_paras,
        [r.label for r in product_rows] + [r.label for r in geographic_rows],
    )
    # If structured tables didn't return rows, fall back to an MD&A scan
    # for common segment names (Specialty / Rubber / Cloud / Services / etc.).
    fallback_segments = (
        extract_fallback_segments(latest_paras)
        if (not product_rows and not geographic_rows)
        else []
    )

    report: ReportBundle = build_report(
        form, changes,
        latest_paragraphs=latest_paras,
        income_metrics=income,
        cashflow_metrics=cashflow,
        balance_metrics=balance,
        product_rows=product_rows,
        geographic_rows=geographic_rows,
        drivers=drivers,
    )

    # Filing timeliness if we can compute it -- compares report date to
    # filing date and notes the lag in days.
    timeliness_note = _filing_timeliness(latest)

    # Header signal row
    head_cols = st.columns([3, 2])
    with head_cols[0]:
        st.markdown(
            f"**Overall Change Signal:** {_signal_pill(report.overall_signal)}",
            unsafe_allow_html=True,
        )
        if report.key_topics:
            chips = " ".join(
                f'<span class="fd-tag">{t}</span>' for t in report.key_topics
            )
            st.markdown(
                f"**Key Disclosure Topics:** {chips}", unsafe_allow_html=True,
            )
    with head_cols[1]:
        if timeliness_note:
            st.caption(timeliness_note)

    st.divider()

    # ----- 2. Executive Summary --------------------------------------------
    st.markdown("### Executive Summary")
    st.write(report.executive_summary)

    # 8-K filings: stop here for the structured sections (they don't apply)
    # but still show the topic cards and red-flag findings below.
    is_8k = form.upper() == "8-K"

    # ----- 3. Key Financials -----------------------------------------------
    if not is_8k:
        st.markdown("### Key Financials")
        st.caption(
            f"Values from SEC XBRL companyfacts. Latest = {form} period; "
            "prior = same fiscal quarter one year earlier when available."
        )
        _render_metrics_grid(income)
        _render_evidence(
            report.ranked_changes,
            {"Key Financials", "Revenue Recognition"},
        )

    # ----- 4. Business Drivers --------------------------------------------
    if not is_8k:
        st.markdown("### Business Drivers")
        st.caption(
            "Why the period moved -- in management's own words, grouped "
            "by category (pricing / volume / mix, cost, FX, etc.)."
        )
        _render_drivers(drivers)

    # ----- 5. Guidance / Outlook ------------------------------------------
    if not is_8k:
        st.markdown("### Guidance / Outlook")
        st.caption(
            "Forward-looking statements detected in the filing, grouped by "
            "metric hint."
        )
        _render_guidance(guidance)

    # ----- 6. Segment / Product / Geographic Performance ------------------
    if not is_8k:
        st.markdown("### Segment / Product / Geographic Performance")
        st.caption(
            "Disaggregated revenue lines parsed from the latest filing's "
            "HTML tables. Per-line color is quoted from the MD&A when "
            "available."
        )
        if product_rows or geographic_rows:
            bcols = st.columns(2)
            with bcols[0]:
                _render_breakdown_with_commentary(
                    "Product / Service", product_rows, segment_quotes
                )
            with bcols[1]:
                _render_breakdown_with_commentary(
                    "Geographic / Region", geographic_rows, segment_quotes
                )
        else:
            _render_fallback_segments(fallback_segments)
            if not fallback_segments:
                st.write(
                    "_Not detected in this basic demo: no structured "
                    "disaggregated-revenue table matched, and no familiar "
                    "segment names surfaced in MD&A prose._"
                )
        _render_evidence(
            report.ranked_changes,
            {"Segment Performance", "Revenue Recognition"},
        )

    # ----- 7. Liquidity & Capital Allocation ------------------------------
    if not is_8k:
        st.markdown("### Liquidity & Capital Allocation")
        st.caption(
            "Balance-sheet positions (period-end) and capital-return cash "
            "flows from SEC XBRL companyfacts."
        )
        st.markdown("**Balance Sheet (Period End)**")
        _render_metrics_grid(balance)
        st.markdown("**Capital Allocation Flows**")
        _render_metrics_grid(cashflow)
        _render_evidence(
            report.ranked_changes,
            {
                "Liquidity and Capital Resources",
                "Debt and Covenants",
                "Capital Allocation",
            },
        )

    # ----- 8. Risk / Legal / Controls -------------------------------------
    st.markdown("### Risk / Legal / Controls")
    st.caption(
        "Severe red flags mean actual events (going concern, covenant "
        "breach, default, material weakness, investigation, impairment "
        "charge, litigation update). Topic sentiment is the tone of "
        "disclosure language and is shown separately below."
    )

    # --- Severe Red Flags ---------------------------------------------------
    st.markdown("**Severe Red Flags**")
    _render_severe_red_flags(report.watch_items)

    # --- Resolved / Removed Prior Risks ------------------------------------
    resolved_items = [w for w in report.watch_items if w.classification == "resolved"]
    if resolved_items:
        st.markdown("**Resolved / Removed Prior Risks**")
        _render_resolved_risks(report.watch_items)

    # --- Cautionary Disclosure Topics --------------------------------------
    risk_topics = {"Risk Factors", "Legal Proceedings", "Controls and Procedures"}
    risk_cards = [c for c in report.topic_cards if c.topic in risk_topics]
    cautionary_cards = [
        c for c in risk_cards
        if c.sentiment in ("Negative", "Slightly Negative", "Mixed")
        and not c.downgraded_from
    ]
    neutralized_cards = [
        c for c in risk_cards
        if c.sentiment == "Neutral" or c.downgraded_from
    ]

    if cautionary_cards or neutralized_cards:
        st.markdown("**Cautionary Disclosure Topics**")
        if cautionary_cards:
            st.caption(
                "However, some disclosure categories still screen as "
                "cautionary based on language changes or filing context."
            )
            for c in cautionary_cards:
                _render_cautionary_topic_card(c)
        if neutralized_cards:
            for c in neutralized_cards:
                _render_cautionary_topic_card(c)
    else:
        st.write(
            "_Risk, legal, and internal-controls disclosures appear "
            "broadly unchanged versus the prior comparable filing._"
        )

    # --- Non-findings (collapsed) ------------------------------------------
    _render_non_findings(report.watch_items)
    _render_evidence(report.ranked_changes, risk_topics)

    # ----- 9. Topic Sentiment Cards ---------------------------------------
    st.markdown("### Topic Sentiment Cards")
    remaining = [c for c in report.topic_cards if c.topic not in risk_topics]
    if not remaining:
        st.write(
            "_No additional topic-level activity to report versus the prior "
            "comparable filing._"
        )
    else:
        for c in remaining:
            _render_topic_card(c)

    # ----- 8. Disclaimer ---------------------------------------------------
    st.divider()
    st.caption(
        "This output is AI-generated and rule-based. It may contain errors "
        "and is for educational/research workflow purposes only, not "
        "investment advice. Verify all conclusions against the original SEC "
        "filings."
    )


def _filing_timeliness(latest: Filing) -> Optional[str]:
    """Return a short timeliness note ('Filed N days after period end')."""
    try:
        from datetime import date
        f = date.fromisoformat(latest.filing_date)
        r = date.fromisoformat(latest.report_date)
        days = (f - r).days
        return f"Filing lag: {days} days after period end."
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("FilingDiff")
st.caption(
    "SEC filing insight reports — XBRL-driven financials, parsed revenue "
    "tables, and topic-level sentiment, with paragraph diffs kept as "
    "supporting evidence."
)

if not run_btn and "fd_last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Run analysis** to begin.")
    st.stop()

if run_btn:
    st.session_state["fd_last_ticker"] = ticker_input
active_ticker = st.session_state.get("fd_last_ticker", ticker_input)

with st.spinner(f"Looking up {active_ticker} on EDGAR…"):
    info = cached_lookup_company(active_ticker)

if info is None:
    st.error(
        f"Could not find ticker `{active_ticker}` in the SEC ticker file. "
        "Check the symbol and try again."
    )
    st.stop()

company = Company(cik=info["cik"], ticker=info["ticker"], name=info["name"])

with st.spinner(f"Loading filing history for {company.name}…"):
    submissions = cached_submissions(company.cik)

with st.spinner(f"Loading XBRL companyfacts for {company.name}…"):
    facts = cached_company_facts(company.cik)

if facts is None:
    st.warning(
        "SEC XBRL companyfacts API returned no data for this company. "
        "Key Financials and Liquidity sections will be marked 'Not detected'."
    )

forms_to_show = [
    ("10-Q", None),
    ("10-K", None),
    (
        "8-K",
        "8-Ks are event-based filings. This tab is rendered as an "
        "event-intelligence report: what event was disclosed, the item "
        "numbers, exhibits, key financial metrics (for Item 2.02 earnings "
        "releases), guidance, and customer / contract highlights.",
    ),
]

tabs = st.tabs([f"📊 {f}" for f, _ in forms_to_show])
for (form, note), tab in zip(forms_to_show, tabs):
    with tab:
        filings = latest_two_filings(submissions, form)
        if form == "8-K":
            render_8k_tab(company, form, filings, note=note)
        else:
            render_filing_tab(company, form, filings, facts, note=note)
