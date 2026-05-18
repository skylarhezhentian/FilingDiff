"""
analysis.py
-----------
Lightweight analytics layer for the polished filing report.

This module no longer tries to produce a "raw paragraph diff" view --
that is now handled inside optional evidence expanders. What remains:

  - Topic detection over filing text and paragraph changes.
  - Sentiment scoring per topic (Positive / Negative / Mixed / etc.).
  - A small ranking pass that lets us surface the highest-relevance
    changes inside per-section evidence expanders.
  - A red-flag scan, kept compact: only phrases that *actually appear*
    in changed paragraphs are reported (the full Found/Not-Found
    watchlist is gone, per the latest design).
  - A rule-based Executive Summary that combines XBRL deltas, table
    rows, and topic activity into one specific narrative paragraph.

What is intentionally *not* here anymore:
  - Analyst follow-up questions
  - Topic-grouped redline blocks
  - Not-Found watchlist checks
  - Filing-history estimate grid
  - MD&A bullet subsections (the new layout sources those from XBRL +
    parsed tables instead of from prose excerpts).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from compare import ParagraphChange


# ---------------------------------------------------------------------------
# Topic dictionary
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Key Financials": [
        "net sales", "net revenue", "total revenue", "operating income",
        "operating margin", "gross margin", "gross profit", "earnings per share",
        "diluted earnings", "net income", "operating loss", "net loss",
    ],
    "Guidance / Outlook": [
        "guidance", "outlook", "we expect", "we anticipate", "fiscal 20",
        "full year", "for the remainder of", "forward-looking",
    ],
    "Liquidity and Capital Resources": [
        "liquidity", "cash and cash equivalents", "working capital",
        "credit facility", "revolving credit", "line of credit",
        "capital resources", "sources of liquidity",
        "marketable securities",
    ],
    "Capital Allocation": [
        "share repurchase", "stock repurchase", "buyback",
        "repurchase program", "dividend", "return capital",
        "10b5-1",
    ],
    "Revenue Recognition": [
        "revenue recognition", "performance obligation", "transaction price",
        "deferred revenue", "contract liability", "asc 606",
    ],
    "Segment Performance": [
        "segment", "reportable segment", "geographic", "product line",
        "iphone", "mac", "ipad", "wearables", "services",
        "americas", "greater china", "rest of asia pacific",
    ],
    "Debt and Covenants": [
        "covenant", "indenture", "senior notes", "term loan", "notes due",
        "interest rate swap", "long-term debt", "credit agreement",
        "in compliance with", "default",
    ],
    "Customer Concentration": [
        "customer concentration", "one customer", "two customers",
        "major customer", "10% of", "ten percent of", "no single customer",
    ],
    "Risk Factors": [
        "risk factor", "we may be unable", "could materially", "adverse effect",
        "uncertain", "no assurance",
    ],
    "Legal Proceedings": [
        "litigation", "lawsuit", "complaint was filed", "settlement",
        "legal proceedings", "subpoena", "investigation", "class action",
        "antitrust",
    ],
    "Controls and Procedures": [
        "internal control", "disclosure controls", "material weakness",
        "remediation", "icfr",
    ],
    "Related Party Transactions": [
        "related party", "related-party",
    ],
    "Tariffs / Geopolitical Risk": [
        "tariff", "sanction", "export control", "geopolitical",
        "russia", "ukraine", "israel", "china trade", "section 301",
    ],
    "Management Changes": [
        "appointed", "resigned", "stepped down", "ceo transition",
        "chief executive officer", "chief financial officer", "departure of",
        "named as", "principal officer",
    ],
}


# ---------------------------------------------------------------------------
# Sentiment vocab
# ---------------------------------------------------------------------------

POS_WORDS = {
    "increase", "increased", "growth", "grew", "improved", "improvement",
    "higher", "expanded", "strong", "strength", "record", "favorable",
    "compliance with", "remediated", "in compliance", "successful",
    "raised", "outperform",
}

NEG_WORDS = {
    "decrease", "decreased", "decline", "declined", "weak", "weaker",
    "lower", "loss", "losses", "shortfall", "impairment", "restructuring",
    "termination", "going concern", "material weakness", "default",
    "covenant breach", "investigation", "subpoena", "lawsuit",
    "liquidity concern", "downturn", "unfavorable", "headwind",
    "deteriorat", "reduce", "reduced", "delay", "delays",
}


RED_FLAG_PHRASES = [
    "material weakness", "going concern", "substantial doubt",
    "subpoena", "investigation", "default", "covenant breach",
    "impairment", "restructuring", "termination",
    "customer concentration", "liquidity concerns",
    "decline in backlog", "lower guidance", "off-balance sheet",
    "off balance sheet",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TopicCard:
    """One topic-level sentiment card.

    The card is composed by the UI / analysis pipeline from external
    inputs (XBRL numbers + driver quotes + sentiment). The
    ``narrative`` field carries the rendered 3-5 sentence interpretation
    and ``numbers`` carries the headline numeric lines for the card.
    No mechanical "N supporting changes" wording is exposed here.
    """

    topic: str
    sentiment: str  # Positive | Slightly Positive | Neutral | Slightly Negative | Negative | Mixed
    narrative: str
    numbers: list  # list[NumberLine] from narrative.py
    why_it_matters: str
    # Kept internally for *sorting* only; never rendered.
    evidence_count: int = 0


@dataclass
class RankedChange:
    change: ParagraphChange
    topics: list[str]
    score: float
    why_it_matters: str


@dataclass
class RedFlagHit:
    """A red-flag phrase that *appeared in a changed paragraph*."""

    phrase: str
    section: str
    excerpt: str
    change_type: str


@dataclass
class WatchItem:
    """A watchlist phrase + its classification.

    classification is one of:
      - "true"        : phrase is asserted as present (true red flag)
      - "negated"     : phrase is mentioned but explicitly negated (e.g.
                        "Defaults Upon Senior Securities: None")
      - "boilerplate" : recurring SEC disclosure that always reads as
                        a non-finding (typically a header followed by
                        "None" / "Not applicable")
      - "resolved"    : phrase was present in the prior filing and is
                        now removed from changed text (suggests the
                        prior-period risk has dropped out of disclosure)
    Only "true" classifications are shown prominently by the UI.
    """

    phrase: str
    classification: str  # "true" | "negated" | "boilerplate" | "resolved"
    section: str
    excerpt: str
    why_it_matters: str
    change_type: Optional[str] = None  # if surfaced via a paragraph change


@dataclass
class ReportBundle:
    overall_signal: str
    key_topics: list[str]
    executive_summary: str
    topic_cards: list[TopicCard]
    ranked_changes: list[RankedChange]
    red_flag_hits: list[RedFlagHit]
    watch_items: list[WatchItem]


# ---------------------------------------------------------------------------
# Topic detection / sentiment
# ---------------------------------------------------------------------------


def detect_topics(text: str) -> list[str]:
    t = text.lower()
    hits = []
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits.append(topic)
                break
    return hits


def _polarity(text: str) -> tuple[int, int]:
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    return pos, neg


def _label_polarity(pos: int, neg: int) -> str:
    """Conservative polarity label.

    The previous version flipped to Slightly Positive on a single
    polarity hit, which produced overconfident sentiment on weak
    evidence. We now require:
      - At least 3 total hits before crossing into any directional
        Slightly* label.
      - At least a 3-hit gap before calling Positive / Negative outright.
    Sparse evidence stays Neutral, which the UI surfaces honestly.
    """
    total = pos + neg
    if total < 3:
        return "Neutral"
    diff = pos - neg
    if abs(diff) <= 1:
        return "Mixed"
    if diff >= 4:
        return "Positive"
    if diff >= 2:
        return "Slightly Positive"
    if -diff >= 4:
        return "Negative"
    if -diff >= 2:
        return "Slightly Negative"
    return "Mixed"


def detect_red_flag_hits(changes: Iterable[ParagraphChange]) -> list[RedFlagHit]:
    """Only return red-flag phrases that surfaced in *changed* paragraphs."""
    hits: list[RedFlagHit] = []
    seen: set[tuple[str, str]] = set()
    for ch in changes:
        text_low = ch.display_text.lower()
        for phrase in RED_FLAG_PHRASES:
            if phrase in text_low:
                key = (phrase, ch.display_text[:80])
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    RedFlagHit(
                        phrase=phrase,
                        section=ch.section,
                        excerpt=_clip(ch.display_text, 360),
                        change_type=ch.change_type,
                    )
                )
    return hits


# ---------------------------------------------------------------------------
# Watch-item classification (with negation handling)
# ---------------------------------------------------------------------------

# Phrases that, when found within a small window of the watchlist
# phrase, mean it is being *negated* rather than asserted.
_NEGATION_PATTERNS = [
    r":\s*None\b",
    r":\s*Not\s+Applicable\b",
    r":\s*N/?A\b",
    r"\bNone\b",
    r"\bnot\s+applicable\b",
    r"\bno\s+(?:material|known|pending|reportable|current|outstanding|director|officer|off-?balance)\b",
    r"\bdid\s+not\b",
    r"\bhave\s+not\b",
    r"\bwere\s+no\b",
    r"\bthere\s+are\s+no\b",
    r"\bthere\s+were\s+no\b",
    r"\bnone\s+of\s+the\b",
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bare\s+not\b",
    r"\bis\s+not\b",
    r"\bwere\s+not\b",
]

# Per-phrase context windows: how far before / after the phrase to
# check for negation language.
_NEG_WINDOW_BEFORE = 80
_NEG_WINDOW_AFTER = 60

# Watchlist phrases that, in SEC filings, almost always show up as the
# *header* of a "None" / "Not applicable" line item. These are flagged
# as boilerplate when found in a negated context, so they don't pollute
# the primary watch list.
_BOILERPLATE_HEADERS = {
    "off-balance sheet",
    "off balance sheet",
}


def _is_negated(text: str, phrase: str) -> tuple[bool, bool]:
    """Return (is_negated, looks_like_boilerplate_header)."""
    low = text.lower()
    idx = low.find(phrase)
    if idx < 0:
        return False, False
    window = text[max(0, idx - _NEG_WINDOW_BEFORE) : idx + len(phrase) + _NEG_WINDOW_AFTER]
    win_low = window.lower()
    for pat in _NEGATION_PATTERNS:
        if re.search(pat, window, flags=re.IGNORECASE):
            # Boilerplate is a negated phrase that is also one of the
            # known recurring headers like "Off-balance Sheet" / "None".
            is_boilerplate = phrase in _BOILERPLATE_HEADERS or (
                # "Defaults Upon Senior Securities: None" pattern.
                re.search(r"\b(?:defaults?|controls?|legal\s+proceedings?|securities)\b", win_low)
                and "none" in win_low
            )
            return True, bool(is_boilerplate)
    return False, False


_RED_FLAG_WHY: dict[str, str] = {
    "material weakness": "Deficiency in internal control over financial reporting; can precede restatements.",
    "going concern": "Doubt about ability to continue operations for the next 12 months.",
    "substantial doubt": "Formal going-concern accounting language.",
    "subpoena": "Compelled production of documents; signals an active investigation.",
    "investigation": "Active regulator / government inquiry. Outcomes can include fines or restatements.",
    "default": "Contractual failure -- typically on debt -- can accelerate maturities.",
    "covenant breach": "Lender may demand repayment or restrict capital deployment.",
    "impairment": "Asset write-down; signals deteriorated economics in a unit or asset.",
    "restructuring": "Reorganization affecting severance / facilities; value-positive or destructive case-by-case.",
    "termination": "Material contract / customer / executive separation.",
    "customer concentration": "Revenue dependent on one or a few customers; step-function risk.",
    "liquidity concerns": "Explicit cautionary language about funding ability.",
    "decline in backlog": "Reduction in committed future revenue; leading indicator.",
    "lower guidance": "Management has reduced expectations vs. prior communication.",
    "off-balance sheet": "Arrangements not reflected on the balance sheet; size the exposure.",
    "off balance sheet": "Arrangements not reflected on the balance sheet; size the exposure.",
}


def classify_watch_items(
    changes: Iterable[ParagraphChange],
    latest_paragraphs: list,
) -> list[WatchItem]:
    """Classify each watchlist phrase as True / Negated / Boilerplate / Resolved.

    Two-stage scan:
      - Stage 1: walk changed paragraphs. If a phrase appears in a New /
        Big-Change paragraph and is *not* negated, classify as True.
        If it appears in a Deleted paragraph, classify as Resolved.
        Negated => Negated / Boilerplate.
      - Stage 2: scan latest filing text for any phrases still
        unaccounted for. Classify as True / Negated / Boilerplate.

    Phrases not found anywhere are omitted (no "Not Found" noise).
    """
    items: list[WatchItem] = []
    classified: set[str] = set()
    latest_text_blob = "\n".join(
        (p.text if hasattr(p, "text") else str(p)) for p in latest_paragraphs
    )

    for ch in changes:
        text = ch.display_text
        low = text.lower()
        for phrase in RED_FLAG_PHRASES:
            if phrase in classified:
                continue
            if phrase not in low:
                continue
            negated, boilerplate = _is_negated(text, phrase)
            if ch.change_type == "Deleted" and not negated:
                cls = "resolved"
            elif boilerplate:
                cls = "boilerplate"
            elif negated:
                cls = "negated"
            else:
                cls = "true"
            items.append(
                WatchItem(
                    phrase=phrase,
                    classification=cls,
                    section=ch.section,
                    excerpt=_clip(text, 360),
                    why_it_matters=_RED_FLAG_WHY.get(phrase, "Watchlist item."),
                    change_type=ch.change_type,
                )
            )
            classified.add(phrase)

    # Stage 2: anything not yet classified, scan the full latest filing.
    for phrase in RED_FLAG_PHRASES:
        if phrase in classified:
            continue
        if phrase not in latest_text_blob.lower():
            continue
        negated, boilerplate = _is_negated(latest_text_blob, phrase)
        if boilerplate:
            cls = "boilerplate"
        elif negated:
            cls = "negated"
        else:
            cls = "true"
        # Extract a small excerpt around the first hit.
        idx = latest_text_blob.lower().find(phrase)
        ex = latest_text_blob[max(0, idx - 80) : idx + 200]
        items.append(
            WatchItem(
                phrase=phrase,
                classification=cls,
                section="—",
                excerpt=_clip(ex, 360),
                why_it_matters=_RED_FLAG_WHY.get(phrase, "Watchlist item."),
                change_type=None,
            )
        )
        classified.add(phrase)

    # Order: true > resolved > negated > boilerplate.
    order = {"true": 0, "resolved": 1, "negated": 2, "boilerplate": 3}
    items.sort(key=lambda w: order.get(w.classification, 9))
    return items


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


_CHANGE_TYPE_WEIGHT = {
    "New": 1.4,
    "Deleted": 1.2,
    "Big Change": 1.3,
    "Medium Change": 1.0,
    "Small Change": 0.6,
}


def rank_changes(changes: list[ParagraphChange]) -> list[RankedChange]:
    ranked: list[RankedChange] = []
    for ch in changes:
        topics = detect_topics(ch.display_text)
        score = _CHANGE_TYPE_WEIGHT.get(ch.change_type, 0.5)
        score += 0.4 * len(topics)
        low = ch.display_text.lower()
        for phrase in RED_FLAG_PHRASES:
            if phrase in low:
                score += 1.5
                break
        n = len(ch.display_text)
        if 200 <= n <= 1500:
            score += 0.3
        ranked.append(
            RankedChange(
                change=ch,
                topics=topics,
                score=score,
                why_it_matters=_why_it_matters(ch, topics),
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def _why_it_matters(ch: ParagraphChange, topics: list[str]) -> str:
    bits = []
    if ch.change_type == "New":
        bits.append("Newly disclosed language not present in the prior filing")
    elif ch.change_type == "Deleted":
        bits.append("Language that appeared in the prior filing was removed")
    elif ch.change_type == "Big Change":
        bits.append("Substantively rewritten disclosure")
    else:
        bits.append(f"{ch.change_type} to existing disclosure")
    if topics:
        bits.append("touches " + ", ".join(topics[:3]))
    return "; ".join(bits) + "."


def filter_changes_by_topics(
    ranked: list[RankedChange], wanted: set[str], limit: int = 5
) -> list[RankedChange]:
    """Return up to `limit` ranked changes whose topics intersect `wanted`."""
    out = [r for r in ranked if any(t in wanted for t in r.topics)]
    return out[:limit]


# ---------------------------------------------------------------------------
# Topic sentiment cards
# ---------------------------------------------------------------------------


def build_topic_cards(
    ranked: list[RankedChange],
    latest_paragraphs: list,
    *,
    income_metrics: list,
    cashflow_metrics: list,
    balance_metrics: list,
) -> list[TopicCard]:
    """Build topic sentiment cards combining sentiment + XBRL numbers + driver quote.

    No mechanical "N supporting changes" wording leaks into the rendered
    card. The polarity signal is derived from the changed paragraphs as
    before; the rest of the card pulls in real numbers and a quoted
    driver sentence so the narrative actually carries content.
    """
    # Locally import to keep the module decoupled at top-level.
    from narrative import (
        TOPIC_WHY, topic_narrative, topic_numbers,
        CAUSAL_PHRASES, _find_sentence_with,
    )

    by_topic: dict[str, list[RankedChange]] = defaultdict(list)
    for r in ranked:
        for t in r.topics:
            by_topic[t].append(r)

    cards: list[TopicCard] = []
    for topic, items in by_topic.items():
        pos_total = neg_total = 0
        for r in items[:8]:
            p, n = _polarity(r.change.display_text)
            pos_total += p
            neg_total += n
        sentiment = _label_polarity(pos_total, neg_total)

        numbers = topic_numbers(topic, income_metrics, cashflow_metrics, balance_metrics)
        driver_quote = _topic_driver_quote(topic, latest_paragraphs, CAUSAL_PHRASES, _find_sentence_with)
        narrative = topic_narrative(topic, sentiment, numbers, driver_quote)
        why = TOPIC_WHY.get(topic, "")

        cards.append(
            TopicCard(
                topic=topic,
                sentiment=sentiment,
                narrative=narrative,
                numbers=numbers,
                why_it_matters=why,
                evidence_count=len(items),
            )
        )

    priority = {
        "Controls and Procedures": 0,
        "Legal Proceedings": 1,
        "Risk Factors": 2,
        "Debt and Covenants": 3,
        "Liquidity and Capital Resources": 4,
        "Capital Allocation": 5,
        "Customer Concentration": 6,
        "Revenue Recognition": 7,
        "Segment Performance": 8,
        "Guidance / Outlook": 9,
        "Tariffs / Geopolitical Risk": 10,
        "Management Changes": 11,
        "Related Party Transactions": 12,
        "Key Financials": 13,
    }
    cards.sort(key=lambda c: (priority.get(c.topic, 50), -c.evidence_count))
    return cards


def _topic_driver_quote(
    topic: str,
    latest_paragraphs: list,
    causal_phrases: list,
    finder,
) -> Optional[str]:
    """Pull a single driver sentence relevant to a topic, when available.

    The keyword set is built from TOPIC_KEYWORDS so we don't repeat
    ourselves. We accept the first sentence that mentions any topic
    keyword and includes a causal phrase.
    """
    kws = [k.lower() for k in TOPIC_KEYWORDS.get(topic, [])]
    if not kws:
        return None
    for p in latest_paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        low = text.lower()
        if not any(k in low for k in kws):
            continue
        for phrase in causal_phrases:
            if phrase in low:
                sent = finder(text, phrase)
                if sent and any(k in sent.lower() for k in kws):
                    if 40 <= len(sent) <= 400:
                        return sent
    return None


# ---------------------------------------------------------------------------
# Executive summary
#
# Combines XBRL deltas + parsed table deltas + topic activity into a
# single specific paragraph. The XBRL / table arguments are typed as
# `list` (rather than `list[XBRLMetric]` etc.) to keep this module
# decoupled from the xbrl/tables modules at import time.
# ---------------------------------------------------------------------------


def build_executive_summary(
    form: str,
    income_metrics: list,
    cashflow_metrics: list,
    balance_metrics: list,
    product_rows: list,
    geographic_rows: list,
    ranked: list[RankedChange],
    red_flag_hits: list[RedFlagHit],
    drivers: list,
) -> tuple[str, str]:
    """Return (overall_signal, executive_summary_paragraph).

    Structured as a short research-note paragraph covering, in order:
      1. Lead sentence: headline revenue + EPS (or operating income).
      2. What improved: the up movers (P&L deltas, product/geo bests).
      3. What worsened: the down movers (P&L deltas, product/geo worsts).
      4. What drove the change: the highest-priority driver sentence.
      5. Main risk / watch item: red-flag callout or "no red-flag
         language surfaced".
      6. Overall signal: one closing sentence reiterating the call.

    No "N disclosure changes" / "tagged to topic" wording is emitted.
    """
    rev = _find_by_label(income_metrics, "Net Revenue")
    op = _find_by_label(income_metrics, "Operating Income")
    ni = _find_by_label(income_metrics, "Net Income")
    eps = _find_by_label(income_metrics, "Diluted EPS")
    gm = _find_by_label(income_metrics, "Gross Profit")
    cash = _find_by_label(balance_metrics, "Cash & Equivalents")
    debt_lt = _find_by_label(balance_metrics, "Long-Term Debt")
    buybacks = _find_by_label(cashflow_metrics, "Stock Repurchases")
    ocf = _find_by_label(cashflow_metrics, "Operating Cash Flow")

    overall = _overall_signal(rev, op, ni, ranked, red_flag_hits)

    parts: list[str] = []

    # ---- 1. Lead sentence ------------------------------------------------
    lead = _lead_sentence(form, rev, eps, op)
    if lead:
        parts.append(lead)

    # ---- 2. What improved ------------------------------------------------
    improved = _improved_clause(rev, op, ni, gm, cash, ocf, product_rows, geographic_rows)
    if improved:
        parts.append(improved)

    # ---- 3. What worsened ------------------------------------------------
    worsened = _worsened_clause(rev, op, ni, debt_lt, ocf, product_rows, geographic_rows)
    if worsened:
        parts.append(worsened)

    # ---- 4. What drove it -----------------------------------------------
    driver_line = _driver_clause(drivers)
    if driver_line:
        parts.append(driver_line)

    # ---- 5. Risk / watch item -------------------------------------------
    parts.append(_risk_clause(red_flag_hits))

    # ---- 6. Overall signal closer ---------------------------------------
    parts.append(f"On balance the period reads as {overall.lower()} versus the prior comparable filing.")

    if not parts:
        parts.append(
            f"Insufficient XBRL or disclosure data to summarize this {form}. "
            "Verify against the source filing."
        )
    return overall, " ".join(parts)


def _lead_sentence(form: str, rev, eps, op) -> Optional[str]:
    """Compose the headline opening of the executive summary."""
    if rev is None or rev.latest_val is None:
        return None
    head = f"For the period ending {rev.latest_end}, the {form} reported revenue of {_money(rev.latest_val)}"
    if rev.delta_pct is not None:
        verb = "up" if rev.delta_pct >= 0 else "down"
        head += f", {verb} {abs(rev.delta_pct):.1f}% versus the prior comparable period"
    if eps and eps.latest_val is not None:
        head += f", with diluted EPS of {_eps(eps.latest_val)}"
        if eps.delta_pct is not None:
            verb = "up" if eps.delta_pct >= 0 else "down"
            head += f" ({verb} {abs(eps.delta_pct):.1f}%)"
    elif op and op.latest_val is not None and op.delta_pct is not None:
        verb = "expanding" if op.delta_pct >= 0 else "compressing"
        head += f", with operating income {verb} {abs(op.delta_pct):.1f}% to {_money(op.latest_val)}"
    return head + "."


def _improved_clause(rev, op, ni, gm, cash, ocf, product_rows, geographic_rows) -> Optional[str]:
    bullets: list[str] = []
    for m, label in (
        (op, "operating income"),
        (ni, "net income"),
        (gm, "gross profit"),
        (ocf, "operating cash flow"),
        (cash, "cash and equivalents"),
    ):
        if m and m.delta_pct is not None and m.delta_pct > 1.0 and m.latest_val is not None:
            bullets.append(f"{label} +{m.delta_pct:.1f}% to {_money(m.latest_val)}")

    for rows, label_prefix in ((product_rows, "Product"), (geographic_rows, "Region")):
        movers = [r for r in rows if r.delta_pct is not None and r.delta_pct > 1.0]
        movers.sort(key=lambda r: r.delta_pct, reverse=True)
        if movers:
            best = movers[0]
            bullets.append(f"{label_prefix.lower()} strength in {best.label} (+{best.delta_pct:.1f}%)")

    if not bullets:
        return None
    return "On the positive side, " + _humanize_list(bullets[:4]) + "."


def _worsened_clause(rev, op, ni, debt_lt, ocf, product_rows, geographic_rows) -> Optional[str]:
    bullets: list[str] = []
    for m, label in (
        (rev, "revenue"),
        (op, "operating income"),
        (ni, "net income"),
        (ocf, "operating cash flow"),
    ):
        if m and m.delta_pct is not None and m.delta_pct < -1.0 and m.latest_val is not None:
            bullets.append(f"{label} {m.delta_pct:.1f}% to {_money(m.latest_val)}")

    for rows, label_prefix in ((product_rows, "Product"), (geographic_rows, "Region")):
        movers = [r for r in rows if r.delta_pct is not None and r.delta_pct < -1.0]
        movers.sort(key=lambda r: r.delta_pct)
        if movers:
            worst = movers[0]
            bullets.append(f"{label_prefix.lower()} softness in {worst.label} ({worst.delta_pct:+.1f}%)")

    if not bullets:
        return None
    return "On the negative side, " + _humanize_list(bullets[:4]) + "."


def _driver_clause(drivers: list) -> Optional[str]:
    """Pick the highest-priority driver sentence to anchor the 'what drove it' clause."""
    if not drivers:
        return None
    top = drivers[0]
    cat = getattr(top, "category", "")
    excerpt = getattr(top, "excerpt", "")
    if not excerpt:
        return None
    return f"Management attributes the period's moves to {cat.lower()}: \"{excerpt}\""


def _risk_clause(red_flag_hits: list[RedFlagHit]) -> str:
    if not red_flag_hits:
        return "No red-flag language surfaced in the changed disclosures this period."
    unique = sorted({h.phrase for h in red_flag_hits})
    return (
        "Watch items include "
        + _humanize_list([p.title() for p in unique[:4]])
        + " language flagged in changed disclosures."
    )


def _humanize_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _overall_signal(
    rev, op, ni, ranked: list[RankedChange], red_flag_hits: list[RedFlagHit]
) -> str:
    """Coarse overall signal label based on revenue / operating / net deltas."""
    score = 0
    samples = 0
    for m in (rev, op, ni):
        if m and m.delta_pct is not None:
            samples += 1
            if m.delta_pct > 1.0:
                score += 1
            elif m.delta_pct < -1.0:
                score -= 1

    # Topic polarity overlay from changed paragraphs.
    pos = neg = 0
    for r in ranked[:30]:
        p, n = _polarity(r.change.display_text)
        pos += p
        neg += n
    if pos - neg >= 5:
        score += 1
    elif neg - pos >= 5:
        score -= 1

    if samples == 0 and pos == 0 and neg == 0:
        return "Neutral"

    if red_flag_hits and score >= 0:
        return "Mixed"
    if score >= 2:
        return "Positive"
    if score == 1:
        return "Slightly Positive"
    if score == 0:
        return "Mixed" if red_flag_hits else "Neutral"
    if score == -1:
        return "Slightly Negative"
    return "Negative"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_by_label(metrics: list, label: str):
    for m in metrics:
        if getattr(m, "label", None) == label:
            return m
    return None


def _money(val: float) -> str:
    absv = abs(val)
    if absv >= 1e9:
        return f"${val/1e9:.2f}B"
    if absv >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"


def _eps(val: float) -> str:
    return f"${val:.2f}"


def _pct(p: float) -> str:
    return f"{p:+.1f}%"


def _top_topics(ranked: list[RankedChange], k: int = 4) -> list[str]:
    counts: Counter[str] = Counter()
    for r in ranked:
        for t in r.topics:
            counts[t] += 1
    return [t for t, _ in counts.most_common(k)]


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def build_report(
    form: str,
    changes: list[ParagraphChange],
    *,
    latest_paragraphs: list,
    income_metrics: list,
    cashflow_metrics: list,
    balance_metrics: list,
    product_rows: list,
    geographic_rows: list,
    drivers: list,
) -> ReportBundle:
    ranked = rank_changes(changes)
    red_flag_hits = detect_red_flag_hits(changes)
    watch_items = classify_watch_items(changes, latest_paragraphs)
    overall, exec_summary = build_executive_summary(
        form, income_metrics, cashflow_metrics, balance_metrics,
        product_rows, geographic_rows, ranked,
        [w for w in watch_items if w.classification == "true"], drivers,
    )
    cards = build_topic_cards(
        ranked, latest_paragraphs,
        income_metrics=income_metrics,
        cashflow_metrics=cashflow_metrics,
        balance_metrics=balance_metrics,
    )
    key_topics = _top_topics(ranked, k=6) or [c.topic for c in cards[:6]]
    return ReportBundle(
        overall_signal=overall,
        key_topics=key_topics,
        executive_summary=exec_summary,
        topic_cards=cards,
        ranked_changes=ranked,
        red_flag_hits=red_flag_hits,
        watch_items=watch_items,
    )
