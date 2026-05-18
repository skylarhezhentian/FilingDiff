"""
narrative.py
------------
Higher-quality interpretation helpers that turn raw filing text into the
plain-English bullets a research note would carry.

What lives here:

  - extract_drivers()   : find sentences that explain WHY something
                          changed ("driven by", "primarily due to",
                          "offset by", ...), grouped into business
                          categories (Pricing/Mix, Cost, FX, etc.).
  - extract_guidance()  : find forward-looking statements ("we expect",
                          "for the full year", "guidance", ...).
  - segment_commentary(): find quoted driver sentences keyed by a
                          segment / product / region name so the
                          performance section can carry per-line color.
  - topic_narrative()   : build a 3-5 sentence narrative for a topic
                          sentiment card, combining XBRL numbers,
                          quoted drivers, and a polarity-aware closer.
  - topic_numbers()     : pick the XBRL / table values that belong on
                          each topic card so the card carries real
                          numbers rather than just adjectives.
  - TOPIC_WHY           : a curated one-line "why it matters" per topic.

Style rules enforced here:

  - Never emit phrases like "N disclosure changes", "N supporting
    changes", "tagged to topic X". Those are mechanical and belong
    inside collapsed evidence expanders, not the headline narrative.
  - When a number is not available, omit the clause rather than
    invent one.
  - Quoted sentences are always wrapped in '"..."' so the reader
    knows they are management's words, not ours.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

CAUSAL_PHRASES = [
    # Primary-cause phrases first. SEC writers usually put the headline
    # driver before any "partially offset by" offset clause; matching in
    # this order means we attribute the period to the primary driver and
    # treat offsets as supporting context, not the lead.
    "driven by", "primarily due to", "primarily attributable to",
    "attributable to", "as a result of", "due to", "because of",
    "impacted by", "led by", "supported by", "benefited from",
    "weighed on", "reflecting", "resulting from",
    # Offset phrases last -- only used when nothing else matches.
    "partially offset by", "offset by",
]

GUIDANCE_PHRASES = [
    "we expect", "we anticipate", "we project", "we forecast",
    "for the remainder of", "for the full year", "full-year",
    "fiscal year 20", "guidance", "outlook", "we estimate",
    "for fiscal 20", "expects to", "anticipates", "we plan to",
    "is expected to", "are expected to", "we target", "targets",
    "we forecast", "reaffirm",
]

# Sentences that contain these phrases are *boilerplate* disclaimers, not
# actual guidance. We filter them out before scoring.
GUIDANCE_BOILERPLATE_PHRASES = [
    "forward-looking statements",
    "safe harbor",
    "private securities litigation reform act",
    "actual results may differ",
    "subject to risks and uncertainties",
    "no obligation to update",
    "undertake no obligation",
    "include but are not limited to",
    "are based on management's current",
    "involve risks and uncertainties",
]

# Hard signal words that elevate a sentence into the "this is actual
# guidance" bucket. A sentence needs (a) a forward-looking phrase AND
# (b) at least one of these signals to be surfaced.
GUIDANCE_HARD_SIGNALS = [
    "guidance", "outlook", "raise", "raised", "raises", "raising",
    "lower", "lowered", "lowering", "reaffirm", "reaffirmed",
    "target", "targets", "targeting", "range of", "between",
    "approximately", "in the range", "to be in the range",
]

# Driver-category dictionary: each entry maps a category label to the
# keywords that, if present in the driver sentence, place the sentence
# into that bucket. Order matters -- earlier categories win when a
# sentence matches multiple.
DRIVER_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Pricing / Volume / Mix", [
        "price", "pricing", "volume", "volumes", "mix", "unit",
        "average selling price", "asp",
    ]),
    ("Foreign Exchange", [
        "foreign exchange", "currency", "fx", "translation",
    ]),
    ("Tariffs / Trade", [
        "tariff", "section 301", "sanction", "export control",
    ]),
    ("Cost Inflation / Input Costs", [
        "cost inflation", "raw material", "freight", "logistics",
        "energy cost", "labor cost", "input cost",
    ]),
    ("Restructuring / One-Time", [
        "restructuring", "severance", "facility closure", "impairment",
        "one-time", "one time",
    ]),
    ("Working Capital", [
        "working capital", "receivable", "inventory", "payable", "dso",
    ]),
    ("Capex / Investment", [
        "capital expenditure", "capex", "capacity expansion",
        "property and equipment",
    ]),
    ("Acquisitions / Divestitures", [
        "acquisition", "divestiture", "disposition", "purchase price",
    ]),
    ("Segment / Geographic Mix", [
        "segment", "geographic", "region", "americas", "europe",
        "greater china", "japan", "specialty", "rubber", "products",
        "services", "subscription",
    ]),
]


# A small, curated mapping. Each topic gets one sentence that explains
# why the topic moves a stock.
TOPIC_WHY: dict[str, str] = {
    "Liquidity and Capital Resources":
        "Liquidity headroom dictates buyback pace, M&A optionality, and resilience to working-capital shocks.",
    "Debt and Covenants":
        "Covenant compliance and the maturity tower drive refinancing risk and fixed-charge coverage.",
    "Risk Factors":
        "Material changes to Item 1A often foreshadow disclosures management expects to lean on later.",
    "Legal Proceedings":
        "Active litigation can crystallize as cash outflows or constrain product and commercial decisions.",
    "Controls and Procedures":
        "Internal control issues affect filing reliability and may require remediation spend or restatement.",
    "Customer Concentration":
        "Concentration creates step-function downside if a top customer rolls off or pushes back on pricing.",
    "Revenue Recognition":
        "Policy changes can shift the timing of recognized revenue versus economic delivery.",
    "Segment Performance":
        "Segment mix typically explains the bulk of consolidated YoY change.",
    "Capital Allocation":
        "Buyback pace and dividend policy signal management's confidence and capital plans.",
    "Guidance / Outlook":
        "Guidance is the single most actionable forward signal in the filing.",
    "Tariffs / Geopolitical Risk":
        "Tariffs hit gross margin directly unless pass-through pricing is achievable.",
    "Management Changes":
        "Executive transitions affect strategic continuity and incentive alignment.",
    "Key Financials":
        "Headline P&L movements anchor the period's narrative.",
    "Related Party Transactions":
        "Related-party arrangements warrant independent scrutiny of pricing and terms.",
}


# Topic -> list of (metric_label, source) pairs the card should display.
# `source` is one of "income", "cashflow", "balance".
TOPIC_METRICS: dict[str, list[tuple[str, str]]] = {
    "Liquidity and Capital Resources": [
        ("Cash & Equivalents", "balance"),
        ("Short-Term Investments", "balance"),
        ("Operating Cash Flow", "cashflow"),
    ],
    "Debt and Covenants": [
        ("Long-Term Debt", "balance"),
        ("Current Portion of Debt", "balance"),
    ],
    "Capital Allocation": [
        ("Stock Repurchases", "cashflow"),
        ("Dividends Paid", "cashflow"),
        ("Capital Expenditures", "cashflow"),
    ],
    "Key Financials": [
        ("Net Revenue", "income"),
        ("Gross Profit", "income"),
        ("Operating Income", "income"),
        ("Net Income", "income"),
        ("Diluted EPS", "income"),
    ],
    "Segment Performance": [
        ("Net Revenue", "income"),
        ("Gross Profit", "income"),
    ],
    "Revenue Recognition": [
        ("Net Revenue", "income"),
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriverNote:
    category: str
    excerpt: str  # raw source sentence
    causal_phrase: str
    headline: str = ""  # analyst-voice polished bullet


@dataclass
class GuidanceNote:
    metric_hint: str  # "Revenue", "Margin", "EPS", "Cash Flow", "Capex", "EBITDA", "General"
    excerpt: str  # raw source sentence
    headline: str = ""  # analyst-voice polished bullet


@dataclass
class NumberLine:
    """One numeric line on a topic sentiment card."""

    label: str
    value: str  # display string, e.g. "$45.6B"
    delta: Optional[str]  # display string, e.g. "+12.8%" or None


# ---------------------------------------------------------------------------
# Sentence-level helpers
# ---------------------------------------------------------------------------


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _sentences(text: str) -> list[str]:
    """Crude sentence splitter -- good enough for SEC prose."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _find_sentence_with(text: str, phrase: str) -> Optional[str]:
    """Return the first sentence in `text` containing `phrase`."""
    low = text.lower()
    if phrase not in low:
        return None
    for sent in _sentences(text):
        if phrase in sent.lower():
            # Clip extremely long sentences so the UI stays readable.
            return sent if len(sent) <= 500 else sent[:497].rstrip() + "…"
    return None


def _classify_driver_category(text: str) -> str:
    low = text.lower()
    for label, words in DRIVER_CATEGORIES:
        if any(w in low for w in words):
            return label
    return "Other"


def _classify_guidance_metric(text: str) -> str:
    low = text.lower()
    if "eps" in low or "earnings per share" in low:
        return "EPS"
    if "ebitda" in low:
        return "EBITDA"
    if "margin" in low:
        return "Margin"
    if "capex" in low or "capital expenditure" in low:
        return "Capex"
    if "cash flow" in low or "free cash flow" in low:
        return "Cash Flow"
    if "revenue" in low or "sales" in low:
        return "Revenue"
    return "General"


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


_TOC_LIKE = re.compile(
    r"\b(?:see (?:note|item)|table of contents|page \d+|item \d+\.\d+)\b",
    re.IGNORECASE,
)


def _looks_like_clean_sentence(sent: str) -> bool:
    """Reject sentence fragments that would read badly as analyst bullets."""
    s = sent.strip()
    if not s or len(s) < 40 or len(s) > 500:
        return False
    # Must start with a capital letter or open paren -- avoids mid-thought
    # fragments like "the result of ..." or "to which we ..."
    if not (s[0].isupper() or s[0] in "(\""):
        return False
    # Reject TOC / cross-reference language.
    if _TOC_LIKE.search(s):
        return False
    return True


def extract_drivers(paragraphs, *, max_per_category: int = 2) -> list[DriverNote]:
    """Find driver sentences in the latest filing, bucketed by category.

    Each returned DriverNote includes an analyst-voice ``headline`` that
    the UI renders by default. The raw source ``excerpt`` is only shown
    inside the collapsed evidence expander.
    """
    by_cat: dict[str, list[DriverNote]] = defaultdict(list)
    seen_keys: set[str] = set()

    for p in paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        low = text.lower()
        for phrase in CAUSAL_PHRASES:
            if phrase not in low:
                continue
            sent = _find_sentence_with(text, phrase)
            if not sent or not _looks_like_clean_sentence(sent):
                continue
            key = re.sub(r"\W+", "", sent.lower())[:80]
            if key in seen_keys:
                continue
            cat = _classify_driver_category(sent)
            if len(by_cat[cat]) >= max_per_category:
                break
            seen_keys.add(key)
            note = DriverNote(category=cat, excerpt=sent, causal_phrase=phrase)
            note.headline = polish_driver_headline(note)
            by_cat[cat].append(note)
            break  # one driver per paragraph

    out: list[DriverNote] = []
    for label, _ in DRIVER_CATEGORIES:
        out.extend(by_cat.get(label, []))
    out.extend(by_cat.get("Other", []))
    return out


# ---------------------------------------------------------------------------
# Driver -> analyst-voice headline
# ---------------------------------------------------------------------------


_POS_VERBS = (
    "higher", "increased", "growth", "favorable", "favorably",
    "expanded", "improved", "stronger", "rose", "grew", "benefited",
    "supported", "tailwind",
)
_NEG_VERBS = (
    "lower", "declined", "decreased", "unfavorable", "unfavorably",
    "weaker", "compressed", "pressured", "fell", "weighed",
    "headwind", "softness",
    # Keep "soft" as a word match so we don't trip on "software" or
    # "softness" (already covered). Compiled below as a word-boundary
    # regex.
)

# Compile word-boundary patterns so "favorable" does not match
# "unfavorably" and "soft" does not match "software".
_POS_RE = re.compile(r"\b(?:" + "|".join(_POS_VERBS) + r")\b", re.IGNORECASE)
_NEG_RE = re.compile(r"\b(?:" + "|".join(_NEG_VERBS) + r"|soft)\b", re.IGNORECASE)


def _direction(text: str) -> str:
    """Return "positive" | "negative" | "mixed" | "neutral"."""
    pos = bool(_POS_RE.search(text))
    neg = bool(_NEG_RE.search(text))
    if pos and neg:
        return "mixed"
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "neutral"


def _cause_clause(sent: str, causal_phrase: str) -> Optional[str]:
    """Pull a short cause clause from after the causal phrase."""
    low = sent.lower()
    idx = low.find(causal_phrase)
    if idx < 0:
        return None
    rest = sent[idx + len(causal_phrase):].lstrip(" ,;:")
    # Truncate at any offset-connector phrase within the clause so the
    # primary driver is not contaminated by the counter-trend description.
    rest_low = rest.lower()
    for offset_marker in ("partially offset by", ", offset by", " offset by"):
        oi = rest_low.find(offset_marker)
        if oi > 0:
            rest = rest[:oi].rstrip(", ")
            break
    cut = re.search(r"[.;]", rest)
    if cut:
        rest = rest[: cut.start()]
    rest = rest.strip()
    # Lowercase first letter for grammatical continuity (unless it's an
    # acronym like "FX" / "EMEA").
    if rest and rest[0].isupper() and not (
        len(rest) >= 2 and rest[1].isupper()
    ):
        rest = rest[0].lower() + rest[1:]
    # Strip awkward trailing connector words.
    rest = re.sub(r"\s+(?:and|but|while|although|though)$", "", rest)
    if not 8 <= len(rest) <= 150:
        return None
    return rest


def polish_driver_headline(d: DriverNote) -> str:
    """Produce an analyst-voice headline bullet for a driver.

    The headline is templated by category + detected subject + direction
    and ends with the trimmed cause clause from the source. Falls back
    to a tidy generic phrasing if nothing distinctive is detected.
    """
    low = d.excerpt.lower()
    # Compute direction from the primary clause only — strip the "partially
    # offset by ..." tail so a counter-trend positive doesn't cancel the
    # lead-driver signal ("unfavorably impacted by FX, partially offset by
    # higher margin" should read as negative, not mixed).
    primary_clause = d.excerpt
    for offset_marker in ("partially offset by", ", offset by", " offset by"):
        oi = low.find(offset_marker)
        if oi > 0:
            primary_clause = d.excerpt[:oi]
            break
    direction = _direction(primary_clause)
    cause = _cause_clause(d.excerpt, d.causal_phrase)
    # When the matched phrase is an offset connector (e.g. "partially
    # offset by"), the cause clause describes the *counter-trend*, not
    # the primary driver. Surface it parenthetically so the reader
    # doesn't confuse the offset for the lead.
    if d.causal_phrase in ("partially offset by", "offset by"):
        cause_tail = f" (partially offset by {cause})" if cause else ""
    else:
        cause_tail = f" — {cause}" if cause else ""

    cat = d.category
    # Pricing / Volume / Mix
    if cat == "Pricing / Volume / Mix":
        if "volume" in low and direction == "positive":
            return f"Higher volumes contributed to revenue growth{cause_tail}."
        if "volume" in low and direction == "negative":
            return f"Lower volumes weighed on revenue{cause_tail}."
        if ("price" in low or "pricing" in low or "asp" in low) and direction == "positive":
            return f"Pricing actions supported revenue and margins{cause_tail}."
        if ("price" in low or "pricing" in low or "asp" in low) and direction == "negative":
            return f"Pricing pressure compressed margins{cause_tail}."
        if "mix" in low and direction == "positive":
            return f"Favorable product/customer mix lifted reported results{cause_tail}."
        if "mix" in low and direction == "negative":
            return f"Unfavorable mix pressured reported results{cause_tail}."
        return f"Price, volume, and mix dynamics drove the period's variance{cause_tail}."

    if cat == "Foreign Exchange":
        if direction == "negative":
            return f"Foreign currency translation was a headwind{cause_tail}."
        if direction == "positive":
            return f"Foreign currency translation provided a tailwind{cause_tail}."
        return f"Foreign currency translation moved the reported result{cause_tail}."

    if cat == "Tariffs / Trade":
        if direction == "negative":
            return f"Tariffs and trade actions weighed on the period{cause_tail}."
        return f"Tariffs and trade actions are noted as a factor in the period{cause_tail}."

    if cat == "Cost Inflation / Input Costs":
        if direction == "negative":
            return f"Higher input costs and inflation pressured margins{cause_tail}."
        return f"Input cost dynamics influenced the period's margins{cause_tail}."

    if cat == "Restructuring / One-Time":
        return f"Restructuring or one-time items affected the period's results{cause_tail}."

    if cat == "Working Capital":
        if direction == "negative":
            return f"Working capital was a use of cash this period{cause_tail}."
        if direction == "positive":
            return f"Working capital was a source of cash this period{cause_tail}."
        return f"Working capital movements affected operating cash flow{cause_tail}."

    if cat == "Capex / Investment":
        if direction == "positive":
            return f"Capex stepped up this period{cause_tail}."
        if direction == "negative":
            return f"Capex was lower this period{cause_tail}."
        return f"Capital investment is highlighted as a factor in the period{cause_tail}."

    if cat == "Acquisitions / Divestitures":
        return f"Portfolio actions (acquisitions / divestitures) shaped the period{cause_tail}."

    if cat == "Segment / Geographic Mix":
        if direction == "positive":
            return f"Segment / regional mix was a positive contributor{cause_tail}."
        if direction == "negative":
            return f"Segment / regional mix weighed on the period{cause_tail}."
        return f"Segment / regional mix shaped the period's result{cause_tail}."

    # Generic fallback.
    return f"{cat} contributed to the period's variance{cause_tail}."


def _is_real_guidance(sent: str) -> bool:
    """Reject safe-harbor boilerplate; require a quantitative anchor or hard signal."""
    low = sent.lower()
    # Reject obvious risk-factor / safe-harbor boilerplate sentences.
    if any(b in low for b in GUIDANCE_BOILERPLATE_PHRASES):
        return False
    has_pct = bool(re.search(r"\d{1,3}(?:\.\d+)?\s*%", sent))
    has_dollar = bool(re.search(r"\$\s*\d", sent))
    has_signal = any(s in low for s in GUIDANCE_HARD_SIGNALS)
    has_range_lang = "range of" in low or "between $" in low or " to $" in low
    # At least one hard signal or a quantitative anchor must be present.
    return has_pct or has_dollar or has_signal or has_range_lang


def _polish_guidance_headline(sent: str, metric: str) -> str:
    """Convert a guidance sentence into a clean analyst-style headline."""
    low = sent.lower()
    direction = (
        "raised" if "raise" in low or "raised" in low
        else "lowered" if "lower" in low or "lowered" in low or "cut" in low
        else "reaffirmed" if "reaffirm" in low
        else "set"
    )
    metric_l = metric if metric != "General" else "outlook"
    # Try to lift a numeric snippet (range or value) to embed in the headline.
    # Use a lookahead stop so decimal points inside numbers (e.g. "45.5%") are
    # not mistaken for sentence terminators.
    num = re.search(
        r"(?:range of|between|approximately|approx\.|about)\s+.+?(?=,\s*[a-z]|\.\s|\.$|;|$)",
        sent, flags=re.IGNORECASE,
    )
    if num:
        snippet = num.group(0).strip().rstrip(".")
        return f"Management {direction} {metric_l} guidance, {snippet}."
    pct = re.search(r"-?\d{1,3}(?:\.\d+)?\s*%[^.,;]{0,60}", sent)
    if pct:
        return f"Management {direction} {metric_l} guidance ({pct.group(0).strip()})."
    dollar = re.search(r"\$\s*[\d,]+(?:\.\d+)?\s*(?:million|billion)?", sent, flags=re.IGNORECASE)
    if dollar:
        return f"Management {direction} {metric_l} guidance ({dollar.group(0).strip()})."
    return f"Management {direction} {metric_l} guidance."


def extract_guidance(paragraphs, *, max_per_metric: int = 2) -> list[GuidanceNote]:
    """Find forward-looking statements grouped by metric hint.

    Requires both a forward-looking phrase *and* either a quantitative
    anchor (%, $, range) or a hard guidance signal word (raise / lower /
    reaffirm / target / range of / approximately). Safe-harbor and
    "forward-looking statements"-style boilerplate is filtered out.
    """
    by_metric: dict[str, list[GuidanceNote]] = defaultdict(list)
    seen_keys: set[str] = set()

    for p in paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        low = text.lower()
        for phrase in GUIDANCE_PHRASES:
            if phrase not in low:
                continue
            sent = _find_sentence_with(text, phrase)
            if not sent or not _looks_like_clean_sentence(sent):
                continue
            if not _is_real_guidance(sent):
                continue
            key = re.sub(r"\W+", "", sent.lower())[:80]
            if key in seen_keys:
                continue
            metric = _classify_guidance_metric(sent)
            if len(by_metric[metric]) >= max_per_metric:
                break
            seen_keys.add(key)
            headline = _polish_guidance_headline(sent, metric)
            by_metric[metric].append(
                GuidanceNote(metric_hint=metric, excerpt=sent, headline=headline)
            )
            break

    metric_order = ["Revenue", "EBITDA", "Margin", "EPS", "Cash Flow", "Capex", "General"]
    out: list[GuidanceNote] = []
    for m in metric_order:
        out.extend(by_metric.get(m, []))
    return out


def segment_commentary(
    paragraphs, segment_names: Iterable[str], *, max_per_segment: int = 1
) -> dict[str, str]:
    """Return {segment_name: quoted driver sentence} for each segment found."""
    out: dict[str, str] = {}
    for name in segment_names:
        low_name = name.lower()
        for p in paragraphs:
            text = p.text if hasattr(p, "text") else str(p)
            low = text.lower()
            if low_name not in low:
                continue
            for phrase in CAUSAL_PHRASES:
                if phrase not in low:
                    continue
                sent = _find_sentence_with(text, phrase)
                if sent and low_name in sent.lower() and 40 <= len(sent) <= 500:
                    out[name] = sent
                    break
            if name in out:
                break
        if name in out and len(out) >= max_per_segment * 64:
            # Defensive upper bound; loop already short.
            break
    return out


# ---------------------------------------------------------------------------
# Topic-card narrative builder
# ---------------------------------------------------------------------------


def topic_numbers(
    topic: str,
    income: list,
    cashflow: list,
    balance: list,
) -> list[NumberLine]:
    """Pull the XBRL numbers that belong on a given topic card."""
    defs = TOPIC_METRICS.get(topic, [])
    if not defs:
        return []
    by_source = {"income": income, "cashflow": cashflow, "balance": balance}
    lines: list[NumberLine] = []
    for label, source in defs:
        series = by_source.get(source, [])
        m = next((x for x in series if getattr(x, "label", None) == label), None)
        if not m or m.latest_val is None:
            continue
        value = _format_xbrl_value(m.latest_val, m.unit)
        delta = f"{m.delta_pct:+.1f}%" if m.delta_pct is not None else None
        lines.append(NumberLine(label=label, value=value, delta=delta))
    return lines


def topic_narrative(
    topic: str,
    sentiment: str,
    numbers: list[NumberLine],
    driver_quote: Optional[str] = None,  # accepted for API stability, no longer rendered
) -> str:
    """Produce a clean 2-3 sentence interpretation for a topic card.

    The card itself shows numbers as chips below the narrative, so we
    don't repeat them. The narrative reads as analyst voice -- no raw
    filing quotes, no mechanical closers.
    """
    pieces: list[str] = []

    # Sentence 1: a topic-specific framing line that respects the sentiment.
    framing = _topic_framing_sentence(topic, sentiment)
    if framing:
        pieces.append(framing)

    # Sentence 2: a numeric anchor when available. Keeps it short --
    # the chips below carry the full set.
    if numbers:
        primary = numbers[0]
        if primary.delta:
            pieces.append(
                f"{primary.label} of {primary.value} ({primary.delta}) "
                f"anchors the read for this area."
            )
        else:
            pieces.append(
                f"{primary.label} stood at {primary.value} this period."
            )

    # Sentence 3: a brief "Not enough evidence" caveat when we genuinely
    # don't have the signal. This avoids the previous habit of declaring
    # Slightly Positive on weak input.
    if sentiment == "Neutral" and not numbers:
        pieces.append("Not enough disclosure movement to draw a directional read.")

    return " ".join(pieces)


def _topic_framing_sentence(topic: str, sentiment: str) -> str:
    """One-sentence framing that mentions the topic by name."""
    if topic == "Liquidity and Capital Resources":
        if sentiment in ("Positive", "Slightly Positive"):
            return "Liquidity headroom looks more comfortable than a year ago."
        if sentiment in ("Negative", "Slightly Negative"):
            return "Liquidity is incrementally tighter than the comparable period."
        return "Liquidity disclosures held broadly in line with the prior period."

    if topic == "Debt and Covenants":
        if sentiment in ("Negative", "Slightly Negative"):
            return "Leverage and covenant disclosures lean cautionary."
        if sentiment in ("Positive", "Slightly Positive"):
            return "Debt profile and covenant headroom appear constructive."
        return "Debt disclosures were broadly in line."

    if topic == "Risk Factors":
        if sentiment in ("Negative", "Slightly Negative"):
            return "Risk-factor language tightened versus the comparable filing."
        return "Risk-factor disclosures held in line with the prior comparable filing."

    if topic == "Legal Proceedings":
        if sentiment in ("Negative", "Slightly Negative"):
            return "Litigation language moved in a cautionary direction."
        return "Litigation language held broadly stable."

    if topic == "Controls and Procedures":
        if sentiment in ("Negative", "Slightly Negative"):
            return "Internal control disclosures contain new cautionary language."
        return "Internal control disclosures appear stable."

    if topic == "Capital Allocation":
        if sentiment in ("Positive", "Slightly Positive"):
            return "Capital-return cadence was sustained or stepped up."
        if sentiment in ("Negative", "Slightly Negative"):
            return "Capital-return cadence slowed relative to the comparable period."
        return "Capital allocation disclosures were broadly unchanged."

    if topic == "Segment Performance":
        return "Segment-level commentary points to the mix drivers behind the consolidated print."

    if topic == "Revenue Recognition":
        return "Revenue policy and performance-obligation language was refreshed for the period."

    if topic == "Guidance / Outlook":
        return "Forward-looking commentary was updated this filing."

    if topic == "Tariffs / Geopolitical Risk":
        return "Tariff and geopolitical disclosures evolved versus the comparable period."

    if topic == "Customer Concentration":
        return "Customer-concentration disclosures moved versus the prior comparable filing."

    if topic == "Management Changes":
        return "Executive transition / 10b5-1 language is present in the filing."

    if topic == "Key Financials":
        return "Headline financial commentary was updated for the period."

    # Generic fallback that still avoids mechanical wording.
    return f"{topic} disclosures were updated this filing."


def _polarity_closer(sentiment: str) -> str:
    return {
        "Positive": "On balance, the trend supports a constructive read of this area.",
        "Slightly Positive": "Movement here is incrementally constructive.",
        "Neutral": "On balance the read is neutral versus the prior period.",
        "Mixed": "Signals here are inconsistent and warrant a closer read of the source.",
        "Slightly Negative": "Movement here is mildly cautionary.",
        "Negative": "The trend bears watching as a downside catalyst.",
    }.get(sentiment, "")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _format_xbrl_value(val: float, unit: str) -> str:
    if unit == "USD":
        absv = abs(val)
        if absv >= 1e9:
            return f"${val/1e9:.2f}B"
        if absv >= 1e6:
            return f"${val/1e6:.1f}M"
        return f"${val:,.0f}"
    if unit == "USD/shares":
        return f"${val:.2f}"
    if unit == "shares":
        return f"{val/1e6:.1f}M sh"
    return f"{val:,.2f}"


# ---------------------------------------------------------------------------
# Fallback segment scanner (used when the HTML table parser returns nothing)
# ---------------------------------------------------------------------------

# Broad set of segment / line-item names we've seen in the wild. Used by the
# MD&A fallback when structured table parsing returns nothing -- so we still
# surface "Specialty / Rubber" for an industrial like OEC, "Cloud" / "Office"
# for software-like filers, etc.
FALLBACK_SEGMENT_NAMES = [
    # Industrials (e.g. OEC)
    "Specialty", "Rubber",
    # Apple / consumer hardware
    "iPhone", "Mac", "iPad", "Wearables", "Services", "Products",
    # Consumer software / cloud
    "Cloud", "Software", "Hardware", "Subscription", "License",
    "Subscription Revenue",
    # Geographic
    "Americas", "Europe", "Greater China", "Japan",
    "Rest of Asia Pacific", "EMEA",
    # SaaS metrics
    "RPO", "ARR", "Net Revenue Retention", "Dollar-Based Net Retention",
]


@dataclass
class FallbackSegmentNote:
    name: str
    headline: str  # analyst-voice line
    excerpt: str  # raw source sentence behind the line


def _segment_headline(name: str, sent: str) -> str:
    """Turn a sentence mentioning a segment into an analyst-voice line."""
    direction = _direction(sent)
    low = sent.lower()
    has_pct = re.search(r"\d{1,3}(?:\.\d+)?\s*%", sent)
    has_dollar = re.search(r"\$\s*[\d,]+(?:\.\d+)?\s*(?:million|billion)?", sent, flags=re.IGNORECASE)
    figure = None
    if has_dollar:
        figure = has_dollar.group(0).strip()
    elif has_pct:
        figure = has_pct.group(0).strip()

    if direction == "positive":
        verb = "grew"
    elif direction == "negative":
        verb = "declined"
    else:
        verb = "moved"

    if "rpo" in low or "remaining performance obligation" in low:
        return f"{name} {verb} this period" + (f" ({figure})" if figure else "") + "."
    if "retention" in low:
        return f"{name}: retention disclosure updated" + (f" ({figure})" if figure else "") + "."
    if "subscription" in low:
        return f"{name} subscription revenue {verb}" + (f" ({figure})" if figure else "") + "."

    base = f"{name} revenue {verb}" if "revenue" in low or "sales" in low else f"{name} segment results {verb}"
    if figure:
        base += f" ({figure})"
    return base + "."


def extract_fallback_segments(paragraphs) -> list[FallbackSegmentNote]:
    """Scan MD&A text for recurring segment names when table parsing fails.

    For each candidate segment name, find the *first* paragraph that
    mentions both the segment and a revenue / sales / RPO / retention
    word, then produce a polished one-line headline plus a raw evidence
    quote. Returns at most one note per segment.
    """
    notes: list[FallbackSegmentNote] = []
    seen: set[str] = set()
    for name in FALLBACK_SEGMENT_NAMES:
        if name.lower() in seen:
            continue
        for p in paragraphs:
            text = p.text if hasattr(p, "text") else str(p)
            low = text.lower()
            if name.lower() not in low:
                continue
            if not any(
                k in low for k in (
                    "revenue", "net sales", "rpo",
                    "remaining performance obligation", "retention",
                    "subscription", "segment",
                )
            ):
                continue
            for sent in _sentences(text):
                if name.lower() not in sent.lower():
                    continue
                if not _looks_like_clean_sentence(sent):
                    continue
                # Require the sentence itself to contain a topical signal,
                # not just the surrounding paragraph.
                if not any(
                    k in sent.lower() for k in (
                        "revenue", "sales", "rpo", "retention",
                        "subscription", "segment",
                    )
                ):
                    continue
                notes.append(
                    FallbackSegmentNote(
                        name=name,
                        headline=_segment_headline(name, sent),
                        excerpt=sent,
                    )
                )
                seen.add(name.lower())
                break
            if name.lower() in seen:
                break
    return notes


def _humanize_list(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
