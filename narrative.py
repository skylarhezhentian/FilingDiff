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
    """Structured analyst-style driver card.

    Each card is one synthesised view of a single driver category --
    duplicates are merged at synthesis time, not at render time. The
    card answers: what changed, in which direction, by how much, due to
    what, with what financial impact, and what an analyst should take
    away.
    """

    category: str
    headline: str  # one-sentence summary with direction + magnitude
    driver: str  # "Driver: ..." bullet body
    financial_impact: str  # "Financial impact: ..." bullet body
    investment_read: str  # "Investment read: ..." bullet body
    direction: str  # "increased" | "decreased" | "improved" | "worsened" | "mixed" | "unclear"
    confidence: str  # "high" | "medium" | "low"
    excerpts: list  # list[str] raw source sentences for the evidence expander


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
    # OEC-style segment split: if a sentence compares both Specialty and
    # Rubber segments, classify it as Segment / Geographic Mix even though
    # it contains "volume" (which would otherwise win Pricing/Volume/Mix).
    if "specialty" in low and "rubber" in low:
        return "Segment / Geographic Mix"
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


# ---------------------------------------------------------------------------
# Driver synthesis (richer than the old polished-headline approach)
#
# Each driver card now answers six questions:
#   - What changed?
#   - Direction (increased / decreased / improved / worsened / compressed / expanded)
#   - Magnitude, if available
#   - Cause
#   - Financial impact on revenue / margin / earnings / cash / liquidity
#   - Why it matters (investment read)
#
# We never emit vague verbs ("moved", "affected", "contributed") on their
# own. If we cannot determine direction, we say so explicitly.
# ---------------------------------------------------------------------------


# Stricter directional verb sets used at the *card* level. Word-boundary
# matched so "favorable" never lights up on "unfavorably".
_POS_VERBS = (
    "higher", "increased", "growth", "favorable", "favorably",
    "expanded", "improved", "stronger", "rose", "grew", "benefited",
    "supported", "tailwind",
)
_NEG_VERBS = (
    "lower", "declined", "decreased", "unfavorable", "unfavorably",
    "weaker", "compressed", "pressured", "fell", "weighed",
    "headwind", "softness",
)
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
    # Truncate at any offset-connector phrase so the cause clause reflects
    # the primary driver, not the counter-trend.
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
    if rest and rest[0].isupper() and not (len(rest) >= 2 and rest[1].isupper()):
        rest = rest[0].lower() + rest[1:]
    rest = re.sub(r"\s+(?:and|but|while|although|though)$", "", rest)
    if not 8 <= len(rest) <= 160:
        return None
    return rest


# Subject regexes -- "what is moving" in the sentence.
_SUBJ_VOLUME = re.compile(r"\bvolume[s]?\b", re.IGNORECASE)
_SUBJ_PRICE = re.compile(
    r"\b(?:price[s]?|pricing|asp|average\s+selling\s+price|contractual\s+pric)",
    re.IGNORECASE,
)
_SUBJ_MIX = re.compile(
    r"\b(?:product\s+mix|regional\s+mix|geographic\s+mix|customer\s+mix|mix)\b",
    re.IGNORECASE,
)
_SUBJ_RECV = re.compile(r"\b(?:accounts\s+receivable|receivables|dso|days\s+sales)\b", re.IGNORECASE)
_SUBJ_INV = re.compile(r"\b(?:inventor(?:y|ies))\b", re.IGNORECASE)
_SUBJ_PAY = re.compile(r"\b(?:accounts\s+payable|payables)\b", re.IGNORECASE)
_SUBJ_FEED = re.compile(r"\b(?:feedstock|raw\s+material|oil\s+price|pass[-\s]through)\b", re.IGNORECASE)
_SUBJ_FREIGHT = re.compile(r"\b(?:freight|logistics|energy\s+cost|labor\s+cost)\b", re.IGNORECASE)
_SUBJ_SPECIALTY = re.compile(r"\bspecialty\s+carbon\b|\bspecialty\b", re.IGNORECASE)
_SUBJ_RUBBER = re.compile(r"\brubber\s+carbon\b|\brubber\b", re.IGNORECASE)

_UNFAVORABLE_RE = re.compile(r"\bunfavorab", re.IGNORECASE)
_FAVORABLE_RE = re.compile(r"(?<!un)\bfavorab", re.IGNORECASE)
_TAILWIND_RE = re.compile(r"\btailwind\b|\b(?:benefit|benefited)\b", re.IGNORECASE)
_HEADWIND_RE = re.compile(r"\bheadwind\b|\b(?:weighed|pressured|hurt)\b", re.IGNORECASE)


@dataclass
class _CategorySignals:
    """Per-category bucket of evidence accumulated across the filing."""

    sentences: list  # list[str]
    causes: list  # list[str] -- extracted cause-clause fragments
    pos_hits: int = 0
    neg_hits: int = 0

    def __init__(self) -> None:
        self.sentences = []
        self.causes = []
        self.pos_hits = 0
        self.neg_hits = 0

    def add(self, sent: str, cause: Optional[str], dir_in_primary: str) -> None:
        if sent not in self.sentences:
            self.sentences.append(sent)
        if cause and cause not in self.causes:
            self.causes.append(cause)
        if dir_in_primary == "positive":
            self.pos_hits += 1
        elif dir_in_primary == "negative":
            self.neg_hits += 1
        elif dir_in_primary == "mixed":
            self.pos_hits += 1
            self.neg_hits += 1


def _collect_signals(paragraphs) -> dict[str, _CategorySignals]:
    """Walk paragraphs once and bucket evidence sentences into categories.

    Pass 1: every sentence containing a causal phrase ("driven by", "due
    to", ...) is classified into exactly one category. Direction is read
    from the *primary* clause only -- anything after a "partially offset
    by" connector is stripped so an offset positive cannot cancel a
    lead-driver negative.

    Pass 2: pick up sentences that lack a causal phrase but still convey
    a clear directional + magnitude signal for a specific driver subject
    (FX tailwind / headwind, working-capital use / source, cost
    inflation, etc.). This catches lines like "FX contributed a tailwind
    of approximately $1.0 million" which an SEC writer is unlikely to
    decorate with a causal connector.
    """
    out: dict[str, _CategorySignals] = defaultdict(_CategorySignals)
    seen_sentences: set[str] = set()

    # Pass 1 — causal-phrase based.
    for p in paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        for sent in _sentences(text):
            slow = sent.lower()
            if not _looks_like_clean_sentence(sent):
                continue
            matched_phrase = next(
                (cp for cp in CAUSAL_PHRASES if cp in slow), None
            )
            if not matched_phrase:
                continue
            cat = _classify_driver_category(sent)
            primary = sent
            for offset_marker in ("partially offset by", ", offset by", " offset by"):
                oi = slow.find(offset_marker)
                if oi > 0:
                    primary = sent[:oi]
                    break
            cause = _cause_clause(sent, matched_phrase)
            out[cat].add(sent, cause, _direction(primary))
            seen_sentences.add(sent)

    # Pass 2 — subject + direction + magnitude, no causal phrase required.
    for p in paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        for sent in _sentences(text):
            if sent in seen_sentences:
                continue
            if not _looks_like_clean_sentence(sent):
                continue
            slow = sent.lower()
            cat = _supplemental_category(sent, slow)
            if not cat:
                continue
            out[cat].add(sent, None, _direction(sent))
            seen_sentences.add(sent)

    return out


def _supplemental_category(sent: str, slow: str) -> Optional[str]:
    """Classify a non-causal sentence into a driver category when the
    subject is unambiguous and we have a direction or magnitude signal.
    Returns None otherwise so the sentence is skipped.
    """
    has_dollar = bool(_RE_DOLLAR.search(sent))
    has_pct = bool(_RE_PCT.search(sent))
    has_kmt = bool(_RE_KMT.search(sent))
    has_mag = has_dollar or has_pct or has_kmt

    # Foreign Exchange -- subject + direction word is enough; magnitude
    # bumps confidence later.
    if re.search(r"\b(?:foreign\s+exchange|currency\s+translation|currency|\bfx\b|translation)\b", slow):
        if (
            _TAILWIND_RE.search(sent) or _HEADWIND_RE.search(sent)
            or _FAVORABLE_RE.search(sent) or _UNFAVORABLE_RE.search(sent)
            or "weakened" in slow or "strengthened" in slow
            or has_mag
        ):
            return "Foreign Exchange"

    # Working Capital -- explicit "use of cash" / "source of cash" wording.
    if re.search(r"\bworking\s+capital\b", slow):
        if (
            "use of cash" in slow or "cash use" in slow
            or "source of cash" in slow or "released" in slow
            or _SUBJ_RECV.search(slow) or _SUBJ_INV.search(slow)
        ):
            return "Working Capital"

    # Cost Inflation -- input-cost subject with a direction signal.
    if _SUBJ_FEED.search(slow) or _SUBJ_FREIGHT.search(slow) or "input cost" in slow:
        if _POS_RE.search(sent) or _NEG_RE.search(sent) or has_mag:
            return "Cost Inflation / Input Costs"

    # Segment-level OEC / multi-region split.
    if "specialty" in slow and "rubber" in slow:
        return "Segment / Geographic Mix"

    return None


# Numeric magnitude extraction ------------------------------------------------

_RE_KMT = re.compile(r"(-?\d+(?:\.\d+)?)\s*kmt\b", re.IGNORECASE)
_RE_DOLLAR = re.compile(
    r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)
_RE_PCT = re.compile(r"-?\d+(?:\.\d+)?\s*%")
_RE_BPS = re.compile(r"-?\d+\s*(?:bps|basis\s+points)\b", re.IGNORECASE)


def _first_dollar(text: str) -> Optional[str]:
    m = _RE_DOLLAR.search(text)
    return m.group(0).strip() if m else None


def _first_pct(text: str) -> Optional[str]:
    m = _RE_PCT.search(text)
    return m.group(0).strip() if m else None


def _first_kmt(text: str) -> Optional[str]:
    m = _RE_KMT.search(text)
    if not m:
        return None
    return f"{m.group(1)} kmt"


def _volume_change(text: str) -> Optional[tuple[str, Optional[str]]]:
    """Detect 'volume increased X kmt ... to Y kmt' patterns.

    Returns (change_amount, total_after) or None. ``total_after`` may be
    None if only the delta was disclosed.
    """
    # "increased 4.8 kmt year over year to 256.5 kmt"
    m = re.search(
        r"volume[s]?\s+(?:increased|decreased|grew|declined|rose|fell)\s+"
        r"(?:by\s+)?(\d+(?:\.\d+)?)\s*kmt"
        r"(?:[^.]{0,80}?to\s+(\d+(?:\.\d+)?)\s*kmt)?",
        text, re.IGNORECASE,
    )
    if m:
        change = f"{m.group(1)} kmt"
        total = f"{m.group(2)} kmt" if m.group(2) else None
        return change, total
    return None


# Per-category synthesizers ---------------------------------------------------


def _conf_from_signals(sigs: _CategorySignals, has_magnitude: bool) -> str:
    n_sent = len(sigs.sentences)
    if has_magnitude and n_sent >= 1 and (sigs.pos_hits + sigs.neg_hits) >= 1:
        return "high"
    if n_sent >= 1 and (sigs.pos_hits + sigs.neg_hits) >= 1:
        return "medium"
    return "low"


def _undetermined(category: str, sigs: _CategorySignals) -> Optional[DriverNote]:
    """Emit an honest 'direction not detected' card instead of a vague one."""
    if not sigs.sentences:
        return None
    return DriverNote(
        category=category,
        headline=f"{category}: direction not clearly detected — see source filing.",
        driver="",
        financial_impact="",
        investment_read="",
        direction="unclear",
        confidence="low",
        excerpts=sigs.sentences[:3],
    )


def _join_clauses(parts: list[str]) -> str:
    parts = [p.strip().rstrip(".") for p in parts if p and p.strip()]
    if not parts:
        return ""
    s = ". ".join(parts)
    return s.rstrip(".") + "."


def _synth_pricing_volume_mix(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    low = text_all.lower()

    # Volume direction & magnitude.
    vc = _volume_change(text_all)
    vol_dir = None
    for s in sigs.sentences:
        if not _SUBJ_VOLUME.search(s):
            continue
        if re.search(r"\bvolume[s]?\s+(?:increased|grew|rose|expanded|higher)\b", s, re.IGNORECASE):
            vol_dir = "up"
        elif re.search(r"\bvolume[s]?\s+(?:decreased|declined|fell|lower|weaker)\b", s, re.IGNORECASE):
            vol_dir = "down"
    if not vol_dir and vc:
        # Fallback: assume vc reflects whatever the sentence verb said.
        # _volume_change requires a movement verb in its regex, so this
        # branch typically infers "up" for increases. Use a conservative
        # secondary check.
        for s in sigs.sentences:
            if "volume" in s.lower() and _DIRECTION_DOWN_RE.search(s):
                vol_dir = "down"; break
            if "volume" in s.lower() and _DIRECTION_UP_RE.search(s):
                vol_dir = "up"; break

    # Price / mix direction.
    price_neg = bool(re.search(r"\b(?:lower|unfavorab|reduced)\s+(?:contractual\s+)?pric", low))
    price_pos = bool(re.search(r"\b(?:higher|favorab|stronger)\s+(?:contractual\s+)?pric", low))
    mix_neg = bool(re.search(r"unfavorable[^.]{0,40}\bmix\b|mix\b[^.]{0,40}\bunfavorab", low))
    mix_pos = bool(re.search(r"\bfavorab[^.]{0,40}\bmix\b|\bmix\b[^.]{0,40}\bfavorab", low))

    has_magnitude = bool(vc) or bool(_first_pct(text_all))
    confidence = _conf_from_signals(sigs, has_magnitude)

    # Build the driver bullet in two layered clauses so the prose flows.
    volume_clause = ""
    if vol_dir == "up" and vc:
        change, total = vc
        volume_clause = (
            f"Volume increased {change} year over year to {total}" if total
            else f"Volume increased {change} year over year"
        )
    elif vol_dir == "down" and vc:
        change, total = vc
        volume_clause = (
            f"Volume decreased {change} year over year to {total}" if total
            else f"Volume decreased {change} year over year"
        )
    elif vol_dir == "up":
        volume_clause = "Volume was higher year over year"
    elif vol_dir == "down":
        volume_clause = "Volume was lower year over year"
    if volume_clause and sigs.causes:
        volume_clause += f", driven by {sigs.causes[0]}"
    pressure_clause = ""
    if mix_neg or price_neg:
        bits = []
        if price_neg:
            bits.append("lower contractual pricing")
        if mix_neg:
            bits.append("unfavorable product / regional mix")
        pressure_clause = ", ".join(bits) + " pressured profitability"
    if volume_clause and pressure_clause:
        driver = f"{volume_clause}. {pressure_clause[0].upper()}{pressure_clause[1:]}."
    elif volume_clause:
        driver = f"{volume_clause}."
    elif pressure_clause:
        driver = f"{pressure_clause[0].upper()}{pressure_clause[1:]}."
    else:
        driver = ""

    # Headline + financial impact + read.
    if vol_dir == "up" and (price_neg or mix_neg):
        headline = "Volumes improved, but mix and pricing hurt profitability."
        direction = "mixed"
        fi = (
            "Despite higher volume, revenue declined because product / regional "
            "mix and contractual pricing were unfavorable. Gross profit and "
            "operating income compressed as a result."
        )
        read = (
            "Demand is not the main problem; margin quality and pricing are "
            "the pressure points. Watch for stabilisation in contractual "
            "pricing and a mix shift back toward higher-spec products."
        )
    elif vol_dir == "up" and (price_pos or mix_pos):
        headline = "Volumes and pricing both supported revenue and margin."
        direction = "increased"
        fi = "Higher volume combined with favorable mix / pricing lifted revenue and gross profit."
        read = "Cleanest setup possible — sustainability is the question; watch for any pricing rollover."
    elif vol_dir == "down" and not (price_pos or mix_pos):
        headline = "Lower volumes weighed on revenue and gross profit."
        direction = "decreased"
        fi = "Volume softness flowed through to revenue and gross profit; pricing did not offset."
        read = "Volume weakness is the more structural worry — confirm whether it reflects end-market or share loss."
    elif vol_dir == "down" and (price_pos or mix_pos):
        headline = "Volumes declined, partially offset by stronger pricing / mix."
        direction = "mixed"
        fi = "Pricing and mix mitigated the volume decline at the gross profit line."
        read = "Pricing discipline is intact, but the volume slope still matters more for the next print."
    elif vol_dir is None and (price_neg or mix_neg):
        headline = "Pricing and / or mix were unfavorable this period."
        direction = "decreased"
        fi = "Margin compression visible on the gross profit / operating income lines."
        read = "Watch for evidence of contractual pricing reset and product-mix normalisation."
    elif vol_dir is None and (price_pos or mix_pos):
        headline = "Pricing and / or mix were favorable this period."
        direction = "increased"
        fi = "Margin expansion from price / mix even without explicit volume disclosure."
        read = "Confirm next quarter whether favorable mix is structural or transient."
    else:
        return _undetermined("Pricing / Volume / Mix", sigs)

    return DriverNote(
        category="Pricing / Volume / Mix",
        headline=headline,
        driver=driver or "Volume / price / mix dynamics described in MD&A.",
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_foreign_exchange(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    pct = _first_pct(text_all)
    # FX direction inference -- look across all sentences.
    # NOTE: currency-direction words ("weakened" / "strengthened") describe
    # the FX rate move, not the P&L impact direction — a weaker USD is a
    # tailwind for a US reporter, the opposite for a euro reporter. We
    # only rely on explicit impact-direction language here.
    is_headwind = any(
        _UNFAVORABLE_RE.search(s) or _HEADWIND_RE.search(s)
        or "negative impact" in s.lower() or "reduced" in s.lower()
        for s in sigs.sentences
    )
    is_tailwind = any(
        _FAVORABLE_RE.search(s) or _TAILWIND_RE.search(s)
        or "positive impact" in s.lower() or "benefit" in s.lower()
        for s in sigs.sentences
    )
    confidence = _conf_from_signals(sigs, bool(dollar or pct))
    mag_clause = ""
    if dollar:
        mag_clause = f" of approximately {dollar}"
    elif pct:
        mag_clause = f" of approximately {pct}"

    if is_tailwind and not is_headwind:
        headline = f"Foreign currency translation was a tailwind{mag_clause}."
        direction = "improved"
        fi = "Modest positive contribution to reported revenue and / or operating income."
        read = "Currency exposure is constructive this period — not a structural driver, monitor next quarter."
    elif is_headwind and not is_tailwind:
        headline = f"Foreign currency translation was a headwind{mag_clause}."
        direction = "worsened"
        fi = "Negative translation impact on reported revenue and / or operating income."
        read = "Watch dollar trajectory; if FX persists, full-year reported growth will be muted versus organic."
    elif is_tailwind and is_headwind:
        headline = "Foreign currency translation effects were mixed across regions."
        direction = "mixed"
        fi = "Regional tailwinds and headwinds partially offset at the consolidated level."
        read = "Net FX impact is muted; focus on organic / constant-currency growth."
    else:
        return _undetermined("Foreign Exchange", sigs)

    if dollar:
        driver = f"Currency translation moved the reported number by approximately {dollar}."
    elif sigs.causes:
        cause = sigs.causes[0]
        driver = f"{cause[0].upper()}{cause[1:]}."
    else:
        driver = "Currency translation is called out as a factor in MD&A."

    return DriverNote(
        category="Foreign Exchange",
        headline=headline,
        driver=driver,
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_cost_inflation(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    low = text_all.lower()
    dollar = _first_dollar(text_all)
    pct = _first_pct(text_all)
    bps = _RE_BPS.search(text_all)
    mag_clause = ""
    if bps:
        mag_clause = f" (~{bps.group(0).strip()})"
    elif pct:
        mag_clause = f" (~{pct})"
    elif dollar:
        mag_clause = f" (~{dollar})"

    is_negative = sigs.neg_hits >= sigs.pos_hits
    has_mag = bool(dollar or pct or bps)
    confidence = _conf_from_signals(sigs, has_mag)

    if not is_negative and sigs.pos_hits == 0:
        return _undetermined("Cost Inflation / Input Costs", sigs)

    headline = f"Higher input costs compressed margins{mag_clause}."
    direction = "compressed"
    driver_bits = []
    if "feedstock" in low or "raw material" in low or "oil" in low:
        driver_bits.append("higher feedstock / raw material costs")
    if "freight" in low or "logistics" in low:
        driver_bits.append("higher freight / logistics")
    if "energy" in low:
        driver_bits.append("higher energy")
    if "labor" in low:
        driver_bits.append("higher labor")
    if not driver_bits:
        driver_bits.append("higher input costs")
    driver = ", ".join(driver_bits).capitalize()
    if sigs.causes:
        driver = f"{driver} — {sigs.causes[0]}"

    fi = "Gross margin compressed; pass-through pricing did not fully recover the input-cost run-up."
    read = (
        "Pass-through pricing ability is the structural margin lever. If the "
        "input-cost spike unwinds, margins should recover; if it sticks, "
        "watch contractual reset timing."
    )

    return DriverNote(
        category="Cost Inflation / Input Costs",
        headline=headline,
        driver=driver,
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_working_capital(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    low = text_all.lower()
    dollar = _first_dollar(text_all)

    has_recv = bool(_SUBJ_RECV.search(low))
    has_inv = bool(_SUBJ_INV.search(low))
    has_pay = bool(_SUBJ_PAY.search(low))
    has_feed = bool(_SUBJ_FEED.search(low))

    # Explicit "use of cash" / "source of cash" wording locks the direction.
    # Otherwise fall back to inference from sentence-level signals.
    explicit_use = any(
        re.search(
            r"\b(?:use\s+of\s+cash|cash\s+use[ds]?|consumed\s+cash|"
            r"investment\s+in\s+working\s+capital|build(?:up|-up)?\s+in\s+(?:inventory|receivable))\b",
            s, re.IGNORECASE,
        ) for s in sigs.sentences
    )
    explicit_source = any(
        re.search(
            r"\b(?:source\s+of\s+cash|release(?:d)?\s+(?:working\s+capital|cash)|"
            r"working\s+capital\s+benefit)\b",
            s, re.IGNORECASE,
        ) for s in sigs.sentences
    )
    if explicit_use:
        use_of_cash, source_of_cash = True, False
    elif explicit_source:
        use_of_cash, source_of_cash = False, True
    else:
        # Inferred — "higher receivable" / "higher inventory" implies a use.
        higher_uses = any(
            re.search(r"\bhigher\s+(?:account[s]?\s+)?(?:receivable|inventor)", s, re.IGNORECASE)
            for s in sigs.sentences
        )
        lower_uses = any(
            re.search(r"\blower\s+(?:account[s]?\s+)?(?:receivable|inventor)", s, re.IGNORECASE)
            for s in sigs.sentences
        )
        use_of_cash = higher_uses or (sigs.neg_hits > sigs.pos_hits and (has_recv or has_inv))
        source_of_cash = lower_uses or (
            not has_recv and not has_inv and sigs.pos_hits > sigs.neg_hits
        )

    confidence = _conf_from_signals(sigs, bool(dollar))

    if use_of_cash and not source_of_cash:
        mag = f" of approximately {dollar}" if dollar else ""
        headline = f"Working capital was a cash use this period{mag}."
        direction = "worsened"
        bits = []
        if has_recv:
            bits.append("higher accounts receivable")
        if has_inv:
            bits.append("inventory build")
        if has_pay:
            bits.append("timing of payables")
        if has_feed:
            bits.append("feedstock / oil-price volatility")
        if not bits:
            bits.append("timing of receipts and payments")
        driver = "Primarily from " + ", ".join(bits) + "."
        fi = "Contributed to lower operating cash flow versus the prior comparable period."
        read = (
            "Watch the receivables build — could indicate slower collections "
            "or pull-forward shipments. Sustained inventory build is also a "
            "leading indicator of demand softening."
        )
    elif source_of_cash and not use_of_cash:
        mag = f" of approximately {dollar}" if dollar else ""
        headline = f"Working capital released cash this period{mag}."
        direction = "improved"
        driver = "Receivables collection / inventory drawdown / payables timing were a net positive."
        fi = "Boosted operating cash flow versus the prior comparable period."
        read = "Sustainability check: confirm next quarter whether release reverses."
    else:
        return _undetermined("Working Capital", sigs)

    return DriverNote(
        category="Working Capital",
        headline=headline,
        driver=driver,
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_segment_geo(sigs: _CategorySignals) -> Optional[DriverNote]:
    """Segment / Geographic Mix synthesis, with OEC Specialty / Rubber split."""
    text_all = " | ".join(sigs.sentences)
    low = text_all.lower()

    specialty_mentioned = bool(_SUBJ_SPECIALTY.search(low))
    rubber_mentioned = bool(_SUBJ_RUBBER.search(low))

    # When both segments appear in one sentence (typical OEC pattern --
    # "Specialty grew, while Rubber declined"), split at the conjunction
    # so each segment's direction is read from its own clause.
    specialty_dir = "neutral"
    rubber_dir = "neutral"
    for s in sigs.sentences:
        clauses = re.split(r"\bwhile\b|\bbut\b|\bhowever\b|;\s+", s, flags=re.IGNORECASE)
        for cl in clauses:
            cl_low = cl.lower()
            if _SUBJ_SPECIALTY.search(cl_low):
                d = _direction(cl)
                if d in ("positive", "negative") and specialty_dir == "neutral":
                    specialty_dir = d
            if _SUBJ_RUBBER.search(cl_low):
                d = _direction(cl)
                if d in ("positive", "negative") and rubber_dir == "neutral":
                    rubber_dir = d

    confidence = _conf_from_signals(sigs, False)

    if specialty_mentioned and rubber_mentioned and (
        specialty_dir != "neutral" or rubber_dir != "neutral"
    ):
        # OEC-style split.
        if specialty_dir == "positive" and rubber_dir == "negative":
            headline = "Specialty segment outperformed; Rubber segment was the drag."
            direction = "mixed"
            driver = (
                "Specialty Carbon Black demand and / or pricing supported the "
                "segment, while Rubber Carbon Black volume / pricing weakened."
            )
        elif specialty_dir == "negative" and rubber_dir == "positive":
            headline = "Rubber segment outperformed; Specialty was softer this period."
            direction = "mixed"
            driver = "Rubber Carbon Black demand held up, while Specialty Carbon Black softened."
        elif specialty_dir == "positive" and rubber_dir == "positive":
            headline = "Both Specialty and Rubber segments contributed positively."
            direction = "improved"
            driver = "Specialty and Rubber Carbon Black both saw constructive volume / pricing dynamics."
        elif specialty_dir == "negative" and rubber_dir == "negative":
            headline = "Both Specialty and Rubber segments weighed on the period."
            direction = "worsened"
            driver = "Specialty and Rubber Carbon Black both showed volume / pricing weakness."
        else:
            return _undetermined("Segment / Geographic Mix", sigs)
        fi = "Segment mix is the primary explanation for the consolidated print."
        read = (
            "Mix shift toward higher-margin Specialty is structurally positive; "
            "Rubber recovery hinges on auto / tire OEM demand and inventory cycle."
        )
        return DriverNote(
            category="Segment / Geographic Mix",
            headline=headline,
            driver=driver,
            financial_impact=fi,
            investment_read=read,
            direction=direction,
            confidence=confidence,
            excerpts=sigs.sentences[:3],
        )

    # Generic regional / segment commentary.
    overall = "positive" if sigs.pos_hits > sigs.neg_hits else (
        "negative" if sigs.neg_hits > sigs.pos_hits else "neutral"
    )
    if overall == "positive":
        headline = "Segment / regional mix supported the period."
        direction = "improved"
        fi = "Mix contribution was a tailwind to consolidated revenue and / or margin."
    elif overall == "negative":
        headline = "Segment / regional mix weighed on the period."
        direction = "worsened"
        fi = "Mix contribution was a headwind to consolidated revenue and / or margin."
    else:
        return _undetermined("Segment / Geographic Mix", sigs)
    driver = sigs.causes[0] if sigs.causes else "Regional / segment-level performance differed across the portfolio."
    read = "Look at segment-level disclosure on the next call to confirm whether the mix shift persists."
    return DriverNote(
        category="Segment / Geographic Mix",
        headline=headline,
        driver=driver,
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_restructuring(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    confidence = _conf_from_signals(sigs, bool(dollar))
    mag = f" of approximately {dollar}" if dollar else ""
    headline = f"Restructuring / one-time charges{mag} affected reported results."
    return DriverNote(
        category="Restructuring / One-Time",
        headline=headline,
        driver=sigs.causes[0] if sigs.causes else "Severance / facility / impairment charges disclosed in the period.",
        financial_impact="Reported operating income includes a one-time charge; adjusted operating income excludes it.",
        investment_read="Distinguish underlying run-rate from one-time noise when modelling forward periods.",
        direction="worsened",
        confidence=confidence,
        excerpts=sigs.sentences[:3],
    )


def _synth_tariffs(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    pct = _first_pct(text_all)
    mag = f" (~{dollar})" if dollar else (f" (~{pct})" if pct else "")
    if sigs.neg_hits >= sigs.pos_hits and (sigs.neg_hits + sigs.pos_hits) >= 1:
        headline = f"Tariffs / trade actions weighed on the period{mag}."
        direction = "worsened"
    elif sigs.pos_hits > sigs.neg_hits:
        return _undetermined("Tariffs / Trade", sigs)
    else:
        return _undetermined("Tariffs / Trade", sigs)
    fi = "Gross margin impact unless pass-through pricing is achievable."
    read = "Track quarterly tariff exposure quantification and pricing actions in response."
    return DriverNote(
        category="Tariffs / Trade",
        headline=headline,
        driver=sigs.causes[0] if sigs.causes else "Specific tariff / trade-action language called out in MD&A.",
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=_conf_from_signals(sigs, bool(dollar or pct)),
        excerpts=sigs.sentences[:3],
    )


def _synth_capex(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    pct = _first_pct(text_all)
    mag = f" of {dollar}" if dollar else (f" ({pct})" if pct else "")
    up = sigs.pos_hits > sigs.neg_hits
    down = sigs.neg_hits > sigs.pos_hits
    if up:
        headline = f"Capex stepped up this period{mag}."
        direction = "increased"
        fi = "Higher investment temporarily reduces free cash flow."
    elif down:
        headline = f"Capex was lower this period{mag}."
        direction = "decreased"
        fi = "Lower investment supports near-term free cash flow conversion."
    else:
        return _undetermined("Capex / Investment", sigs)
    read = "Reconcile capex intensity vs the long-term growth / maintenance split disclosed by management."
    return DriverNote(
        category="Capex / Investment",
        headline=headline,
        driver=sigs.causes[0] if sigs.causes else "Capacity / property / equipment investment called out in the period.",
        financial_impact=fi,
        investment_read=read,
        direction=direction,
        confidence=_conf_from_signals(sigs, bool(dollar or pct)),
        excerpts=sigs.sentences[:3],
    )


def _synth_acquisitions(sigs: _CategorySignals) -> Optional[DriverNote]:
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    mag = f" (~{dollar})" if dollar else ""
    return DriverNote(
        category="Acquisitions / Divestitures",
        headline=f"Portfolio actions shaped the period{mag}.",
        driver=sigs.causes[0] if sigs.causes else "Acquisition / divestiture disclosed in MD&A.",
        financial_impact="Reported numbers reflect M&A contribution; isolate organic growth versus inorganic.",
        investment_read="Watch organic-only growth and integration commentary on the next call.",
        direction="mixed",
        confidence=_conf_from_signals(sigs, bool(dollar)),
        excerpts=sigs.sentences[:3],
    )


def _synth_generic(category: str, sigs: _CategorySignals) -> Optional[DriverNote]:
    if sigs.pos_hits == 0 and sigs.neg_hits == 0:
        return _undetermined(category, sigs)
    text_all = " | ".join(sigs.sentences)
    dollar = _first_dollar(text_all)
    pct = _first_pct(text_all)
    mag = f" (~{dollar})" if dollar else (f" (~{pct})" if pct else "")
    up = sigs.pos_hits > sigs.neg_hits
    headline = (
        f"{category} contributed positively to the period{mag}." if up
        else f"{category} weighed on the period{mag}."
    )
    direction = "improved" if up else "worsened"
    return DriverNote(
        category=category,
        headline=headline,
        driver=sigs.causes[0] if sigs.causes else f"{category} called out in MD&A.",
        financial_impact="Specific financial-statement impact depends on the underlying driver — see source evidence.",
        investment_read="Confirm structural vs transient and re-rate accordingly.",
        direction=direction,
        confidence=_conf_from_signals(sigs, bool(dollar or pct)),
        excerpts=sigs.sentences[:3],
    )


# Direction regexes used by the volume_change fallback.
_DIRECTION_UP_RE = re.compile(
    r"\b(?:increased|grew|rose|expanded|higher|up)\b", re.IGNORECASE
)
_DIRECTION_DOWN_RE = re.compile(
    r"\b(?:decreased|declined|fell|lower|down|contracted|weaker)\b",
    re.IGNORECASE,
)


_SYNTHESIZERS: dict[str, callable] = {
    "Pricing / Volume / Mix": _synth_pricing_volume_mix,
    "Foreign Exchange": _synth_foreign_exchange,
    "Tariffs / Trade": _synth_tariffs,
    "Cost Inflation / Input Costs": _synth_cost_inflation,
    "Restructuring / One-Time": _synth_restructuring,
    "Working Capital": _synth_working_capital,
    "Capex / Investment": _synth_capex,
    "Acquisitions / Divestitures": _synth_acquisitions,
    "Segment / Geographic Mix": _synth_segment_geo,
}


def extract_drivers(paragraphs, *, max_per_category: int = 1) -> list[DriverNote]:
    """Produce one structured driver card per detected category.

    Cards are merged at synthesis time -- there is at most one card per
    category, not one per sentence. Each card carries:

      - ``headline``: one sentence with direction and magnitude
      - ``driver`` / ``financial_impact`` / ``investment_read``: bullet
        bodies the UI renders directly
      - ``confidence``: "high" | "medium" | "low"
      - ``excerpts``: raw source sentences for the evidence expander

    If the direction can't be determined for a category that has source
    evidence, an honest "direction not clearly detected" card is emitted
    rather than a vague generic one.
    """
    signals = _collect_signals(paragraphs)
    out: list[DriverNote] = []
    seen_cats: set[str] = set()
    for label, _ in DRIVER_CATEGORIES:
        sigs = signals.get(label)
        if not sigs or not sigs.sentences:
            continue
        synth = _SYNTHESIZERS.get(label)
        card = synth(sigs) if synth else _synth_generic(label, sigs)
        if card and label not in seen_cats:
            out.append(card)
            seen_cats.add(label)
    other = signals.get("Other")
    if other and other.sentences and "Other" not in seen_cats:
        card = _synth_generic("Other", other)
        if card:
            out.append(card)
    return out


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
