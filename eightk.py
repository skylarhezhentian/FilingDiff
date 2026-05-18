"""
eightk.py
---------
Event-intelligence extraction for 8-K filings.

8-Ks are point-in-time disclosure events, not period-over-period
comparatives. The main question a reader has is "what event did the
company disclose, and why does it matter?" This module answers that by
producing a structured ``EightKEvent`` with:

  - Event type (e.g. "Earnings Release / Results of Operations")
  - Item numbers disclosed (Item 2.02, Item 9.01, ...)
  - Exhibits referenced (Exhibit 99.1, Exhibit 10.1, ...)
  - Materiality (high / medium / low)
  - Overall read (one-line directional assessment)
  - Event summary (plain English)
  - Key financial metrics (revenue / net loss / adjusted EBITDA / cash / RPO / ...)
  - Guidance / outlook if quantified
  - Business highlights (customer wins, contract updates)
  - Investment read (so-what for an investor)
  - Raw excerpts for a source-evidence expander

The metric / highlight extractors are deliberately conservative -- they
only surface lines the regex set can reliably anchor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field  # noqa: F401 (field used at runtime)
from typing import Optional


# ---------------------------------------------------------------------------
# 8-K Item taxonomy
# ---------------------------------------------------------------------------

# Item code -> (friendly name, materiality)
ITEM_CODES: dict[str, tuple[str, str]] = {
    "1.01": ("Entry into Material Definitive Agreement", "high"),
    "1.02": ("Termination of Material Definitive Agreement", "high"),
    "1.03": ("Bankruptcy or Receivership", "high"),
    "1.04": ("Mine Safety Reporting", "low"),
    "2.01": ("Completion of Acquisition or Disposition of Assets", "high"),
    "2.02": ("Results of Operations and Financial Condition", "high"),
    "2.03": ("Creation of a Direct Financial Obligation", "high"),
    "2.04": ("Triggering Event -- Acceleration of Obligation", "high"),
    "2.05": ("Costs Associated with Exit or Disposal Activities", "medium"),
    "2.06": ("Material Impairment", "high"),
    "3.01": ("Listing Standards Notice / Delisting", "high"),
    "3.02": ("Unregistered Sales of Equity Securities", "medium"),
    "3.03": ("Material Modification to Rights of Security Holders", "medium"),
    "4.01": ("Changes in Registrant's Certifying Accountant", "high"),
    "4.02": ("Non-Reliance on Previously Issued Financial Statements", "high"),
    "5.01": ("Changes in Control of Registrant", "high"),
    "5.02": ("Departure / Election / Appointment of Officers / Directors", "medium"),
    "5.03": ("Amendments to Articles of Incorporation or Bylaws", "low"),
    "5.04": ("Temporary Suspension of Trading Under Plans", "medium"),
    "5.05": ("Amendments to the Code of Ethics", "low"),
    "5.07": ("Submission of Matters to a Vote of Security Holders", "low"),
    "5.08": ("Shareholder Director Nominations", "low"),
    "6.01": ("ABS Informational and Computational Material", "low"),
    "7.01": ("Regulation FD Disclosure", "medium"),
    "8.01": ("Other Events", "low"),
    "9.01": ("Financial Statements and Exhibits", "low"),
}

# Per-item friendly event-type label. Used to pick the headline event
# type when multiple items appear (highest materiality wins).
EVENT_TYPE_FROM_ITEM: dict[str, str] = {
    "1.01": "Material Agreement",
    "1.02": "Material Agreement Termination",
    "1.03": "Bankruptcy / Receivership",
    "2.01": "Acquisition / Disposition",
    "2.02": "Earnings Release / Results of Operations",
    "2.03": "Material Financial Obligation",
    "2.04": "Triggering Event",
    "2.05": "Exit / Disposal Activity",
    "2.06": "Material Impairment",
    "3.01": "Listing / Delisting",
    "4.01": "Auditor Change",
    "4.02": "Restatement / Non-Reliance",
    "5.01": "Change of Control",
    "5.02": "Officer / Director Change",
    "5.07": "Shareholder Vote Result",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Event",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KeyMetric:
    label: str
    value: str
    delta: Optional[str] = None  # "+12% YoY" / "(15.6)%"
    note: Optional[str] = None


@dataclass
class Highlight:
    title: str
    text: str


@dataclass
class ExhibitParseStatus:
    """Per-exhibit parsing diagnostics for the debug expander."""

    doc_type: str
    name: str
    url: str
    parsed: bool
    char_count: int
    note: Optional[str] = None


@dataclass
class EightKEvent:
    event_type: str
    item_numbers: list[str]
    item_descriptions: list[str]
    exhibits: list[str]
    materiality: str
    overall_read: str
    event_summary: str
    key_metrics: list[KeyMetric]
    guidance_lines: list[str]
    highlights: list[Highlight]
    investment_read: str
    raw_excerpts: list[str]
    # Parsing diagnostics. Always populated; rendered inside a collapsed
    # debug expander so users can see exactly what was downloaded.
    primary_parsed: bool = True
    primary_char_count: int = 0
    exhibits_status: list = field(default_factory=list)  # list[ExhibitParseStatus]
    exhibit_99_1_detected: bool = False


# ---------------------------------------------------------------------------
# Item / Exhibit detection
# ---------------------------------------------------------------------------


_ITEM_RE = re.compile(r"\bItem\s+(\d+\.\d+)\b", re.IGNORECASE)
_EXHIBIT_RE = re.compile(r"\bExhibit\s+(\d+\.\d+[A-Za-z]?)\b", re.IGNORECASE)
_MAT_RANK = {"low": 0, "medium": 1, "high": 2}
_MAT_NAME = {0: "low", 1: "medium", 2: "high"}


def _items_in(text: str) -> list[str]:
    raw = _ITEM_RE.findall(text)
    # Keep only items defined in our taxonomy.
    return sorted({i for i in raw if i in ITEM_CODES})


def _exhibits_in(text: str) -> list[str]:
    return sorted(set(_EXHIBIT_RE.findall(text)))


def _pick_materiality(items: list[str]) -> str:
    if not items:
        return "low"
    ranks = [_MAT_RANK[ITEM_CODES[i][1]] for i in items if i in ITEM_CODES]
    if not ranks:
        return "low"
    return _MAT_NAME[max(ranks)]


def _pick_event_type(items: list[str]) -> str:
    if not items:
        return "Other Material Event (item not detected)"
    ordered = sorted(
        items,
        key=lambda i: -_MAT_RANK.get(ITEM_CODES.get(i, ("", "low"))[1], 0),
    )
    for i in ordered:
        if i in EVENT_TYPE_FROM_ITEM:
            return EVENT_TYPE_FROM_ITEM[i]
    name, _ = ITEM_CODES.get(ordered[0], ("Other Material Event", "low"))
    return name


# ---------------------------------------------------------------------------
# Metric extraction (Item 2.02 earnings filings)
# ---------------------------------------------------------------------------

# Reusable amount fragment: matches plain "$30.0 million", "(2.4) million",
# "$30,123", "$1.2 billion", etc. We let the surrounding context anchor
# whether the figure is positive or parenthesised-negative.
_AMOUNT_FRAG = (
    r"\$?\s*\(?\s*\$?\s*[\d][\d,]*(?:\.\d+)?\s*\)?\s*"
    r"(?:million|billion|thousand|M\b|B\b)?"
)

# Optional YoY delta fragment in the metric sentence.
_DELTA_FRAG = (
    r"(?:(?P<delta>(?:up|down|increase[ds]?|decrease[ds]?|\+|-)?\s*\d+(?:\.\d+)?\s*%"
    r"(?:\s+(?:year[\s-]over[\s-]year|yoy))?))"
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(",.;")


def _search_metric(
    text: str, patterns: list[str]
) -> Optional[tuple[str, Optional[str]]]:
    """Try each regex; return (value, delta_or_None) for the first match.

    Patterns must capture a ``val`` group and may capture a ``delta``.
    """
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = _clean(m.group("val"))
            delta = None
            if "delta" in m.groupdict() and m.group("delta"):
                delta = _clean(m.group("delta"))
            return val, delta
    return None


# Fragment that consumes an optional footnote-style superscript reference
# right after a metric label, e.g. "adjusted EBITDA (1) of ($10.2) million".
# Without this the regex captures "(1)" as the value.
_FOOTNOTE_FRAG = r"(?:\s*\(\d{1,2}\))?"

# Strict value fragment. Accepts (in priority order):
#   "$X.X million" / "$X,XXX million" / "$X billion"  -- dollar-prefixed
#   "($X.X) million" / "($X) million"                 -- paren-negative w/ $ inside
#   "(X.X) million" / "(X) million"                   -- paren-negative bare
#   "X.X million" / "X million"                       -- bare with required unit
# The "million"/"billion" unit is mandatory whenever there is no "$"
# prefix outside the parentheses, so a stray "(1)" footnote ref never
# looks like a value.
_VAL_FRAG = (
    r"(?P<val>"
    r"\$\s*\(?\s*[\d][\d,]*(?:\.\d+)?\)?(?:\s*(?:million|billion))?"
    r"|"
    r"\(\s*\$\s*[\d][\d,]*(?:\.\d+)?\s*\)\s*(?:million|billion)"
    r"|"
    r"\(\s*[\d][\d,]*(?:\.\d+)?\s*\)\s*(?:million|billion)"
    r"|"
    r"[\d][\d,]*(?:\.\d+)?\s*(?:million|billion)"
    r")"
)
# Lead-in fragment: optional "of / was / totaled / were / came in at /
# stood at" -- the connector between the label and the value.
_LEAD = r"(?:of\s+|was\s+|were\s+|totaled\s+|came\s+in\s+at\s+|stood\s+at\s+)?"


def extract_earnings_metrics(text: str) -> list[KeyMetric]:
    """Extract press-release-style headline metrics from 8-K exhibit text.

    Patterns are anchored on metric *labels* and skip an optional
    footnote reference ``(1)`` that issuers like Spire interpose between
    the label and the value. The value fragment requires either a ``$``
    prefix or a ``million``/``billion`` unit so a stray ``(1)`` is
    never mistaken for a value.
    """
    out: list[KeyMetric] = []
    used_labels: set[str] = set()

    def add(label: str, patterns: list[str]) -> None:
        if label in used_labels:
            return
        hit = _search_metric(text, patterns)
        if hit is None:
            return
        val, delta = hit
        out.append(KeyMetric(label=label, value=val, delta=delta))
        used_labels.add(label)

    # Revenue
    add("Revenue", [
        r"(?:total\s+)?(?:gaap\s+)?revenue[s]?" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+|were\s+|totaled\s+|came\s+in\s+at\s+)?" +
        _VAL_FRAG +
        r"(?:[^.]{0,160}?" + _DELTA_FRAG + r")?",
    ])

    # Operating loss / income (GAAP + non-GAAP variants)
    add("Operating loss", [
        r"(?:gaap\s+|non-?gaap\s+)?operating\s+loss" + _FOOTNOTE_FRAG +
        r"\s+" + _LEAD + _VAL_FRAG,
        r"loss\s+from\s+operations" + _FOOTNOTE_FRAG +
        r"\s+" + _LEAD + _VAL_FRAG,
    ])
    add("Operating income", [
        r"(?:gaap\s+|non-?gaap\s+)?operating\s+income" + _FOOTNOTE_FRAG +
        r"\s+" + _LEAD + _VAL_FRAG,
    ])

    # Net loss / income
    add("Net loss", [
        r"net\s+loss" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+|totaled\s+)?" + _VAL_FRAG,
        r"net\s+loss\s+attributable\s+to[^.]{0,80}?(?:of\s+|was\s+)?" + _VAL_FRAG,
    ])
    add("Net income", [
        r"net\s+income" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+|totaled\s+)?" + _VAL_FRAG,
    ])

    # Adjusted EBITDA -- the prime motivator for the footnote-skip fix.
    add("Adjusted EBITDA", [
        r"adjusted\s+ebitda" + _FOOTNOTE_FRAG +
        r"\s+(?:loss\s+)?" + _LEAD + _VAL_FRAG,
    ])

    # Gross margin (percentage)
    add("Gross margin", [
        r"(?:gaap\s+)?gross\s+margin" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+|came\s+in\s+at\s+|improved\s+(?:to\s+)?)?"
        r"(?P<val>\d{1,3}(?:\.\d+)?\s*%)",
    ])

    # EPS (Diluted preferred over basic when both present)
    add("Diluted EPS", [
        r"diluted\s+(?:net\s+)?(?:loss|earnings|income)\s+per\s+share" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+)?(?P<val>\$?\s*\(?\s*[\d.]+\)?)",
        r"(?:earnings|loss)\s+per\s+diluted\s+share" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+)?(?P<val>\$?\s*\(?\s*[\d.]+\)?)",
    ])

    # Cash, cash equivalents, and marketable securities -- the exact
    # phrasing Spire uses. The optional "as of <date>" clause may
    # contain a comma ("as of March 31, 2026"), so allow it via [^.]{0,60}.
    add("Cash, cash equivalents & marketable securities", [
        r"cash[,]?\s+cash\s+equivalents[,]?\s+(?:and\s+)?(?:short[-\s]term\s+)?"
        r"(?:marketable\s+securities|investments)"
        r"(?:\s+as\s+of\s+[^.]{0,50}?)?"
        r"\s+(?:of\s+|totaled\s+|were\s+|was\s+)"
        r"(?P<val>\$\s*[\d][\d,]*(?:\.\d+)?\s*(?:million|billion))",
        r"cash\s+and\s+(?:short[-\s]term\s+)?marketable\s+securities"
        r"\s+(?:of\s+|totaled\s+|were\s+)?"
        r"(?P<val>\$\s*[\d][\d,]*(?:\.\d+)?\s*(?:million|billion))",
    ])

    # RPO / Remaining Performance Obligations. The Q3-style language
    # ("RPO of more than $200 million") uses ``more than`` / ``over``;
    # capture those too.
    add("Remaining Performance Obligations (RPO)", [
        r"(?:remaining\s+performance\s+obligation[s]?)\s*"
        r"(?:\([^)]+\)\s*)?(?:of\s+|was\s+|stood\s+at\s+|totaled\s+|were\s+|"
        r"more\s+than\s+|over\s+|in\s+excess\s+of\s+)?"
        r"(?P<val>\$\s*[\d][\d,]*(?:\.\d+)?\s*(?:million|billion))",
        r"\brpo\b" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+|stood\s+at\s+|totaled\s+|were\s+|"
        r"more\s+than\s+|over\s+|in\s+excess\s+of\s+)?"
        r"(?P<val>\$\s*[\d][\d,]*(?:\.\d+)?\s*(?:million|billion))",
        r"backlog\s+of\s+(?P<val>\$\s*[\d][\d,]*(?:\.\d+)?\s*(?:million|billion))",
    ])

    # Operating cash flow / burn (both directions of language)
    add("Operating cash flow", [
        r"(?:net\s+)?cash\s+used\s+in\s+operat(?:ing|ions)\s+activities" + _FOOTNOTE_FRAG +
        r"\s+(?:was\s+|of\s+|totaled\s+)?" + _VAL_FRAG,
        r"(?:net\s+)?cash\s+provided\s+by\s+operat(?:ing|ions)\s+activities" + _FOOTNOTE_FRAG +
        r"\s+(?:was\s+|of\s+|totaled\s+)?" + _VAL_FRAG,
        r"operating\s+cash\s+(?:flow|burn|usage)" + _FOOTNOTE_FRAG +
        r"\s+(?:of\s+|was\s+)?" + _VAL_FRAG,
    ])

    # Free cash flow
    add("Free cash flow", [
        r"free\s+cash\s+flow" + _FOOTNOTE_FRAG +
        r"\s+" + _LEAD + _VAL_FRAG,
    ])

    return out


# ---------------------------------------------------------------------------
# Highlights -- customer / contract / business updates
# ---------------------------------------------------------------------------

# Named entities commonly seen in press-release business updates. Add to
# this set as you encounter new issuers; the matcher is case-insensitive
# and word-boundary anchored.
_HIGHLIGHT_ENTITIES = re.compile(
    r"\b("
    r"NOAA|EUMETSAT|NASA|"
    r"National\s+Oceanic|National\s+Aeronautics|"
    r"Deloitte|PwC|EY|KPMG|"
    r"Department\s+of\s+Defense|DoD|Pentagon|Air\s+Force|Space\s+Force|"
    r"U\.?S\.?\s+(?:Army|Navy|Marine|Coast\s+Guard|Government)|"
    r"Defense\s+Innovation\s+Unit|DIU|"
    r"European\s+Space\s+Agency|ESA|European\s+Union|EU\s+Commission|"
    r"Microsoft|Google|Amazon|AWS|Meta|Apple|Oracle|Salesforce|"
    r"SoftBank|Lockheed\s+Martin|Lockheed|Boeing|Raytheon|Northrop|"
    r"General\s+Dynamics|L3Harris|"
    r"Saudi\s+Aramco|TotalEnergies|Shell|Equinor|BP|Chevron|ExxonMobil|"
    r"Walmart|Target|Costco|Kroger|Home\s+Depot|"
    r"Spire|Maxar|Planet|BlackSky|"
    r"the\s+U\.?S\.?\s+Air\s+Force|the\s+U\.?S\.?\s+Navy|the\s+Pentagon"
    r")\b",
    re.IGNORECASE,
)

_HIGHLIGHT_VERBS = re.compile(
    r"\b("
    r"awarded|won|signed|secured|renewed|extended|launched|"
    r"completed|achieved|expanded|delivered|received|"
    r"announced|established|deployed|added|landed|selected|"
    r"entered\s+into|received\s+a\s+new|received\s+an?\s+award"
    r")\b",
    re.IGNORECASE,
)


def extract_highlights(paragraphs, issuer_name: Optional[str] = None) -> list[Highlight]:
    """Find customer-update / contract-update business highlights.

    A highlight is titled by the *partner / customer* entity, not the
    issuer. We prefer the entity that appears *after* the action verb
    in the sentence (the object of the action, typically the
    counterparty). If no entity follows the verb, we fall back to the
    first non-issuer entity in the sentence, and only as a last resort
    use the issuer itself.
    """
    issuer_low = issuer_name.lower().strip() if issuer_name else None

    def _is_issuer(ent: str) -> bool:
        if not issuer_low:
            return False
        el = ent.lower().strip()
        return el == issuer_low or el in issuer_low or issuer_low in el

    highlights: list[Highlight] = []
    seen: set[str] = set()
    for p in paragraphs:
        text = p.text if hasattr(p, "text") else str(p)
        # Press releases routinely run 1500-2500 chars per paragraph,
        # especially the "Business Highlights" section that packs
        # multiple wins into one block.
        if len(text) < 50 or len(text) > 2500:
            continue
        verb_m = _HIGHLIGHT_VERBS.search(text)
        if not verb_m:
            continue
        # All entities in the sentence with their positions.
        entities = [(m.start(), m.group(0)) for m in _HIGHLIGHT_ENTITIES.finditer(text)]
        if not entities:
            continue
        # Prefer entities that appear after the action verb (object of
        # action — typically the counterparty). Filter out issuer
        # self-references.
        verb_pos = verb_m.start()
        post_verb = [(pos, ent) for pos, ent in entities if pos > verb_pos and not _is_issuer(ent)]
        if post_verb:
            entity = post_verb[0][1]
        else:
            non_issuer = [ent for _pos, ent in entities if not _is_issuer(ent)]
            if non_issuer:
                entity = non_issuer[0]
            else:
                entity = entities[0][1]

        key = re.sub(r"\W+", "", text.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        verb = verb_m.group(0).lower()
        title = f"{entity}: {verb}"
        snippet = text if len(text) <= 400 else text[:397].rstrip() + "…"
        highlights.append(Highlight(title=title, text=snippet))
        if len(highlights) >= 8:
            break
    return highlights


# ---------------------------------------------------------------------------
# Summary, overall read, investment read
# ---------------------------------------------------------------------------


def _has_label(metrics: list[KeyMetric], needle: str) -> Optional[KeyMetric]:
    return next((m for m in metrics if needle.lower() in m.label.lower()), None)


def _build_event_summary(
    event_type: str,
    items: list[str],
    metrics: list[KeyMetric],
    highlights: list[Highlight],
    exhibits: list[str],
) -> str:
    parts: list[str] = []
    items_str = ", ".join(f"Item {i}" for i in items) if items else "no item number detected"
    ex_str = ", ".join(f"Exhibit {e}" for e in exhibits) if exhibits else None
    open_line = f"This 8-K is filed as {event_type}"
    if items:
        open_line += f" ({items_str})"
    if ex_str:
        open_line += f" with {ex_str} attached"
    parts.append(open_line + ".")

    rev = _has_label(metrics, "Revenue")
    loss = _has_label(metrics, "Net loss") or _has_label(metrics, "Operating loss")
    ebitda = _has_label(metrics, "Adjusted EBITDA")
    cash = _has_label(metrics, "Cash")
    rpo = _has_label(metrics, "RPO") or _has_label(metrics, "Remaining")

    fin_bits: list[str] = []
    if rev:
        bit = f"revenue of {rev.value}"
        if rev.delta:
            bit += f" ({rev.delta})"
        fin_bits.append(bit)
    if loss:
        fin_bits.append(f"{loss.label.lower()} of {loss.value}")
    if ebitda:
        fin_bits.append(f"adjusted EBITDA of {ebitda.value}")
    if cash:
        fin_bits.append(f"cash position of {cash.value}")
    if rpo:
        fin_bits.append(f"RPO of {rpo.value}")
    if fin_bits:
        parts.append("Key disclosures: " + "; ".join(fin_bits) + ".")

    if highlights:
        ents = sorted({h.title.split(":")[0] for h in highlights})[:4]
        parts.append(
            f"Business / customer references include "
            f"{', '.join(ents)}{' and others' if len(highlights) > 4 else ''}."
        )
    return " ".join(parts)


def _build_overall_read(
    event_type: str,
    materiality: str,
    metrics: list[KeyMetric],
    guidance: list[str],
    highlights: list[Highlight],
) -> str:
    is_earnings = "earnings" in event_type.lower() or "results" in event_type.lower()
    rev = _has_label(metrics, "Revenue")
    has_loss = any("loss" in m.label.lower() for m in metrics)
    has_income = any("income" in m.label.lower() and "loss" not in m.label.lower() for m in metrics)

    rev_positive = bool(
        rev and rev.delta and re.search(r"\b(?:up|increase|\+)\b", rev.delta, re.IGNORECASE)
    )
    rev_negative = bool(
        rev and rev.delta and re.search(r"\b(?:down|decrease|-)\b", rev.delta, re.IGNORECASE)
    )

    if is_earnings:
        if rev_positive and has_loss:
            return (
                "Mixed — revenue growth alongside continued operating losses. "
                "Cash runway, EBITDA trajectory, and RPO conversion are the key gates."
            )
        if rev_positive and not has_loss:
            return "Net constructive — revenue growth with profitable / break-even operating result."
        if rev_negative:
            return "Cautionary — revenue contraction reported; assess whether one-time or trend."
        if has_loss and not rev:
            return "Cautionary — material operating loss reported."
        if has_income and not has_loss:
            return "Net constructive — profitable operating result reported."
        return "Earnings filing — read revenue trajectory, EBITDA, and cash runway."

    if materiality == "high":
        return f"Material event ({event_type}) — read the source for specifics."
    if materiality == "medium":
        return f"Notable event ({event_type}) — likely worth contextualising."
    return "Informational filing — typically not market-moving on its own."


def _build_investment_read(
    metrics: list[KeyMetric],
    guidance: list[str],
    highlights: list[Highlight],
) -> str:
    bits: list[str] = []
    rev = _has_label(metrics, "Revenue")
    cash = _has_label(metrics, "Cash")
    rpo = _has_label(metrics, "RPO") or _has_label(metrics, "Remaining")
    ebitda = _has_label(metrics, "Adjusted EBITDA")
    ocf = _has_label(metrics, "Operating cash flow")

    if rev:
        delta_clause = f" ({rev.delta})" if rev.delta else ""
        bits.append(f"Revenue prints at {rev.value}{delta_clause}.")
    if ebitda:
        bits.append(
            f"Adjusted EBITDA of {ebitda.value} is the cleanest read of "
            f"operating progress at scale."
        )
    if cash:
        runway = f" Cash position of {cash.value} anchors the runway debate."
        if ocf:
            runway += f" Operating cash flow of {ocf.value} sets the burn rate."
        bits.append(runway)
    if rpo:
        bits.append(f"RPO of {rpo.value} signals the contracted forward revenue base.")
    if highlights:
        ents = sorted({h.title.split(":")[0] for h in highlights})[:3]
        bits.append(
            f"Customer / contract updates referenced ({', '.join(ents)}) -- "
            f"assess durability of the wins versus the competitive set."
        )
    if guidance:
        bits.append(
            "Management provided forward guidance -- quantify versus consensus "
            "and the prior trajectory."
        )
    if not bits:
        bits.append(
            "Read the source filing for substantive event detail. The 8-K "
            "primary document may be a cover-only filing referencing an "
            "exhibit not included in this analysis."
        )
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_8k_event(
    cover_paragraphs,
    *,
    exhibit_paragraphs: Optional[list] = None,
    guidance_notes=None,
    issuer_name: Optional[str] = None,
    exhibits_status: Optional[list] = None,
    primary_parsed: bool = True,
    primary_char_count: int = 0,
) -> EightKEvent:
    """Build the structured 8-K event-intelligence report.

    For an Item 2.02 earnings 8-K the substantive content lives in
    Exhibit 99.1, not the cover document. The renderer is responsible
    for downloading the relevant exhibits with
    ``edgar.get_filing_documents`` + ``edgar.select_8k_exhibits`` and
    passing the parsed paragraphs in via ``exhibit_paragraphs``.

      - Items and exhibit references are detected from the *cover*
        document (that is where Item X.XX and "Exhibit 99.1" are
        declared).
      - Metrics and highlights are extracted from the *exhibit* text
        when available, falling back to the cover otherwise.
      - Guidance is sourced from whichever document the caller pre-ran
        through ``narrative.extract_guidance``.
      - ``exhibits_status`` carries one ``ExhibitParseStatus`` record
        per attempted exhibit fetch for the debug expander.
    """
    cover_text = "\n".join(
        (p.text if hasattr(p, "text") else str(p)) for p in cover_paragraphs
    )
    exhibit_paragraphs = exhibit_paragraphs or []
    exhibit_text = "\n".join(
        (p.text if hasattr(p, "text") else str(p)) for p in exhibit_paragraphs
    )

    # Items and exhibit references are declared on the cover sheet.
    items = _items_in(cover_text)
    exhibits = _exhibits_in(cover_text)
    event_type = _pick_event_type(items)
    materiality = _pick_materiality(items)
    item_descriptions = [
        f"Item {i} — {ITEM_CODES[i][0]}" for i in items if i in ITEM_CODES
    ]

    # Metric / highlight extraction prefers the exhibit text when we
    # have it -- that is where the press-release content lives. Fall
    # back to the cover only if no exhibit was parsed.
    metric_source = exhibit_text if exhibit_text.strip() else cover_text
    highlight_source = exhibit_paragraphs if exhibit_paragraphs else cover_paragraphs
    metrics = (
        extract_earnings_metrics(metric_source)
        if "2.02" in items or any("EX-99" in (s.doc_type if hasattr(s, "doc_type") else "")
                                  for s in (exhibits_status or []))
        else []
    )
    highlights = extract_highlights(highlight_source, issuer_name=issuer_name)
    guidance_lines = [getattr(g, "headline", "") for g in (guidance_notes or [])]
    guidance_lines = [g for g in guidance_lines if g]

    event_summary = _build_event_summary(event_type, items, metrics, highlights, exhibits)
    overall_read = _build_overall_read(event_type, materiality, metrics, guidance_lines, highlights)
    investment_read = _build_investment_read(metrics, guidance_lines, highlights)

    # Source-evidence excerpts: prefer the exhibit text (that is what
    # readers will want to verify against), fall back to cover.
    excerpt_source = exhibit_paragraphs if exhibit_paragraphs else cover_paragraphs
    excerpts: list[str] = []
    for p in excerpt_source[:80]:
        text = p.text if hasattr(p, "text") else str(p)
        if 60 <= len(text) <= 700:
            excerpts.append(text)
        if len(excerpts) >= 14:
            break

    return EightKEvent(
        event_type=event_type,
        item_numbers=items,
        item_descriptions=item_descriptions,
        exhibits=exhibits,
        materiality=materiality,
        overall_read=overall_read,
        event_summary=event_summary,
        key_metrics=metrics,
        guidance_lines=guidance_lines,
        highlights=highlights,
        investment_read=investment_read,
        raw_excerpts=excerpts,
        primary_parsed=primary_parsed,
        primary_char_count=primary_char_count,
        exhibits_status=exhibits_status or [],
        exhibit_99_1_detected="99.1" in exhibits,
    )
