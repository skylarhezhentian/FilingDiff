"""
xbrl.py
-------
Pull structured financial metrics from SEC's XBRL companyfacts API.

This is far more reliable than parsing prose. For each us-gaap tag we:
  1. Get the full historical series of values + period info.
  2. Find the entry matching the current 10-Q / 10-K we're comparing.
  3. Find the prior year-over-year comparable entry.
  4. Compute the delta.

For 10-Q comparisons we use a *YoY* prior period (same fiscal quarter,
one year earlier) rather than the immediately prior quarter -- that's
how SEC filings present comparatives and how analysts read them.

We try multiple tag aliases per metric because filers use different
us-gaap concepts (e.g. Apple uses
`RevenueFromContractWithCustomerExcludingAssessedTax`; older filers use
`Revenues` or `SalesRevenueNet`). The first tag with data wins.

If the API or a tag is unavailable, we *omit* the metric. We never
fabricate a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from edgar import http_get


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Flow / income-statement metrics. Compared period-over-period
# (Q3 FY2024 vs Q3 FY2023, etc.).
INCOME_METRICS: list[tuple[str, list[str]]] = [
    ("Net Revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]),
    ("Cost of Revenue", ["CostOfRevenue", "CostOfGoodsAndServicesSold"]),
    ("Gross Profit", ["GrossProfit"]),
    ("R&D Expense", ["ResearchAndDevelopmentExpense"]),
    ("SG&A Expense", ["SellingGeneralAndAdministrativeExpense"]),
    ("Operating Income", ["OperatingIncomeLoss"]),
    ("Net Income", ["NetIncomeLoss"]),
    ("Diluted EPS", ["EarningsPerShareDiluted"]),
    ("Basic EPS", ["EarningsPerShareBasic"]),
]

# Cash-flow statement / capital allocation metrics.
CASHFLOW_METRICS: list[tuple[str, list[str]]] = [
    ("Operating Cash Flow", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    ("Capital Expenditures", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ]),
    ("Stock Repurchases", [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ]),
    ("Dividends Paid", [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ]),
]

# Balance-sheet metrics. Compared point-in-time (period end vs prior end).
BALANCE_METRICS: list[tuple[str, list[str]]] = [
    ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue"]),
    ("Short-Term Investments", [
        "MarketableSecuritiesCurrent",
        "ShortTermInvestments",
    ]),
    ("Long-Term Investments", [
        "MarketableSecuritiesNoncurrent",
        "LongTermInvestments",
    ]),
    ("Total Assets", ["Assets"]),
    ("Current Portion of Debt", [
        "LongTermDebtCurrent",
        "DebtCurrent",
    ]),
    ("Long-Term Debt", [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ]),
    ("Total Stockholders' Equity", ["StockholdersEquity"]),
]


@dataclass
class XBRLMetric:
    """One metric with current vs prior values from XBRL."""

    label: str
    tag: str  # us-gaap concept used
    latest_val: Optional[float]
    latest_end: Optional[str]  # period end (YYYY-MM-DD) of the latest value
    prior_val: Optional[float]
    prior_end: Optional[str]
    unit: str  # "USD", "USD/shares", "shares", etc.
    delta_pct: Optional[float]  # signed percent change vs prior


# ---------------------------------------------------------------------------
# Fetch + select
# ---------------------------------------------------------------------------


def get_company_facts(cik: str) -> Optional[dict]:
    """Fetch the companyfacts JSON for a CIK. Returns None on 404."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        return http_get(url).json()
    except Exception:
        # 404 (no XBRL data on file) or transient issue. Caller renders
        # "Not detected" rather than crashing.
        return None


def _units_for_tag(facts: dict, tag: str) -> tuple[Optional[str], list[dict]]:
    """Return (unit_key, entries) for the given us-gaap tag, or (None, [])."""
    us_gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    node = us_gaap.get(tag)
    if not node:
        return None, []
    units = node.get("units") or {}
    if not units:
        return None, []
    unit = next(iter(units))
    return unit, units[unit] or []


def _is_flow_entry(e: dict) -> bool:
    """Flow entries (income/cash-flow items) have a start + end pair."""
    return bool(e.get("start") and e.get("end"))


def _period_days(e: dict) -> Optional[int]:
    if not _is_flow_entry(e):
        return None
    try:
        return (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
    except (ValueError, KeyError):
        return None


def _filter_entries_for_form(
    entries: list[dict], form: str, kind: str
) -> list[dict]:
    """Filter entries to the form type and a sensible period length.

    `kind` is "flow" (income / cash-flow item) or "stock" (balance-sheet
    item). For 10-Q flow items we keep ~3-month windows so we don't
    confuse a year-to-date value with a quarterly one. For 10-K flow
    items we keep ~12-month windows.
    """
    form_u = form.upper()
    out: list[dict] = []
    for e in entries:
        if e.get("form", "").upper() != form_u:
            continue
        if kind == "stock":
            if e.get("end"):
                out.append(e)
            continue
        days = _period_days(e)
        if days is None:
            continue
        if form_u == "10-Q" and 80 <= days <= 100:
            out.append(e)
        elif form_u == "10-K" and 340 <= days <= 380:
            out.append(e)
    return out


def _select_pair(
    entries: list[dict], form: str
) -> tuple[Optional[dict], Optional[dict]]:
    """Return (latest, prior_YoY) entries, sorted by end date desc."""
    if not entries:
        return None, None
    entries.sort(key=lambda e: e["end"], reverse=True)
    latest = entries[0]

    fy = latest.get("fy")
    fp = latest.get("fp")
    prior = None
    for e in entries[1:]:
        # Prefer same fiscal period, one year earlier (true YoY).
        if e.get("fp") == fp and fy is not None and e.get("fy") == fy - 1:
            prior = e
            break
    # Fallback: just take the next-most-recent entry. We still mark it
    # honestly via the date label rendered by the UI.
    if prior is None and len(entries) > 1:
        prior = entries[1]
    return latest, prior


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_metrics(
    facts: Optional[dict],
    form: str,
    metric_defs: list[tuple[str, list[str]]],
    kind: str,
) -> list[XBRLMetric]:
    """Fetch a set of metrics from the companyfacts JSON.

    `metric_defs` is the (label, [candidate_tags]) list, e.g. INCOME_METRICS.
    `kind` is "flow" or "stock" -- controls how we filter period lengths.
    """
    if not facts:
        return []
    out: list[XBRLMetric] = []
    for label, tags in metric_defs:
        for tag in tags:
            unit, all_entries = _units_for_tag(facts, tag)
            if not all_entries:
                continue
            entries = _filter_entries_for_form(all_entries, form, kind)
            latest, prior = _select_pair(entries, form)
            if not latest:
                continue
            latest_val = latest.get("val")
            prior_val = prior.get("val") if prior else None
            delta_pct = None
            if (
                isinstance(latest_val, (int, float))
                and isinstance(prior_val, (int, float))
                and prior_val != 0
            ):
                delta_pct = (latest_val - prior_val) / abs(prior_val) * 100.0
            out.append(
                XBRLMetric(
                    label=label,
                    tag=tag,
                    latest_val=float(latest_val) if isinstance(latest_val, (int, float)) else None,
                    latest_end=latest.get("end"),
                    prior_val=float(prior_val) if isinstance(prior_val, (int, float)) else None,
                    prior_end=prior.get("end") if prior else None,
                    unit=unit or "",
                    delta_pct=delta_pct,
                )
            )
            break  # stop at first tag that produced data
    return out


def format_value(val: Optional[float], unit: str) -> str:
    """Human-readable display string. Returns '—' for missing values."""
    if val is None:
        return "—"
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


def format_delta(delta_pct: Optional[float]) -> Optional[str]:
    if delta_pct is None:
        return None
    return f"{delta_pct:+.1f}%"


def find_metric(metrics: list[XBRLMetric], label: str) -> Optional[XBRLMetric]:
    for m in metrics:
        if m.label == label:
            return m
    return None
