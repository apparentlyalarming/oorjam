"""Module 4: Analytics Engine & ROI Report Generator.

Responsibilities
----------------
* **Financial metrics conversion.**  Each anomaly carries the *excess* kW it
  caused measured against the trained baseline consumption models.  The two
  canonical spec formulae are applied here:

    ``Energy Wasted (kWh)  = (Actual kW - Baseline Model kW) x Duration (Hours)``
    ``Financial Waste ($)  = Energy Wasted (kWh)  x Utility Rate ($/kWh)``

* **Output ranking.**  Retrofit and behavioural measures are strictly sorted by
  **Payback Period (months)** ascending, then **Estimated Annual ROI (%)**
  descending (per the specification).

* **JSON summaries.**  The engine emits the structured ``AnalyticsReport`` JSON
  contract consumed by ``/api/analytics/current``, plus a human-readable
  markdown brief via ``as_markdown()``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from ..config import Settings
from ..schemas import (
    AnalyticsReport,
    AnomalyRecord,
    DiagnosisBucket,
    Recommendation,
)
from ..zones import ZONE_CATALOG

#: Utility rate used when a zone-specific rate is not supplied ($/kWh).
DEFAULT_RATE_USD_PER_KWH = 0.15

#: Root-cause tags that close-loop to specific retrofit measures.
DIAGNOSIS_TO_RECOMMENDATION: Dict[str, str] = {
    "UNOCCUPIED_LIGHTING_WASTE": "led_occupancy_lighting",
    "DEMAND_VENTILATION_OVERCOOLING": "dcv_demand_ventilation",
    "HUMIDITY_ENVELOPE_LEAK": "envelope_sealing",
    "OFF_HOURS_BASELINE_DRIFT": "plug_management",
}


class AnalyticsEngine:
    """Turns closed anomalies into a ranked, monetised report."""

    def __init__(
        self,
        settings: Settings,
        rate: float = DEFAULT_RATE_USD_PER_KWH,
        catalog: List[Recommendation] | None = None,
    ) -> None:
        self.settings = settings
        self.rate = rate
        self.zone_rates: Dict[str, float] = {}
        self.catalog = list(catalog or _DEFAULT_CATALOG(rate))

    # ---------------------------------------------------------------- reporting
    def _buckets(self, anomalies: List[AnomalyRecord]) -> Dict[str, DiagnosisBucket]:
        """Bucket closed anomalies per diagnosis tag with loss aggregates."""
        buckets: Dict[str, List[AnomalyRecord]] = defaultdict(list)
        for a in anomalies:
            buckets[a.diagnosis].append(a)
        by_tag: Dict[str, DiagnosisBucket] = {}
        for tag, items in buckets.items():
            first = items[0]
            by_tag[tag] = DiagnosisBucket(
                count=len(items),
                kwh=round(sum(a.energy_wasted_kwh for a in items), 3),
                usd=round(sum(a.financial_waste_usd for a in items), 2),
                severity=first.severity,
                marker=first.marker,
                zones=sorted({a.zone_id for a in items}),
            )
        return by_tag

    # ------------------------------------------------------- recommendations
    def _scored_recommendations(
        self, anomalies: List[AnomalyRecord]
    ) -> List[Recommendation]:
        """Scale each catalogue measure and rank by payback then ROI.

        A zone contributes to a measure's case when the zone produced one of the
        measure's target diagnoses.  Savings scale linearly with touched zones.
        """
        touched_zones: Dict[str, set[str]] = defaultdict(set)
        for a in anomalies:
            rec_id = DIAGNOSIS_TO_RECOMMENDATION.get(a.diagnosis)
            if rec_id:
                touched_zones[rec_id].add(a.zone_id)

        scored: List[Recommendation] = []
        for rec in self.catalog:
            zones = touched_zones.get(rec.id)
            if not zones:
                continue
            factor = float(len(zones))
            annual_kwh = round(rec.annual_energy_savings_kwh * factor, 2)
            effective_rate = (
                sum(self.zone_rates.get(zone, self.rate) for zone in zones) / len(zones)
                if zones else self.rate
            )
            annual_saving_usd = annual_kwh * effective_rate
            maintenance = round(rec.maintenance_usd_per_year * factor, 2)
            net = annual_saving_usd - maintenance
            payback_years = rec.capital_cost_usd / max(1e-6, net)
            roi = (
                net / rec.capital_cost_usd * 100.0 if rec.capital_cost_usd else 0.0
            )
            scored.append(
                Recommendation(
                    id=rec.id,
                    title=rec.title,
                    category=rec.category,
                    applies_to=sorted(zones),
                    capital_cost_usd=round(rec.capital_cost_usd, 2),
                    maintenance_usd_per_year=maintenance,
                    annual_energy_savings_kwh=annual_kwh,
                    lifetime_years=rec.lifetime_years,
                    rate_usd_per_kwh=effective_rate,
                    payback_months=round(payback_years * 12.0, 2),
                    annual_roi_pct=round(roi, 2),
                    annual_net_savings_usd=round(net, 2),
                )
            )
        # Strict spec sort: payback ascending, then ROI descending.
        scored.sort(key=lambda r: (r.payback_months, -r.annual_roi_pct))
        return scored

    def active_window(
        self,
        zone_id: str,
        report: AnalyticsReport,
    ) -> Dict[str, float]:
        """Latest per-zone exposure estimates for dashboard header cards."""
        zone_anoms = [a for a in report.anomalies if a.zone_id == zone_id] if report else []
        return {
            "zone_id": zone_id,
            "energy_wasted_kwh": round(sum(a.energy_wasted_kwh for a in zone_anoms), 3),
            "financial_waste_usd": round(sum(a.financial_waste_usd for a in zone_anoms), 2),
            "anomaly_count": len(zone_anoms),
        }

    def build_report(
        self,
        anomalies: List[AnomalyRecord],
        current_markers: Dict[str, str],
        uptime_seconds: float,
    ) -> AnalyticsReport:
        """Assemble the final JSON report from closed anomalies + live state."""
        period_seconds = max(uptime_seconds, 1.0)
        return AnalyticsReport(
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            period_seconds=round(period_seconds, 1),
            uptime_seconds=round(uptime_seconds, 1),
            total_energy_wasted_kwh=round(
                sum(a.energy_wasted_kwh for a in anomalies), 3
            ),
            total_financial_waste_usd=round(
                sum(a.financial_waste_usd for a in anomalies), 2
            ),
            total_anomalies=len(anomalies),
            open_anomalies=int(sum(1 for z in ZONE_CATALOG if current_markers.get(z.zone_id) != "green")),
            current_markers=current_markers,
            by_diagnosis=self._buckets(anomalies),
            anomalies=list(reversed(anomalies)),
            top_recommendations=self._scored_recommendations(anomalies),
        )

    def as_markdown(self, report: AnalyticsReport) -> str:
        """Render the audit brief as GitHub-flavoured markdown."""
        lines = [
            "# Energy-Waste Audit Report",
            f"_Generated {report.generated_at}_",
            f"- Energy wasted: **{report.total_energy_wasted_kwh:,.2f} kWh**",
            f"- Financial waste: **${report.total_financial_waste_usd:,.2f}**",
            f"- Anomalies: **{report.total_anomalies}** closed windows ("
            f"{report.open_anomalies} currently flagged)",
            "",
            "## 1. Live zone status",
            "| Zone | Marker | Diagnosis | Waste to date |",
            "|---|---|---|---:|",
        ]
        for zone in ZONE_CATALOG:
            marker = report.current_markers.get(zone.zone_id, "green")
            win = self.active_window(zone.zone_id, report)
            lines.append(
                f"| {zone.zone_id} | {marker} | "
                f"{'*' if marker != 'green' else '-'} | "
                f"${win['financial_waste_usd']:,.2f} |"
            )
        lines += [
            "",
            "## 2. Waste by root cause",
            "| Diagnosis | Count | kWh | USD |",
            "|---|---:|---:|---:|",
        ]
        for tag, d in sorted(
            report.by_diagnosis.items(), key=lambda kv: -kv[1].usd
        ):
            lines.append(
                f"| `{tag}` | {d.count} | {d.kwh:,.1f} | ${d.usd:,.2f} |"
            )
        lines += [
            "",
            "## 3. Closed anomaly events",
            "| Start | Zone | Diagnosis | kWh | USD |",
            "|---|---|---|---:|---:|",
        ]
        for a in report.anomalies[:20]:
            lines.append(
                f"| {a.start_timestamp} | {a.zone_id} | `{a.diagnosis}` | "
                f"{a.energy_wasted_kwh:,.2f} | ${a.financial_waste_usd:,.2f} |"
            )
        lines += [
            "",
            "## 4. Recommendation ranking (payback, then ROI)",
            "| Rank | Measure | Capital | Annual net | Payback | ROI |",
            "|---:|---|---|---:|---:|---:|",
        ]
        for i, r in enumerate(report.top_recommendations, start=1):
            lines.append(
                f"| {i} | {r.title} | ${r.capital_cost_usd:,.0f} | "
                f"${r.annual_net_savings_usd:,.0f} | {r.payback_months:.1f} mo | "
                f"{r.annual_roi_pct:.1f}% |"
            )
        lines += [
            "",
            "> Measures sorted strictly by payback period ascending; annual ROI "
            "used as the secondary tie-breaker.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendation catalogue
# ---------------------------------------------------------------------------
def _DEFAULT_CATALOG(rate: float) -> List[Recommendation]:
    """Static catalogue; savings are per-zone annual kWh.

    Each measure maps to the diagnosis tag(s) it fixes (see
    ``DIAGNOSIS_TO_RECOMMENDATION``); the engine scales savings by the number
    of zones that exhibited the matching diagnosis.
    """
    return [
        Recommendation(
            id="led_occupancy_lighting",
            title="LED + occupancy-sensor lighting retrofit",
            category="Retrofit",
            capital_cost_usd=26_000,
            maintenance_usd_per_year=1_200,
            annual_energy_savings_kwh=43_000,
            lifetime_years=12,
            rate_usd_per_kwh=rate,
        ),
        Recommendation(
            id="dcv_demand_ventilation",
            title="Demand-controlled ventilation (DCV) retro-commissioning",
            category="Retrofit",
            capital_cost_usd=18_000,
            maintenance_usd_per_year=900,
            annual_energy_savings_kwh=38_500,
            lifetime_years=10,
            rate_usd_per_kwh=rate,
        ),
        Recommendation(
            id="envelope_sealing",
            title="Envelope leak sealing + dehumidifier rebalance",
            category="Retrofit",
            capital_cost_usd=24_000,
            maintenance_usd_per_year=1_300,
            annual_energy_savings_kwh=34_000,
            lifetime_years=15,
            rate_usd_per_kwh=rate,
        ),
        Recommendation(
            id="plug_management",
            title="Smart plug-strip + off-hours scheduling programme",
            category="Behavioural",
            capital_cost_usd=9_000,
            maintenance_usd_per_year=500,
            annual_energy_savings_kwh=24_000,
            lifetime_years=10,
            rate_usd_per_kwh=rate,
        ),
        Recommendation(
            id="hvac_schedule_policy",
            title="Occupancy-driven HVAC setback schedule policy",
            category="Behavioural",
            capital_cost_usd=2_500,
            maintenance_usd_per_year=150,
            annual_energy_savings_kwh=16_000,
            lifetime_years=5,
            rate_usd_per_kwh=rate,
        ),
    ]
