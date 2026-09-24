"""Analytics-engine unit tests: bucketing, ROI/payback scoring and markdown."""

from __future__ import annotations

from app.analytics.reporting import AnalyticsEngine, DIAGNOSIS_TO_RECOMMENDATION
from app.schemas import AnalyticsReport, AnomalyRecord


def _anomaly(zone: str, tag: str, kwh: float, usd: float) -> AnomalyRecord:
    return AnomalyRecord(
        start_timestamp="2026-09-24T10:00:00Z",
        end_timestamp="2026-09-24T10:05:00Z",
        zone_id=zone,
        diagnosis=tag,
        severity="high",
        marker="red",
        iforest_score=-0.5,
        energy_wasted_kwh=kwh,
        financial_waste_usd=usd,
        peak_kw=20.0,
        sample_count=6,
        duration_hours=0.01,
        details={},
    )


def test_diagnosis_to_recommendation_covers_all_tags():
    tags = {
        "DEMAND_VENTILATION_OVERCOOLING",
        "HUMIDITY_ENVELOPE_LEAK",
        "UNOCCUPIED_LIGHTING_WASTE",
        "OFF_HOURS_BASELINE_DRIFT",
    }
    assert DIAGNOSIS_TO_RECOMMENDATION.keys() == tags


def test_ranking_payback_asc_roi_desc(settings):
    engine = AnalyticsEngine(settings)
    scored = engine._scored_recommendations(
        [
            _anomaly("floor_2_east", "UNOCCUPIED_LIGHTING_WASTE", 100, 15),
            _anomaly("floor_1_west", "DEMAND_VENTILATION_OVERCOOLING", 80, 12),
            _anomaly("floor_2_east", "OFF_HOURS_BASELINE_DRIFT", 60, 9),
        ]
    )
    paybacks = [r.payback_months for r in scored]
    assert paybacks == sorted(paybacks)
    # Strict secondary tie-breaker: ROI descending.
    for i in range(len(scored) - 1):
        if scored[i].payback_months == scored[i + 1].payback_months:
            assert scored[i].annual_roi_pct >= scored[i + 1].annual_roi_pct
    for r in scored:
        assert r.annual_net_savings_usd > 0
        assert r.payback_months > 0


def test_zone_factor_scales_savings(settings):
    engine = AnalyticsEngine(settings)
    one = engine._scored_recommendations([_anomaly("floor_2_east", "UNOCCUPIED_LIGHTING_WASTE", 1, 1)])[0]
    two = engine._scored_recommendations(
        [
            _anomaly("floor_2_east", "UNOCCUPIED_LIGHTING_WASTE", 1, 1),
            _anomaly("floor_1_west", "UNOCCUPIED_LIGHTING_WASTE", 1, 1),
        ]
    )[0]
    assert two.annual_energy_savings_kwh == 2 * one.annual_energy_savings_kwh


def test_empty_report_renders(settings):
    engine = AnalyticsEngine(settings)
    report = engine.build_report(anomalies=[], current_markers={}, uptime_seconds=10.0)
    assert isinstance(report, AnalyticsReport)
    assert report.total_anomalies == 0
    md = engine.as_markdown(report)
    assert md.startswith("# Energy-Waste Audit Report")
    assert "0.00 kWh" in md


def test_buckets_aggregate(settings):
    engine = AnalyticsEngine(settings)
    report = engine.build_report(
        anomalies=[
            _anomaly("floor_2_east", "UNOCCUPIED_LIGHTING_WASTE", 10, 1.5),
            _anomaly("floor_1_west", "UNOCCUPIED_LIGHTING_WASTE", 20, 3.0),
        ],
        current_markers={"floor_2_east": "green"},
        uptime_seconds=60.0,
    )
    bucket = report.by_diagnosis["UNOCCUPIED_LIGHTING_WASTE"]
    assert bucket.count == 2 and bucket.kwh == 30.0 and set(bucket.zones) == {
        "floor_2_east",
        "floor_1_west",
    }