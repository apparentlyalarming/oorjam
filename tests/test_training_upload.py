from __future__ import annotations

from datetime import datetime, timedelta, timezone

import joblib

from app.ml.training import ARTIFACT_FILENAME, PIPELINE_VERSION, train_from_points
from app.schemas import Telemetry, TelemetryPoint
from app.zones import ZONE_BY_ID


def test_uploaded_samples_retrain_and_write_local_artifacts(settings, tmp_path):
    settings.model_dir = tmp_path
    settings.window_minutes = 2
    settings.window_min_samples = 10
    settings.isolation_forest_n_estimators = 20
    zone = ZONE_BY_ID["floor_1_west"]
    start = datetime(2026, 9, 25, 8, tzinfo=timezone.utc)
    points = []
    for i in range(30):
        occupancy = i % 20
        point = TelemetryPoint(
            timestamp=(start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
            building_id=zone.building_id,
            zone_id=zone.zone_id,
            telemetry=Telemetry(
                occupancy_count=occupancy,
                co2_ppm=400 + occupancy * 2.6,
                humidity_percent=50,
                temp_indoor_c=22,
                temp_outdoor_c=28,
                hvac_kw=15 + occupancy * 0.2,
                lighting_kw=2 + occupancy * 0.1,
                plug_load_kw=5 + occupancy * 0.08,
            ),
        )
        points.append(point)

    counts = train_from_points(points, settings)

    assert counts == {zone.zone_id: 30}
    payload = joblib.load(tmp_path / ARTIFACT_FILENAME)
    assert payload["pipeline_version"] == PIPELINE_VERSION
    assert payload["n_samples"] > 0
    assert (tmp_path / "baseline_models.joblib").exists()
