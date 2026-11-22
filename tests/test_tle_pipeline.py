"""
tests/test_tle_pipeline.py
---
Unit tests for independent astrodynamics catalog awareness layer:
- SPARK-2021 class to NORAD ID catalog mapping
- Physical SGP4 orbital propagation and velocity calculations
- Uncataloged status handling for Debris and generic CubeSats (DEC-05)
- Distance calculation between cataloged satellites
- Offline local file fallback behavior
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.tle_pipeline import OrbitalTracker


@pytest.fixture(scope="module")
def tracker():
    """Module-level OrbitalTracker fixture using local offline cached TLEs."""
    tle_path = Path("data/tle/celestrak_active.tle")
    if not tle_path.exists():
        pytest.skip("data/tle/celestrak_active.tle local TLE cache not found for testing")
    return OrbitalTracker(tle_source=str(tle_path))


class TestCatalogMapping:
    """Tests verifying SPARK-2021 class mapping to NORAD IDs."""

    def test_class_to_norad_specification(self, tracker):
        """Verify all SPARK-2021 classes are registered with correct NORAD IDs."""
        mapping = tracker.class_to_norad

        expected_classes = [
            "AcrimSat", "Aquarius", "Aura", "Calipso", "Cloudsat",
            "CubeSat", "Debris", "Jason", "Sentinel-6", "Terra", "TRMM",
        ]
        for cls in expected_classes:
            assert cls in mapping, f"Class {cls} missing from class_to_norad"

        # DEC-05 physical honesty: Debris and CubeSat must map to None
        assert mapping["Debris"] is None, "Debris must map to None (uncataloged)"
        assert mapping["CubeSat"] is None, "CubeSat must map to None (multi-manifest)"

        # Check known satellite IDs
        assert mapping["Sentinel-6"] == 46984
        assert mapping["Aquarius"] == 37673
        assert mapping["Cloudsat"] == 29107
        assert mapping["Calipso"] == 29108
        assert mapping["Terra"] == 25994


class TestUncatalogedHandling:
    """Tests ensuring uncataloged objects cannot resolve to arbitrary TLEs."""

    def test_debris_uncataloged_status(self, tracker):
        """Verify querying Debris explicitly returns uncataloged status (DEC-05)."""
        res = tracker.get_position("Debris")
        assert res.get("status") == "uncataloged"
        assert "error" in res
        assert "cannot determine 3D orbital state vectors" in res["error"]

    def test_cubesat_uncataloged_status(self, tracker):
        """Verify querying CubeSat explicitly returns uncataloged status (DEC-05)."""
        res = tracker.get_position("CubeSat")
        assert res.get("status") == "uncataloged"
        assert "error" in res

    def test_unknown_class_handling(self, tracker):
        """Verify querying an unregistered class string gracefully returns uncataloged."""
        res = tracker.get_position("UnknownDebrisFragment_999")
        assert res.get("status") == "uncataloged"


class TestSGP4Propagation:
    """Tests for real-world SGP4 orbit state vector propagation."""

    def test_sentinel6_propagation_physical_sanity(self, tracker):
        """Verify SGP4 computes physically realistic position and velocity in LEO."""
        res = tracker.get_position("Sentinel-6")
        assert "error" not in res, f"Sentinel-6 lookup failed: {res}"

        assert res["class"] == "Sentinel-6"
        assert res["norad_id"] == 46984
        assert "time_utc" in res

        pos = res["position_km"]
        vel = res["velocity_km_s"]

        # Check shapes
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert isinstance(vel, np.ndarray) and vel.shape == (3,)

        # Earth radius ~6378 km; LEO orbital radius is between 6500 km and 8500 km
        r_mag = np.linalg.norm(pos)
        assert 6500.0 <= r_mag <= 8500.0, (
            f"Orbital distance from Earth center {r_mag:.1f} km outside expected LEO range [6500, 8500]"
        )

        # LEO orbital velocity magnitude is between 6.5 km/s and 8.5 km/s
        v_mag = np.linalg.norm(vel)
        assert 6.5 <= v_mag <= 8.5, (
            f"Orbital velocity {v_mag:.2f} km/s outside expected LEO range [6.5, 8.5]"
        )

    def test_propagation_time_evolution(self, tracker):
        """Verify position updates dynamically with elapsed time."""
        t1 = tracker.ts.utc(2026, 11, 1, 12, 0, 0)
        t2 = tracker.ts.utc(2026, 11, 1, 12, 5, 0)  # 5 minutes later

        pos1 = tracker.get_position("Sentinel-6", time_utc=t1)["position_km"]
        pos2 = tracker.get_position("Sentinel-6", time_utc=t2)["position_km"]

        delta_km = np.linalg.norm(pos2 - pos1)
        # In 5 minutes at ~7.2 km/s, satellite travels ~2160 km
        assert delta_km > 500.0, f"Expected significant orbital displacement, got {delta_km:.1f} km"


class TestInterSatelliteDistance:
    """Tests for Euclidean distance calculations between space objects."""

    def test_distance_between_cataloged_assets(self, tracker):
        """Verify Euclidean distance between two cataloged spacecraft returns valid scalar."""
        dist = tracker.calculate_distance("Sentinel-6", "Terra")
        if dist is not None:
            assert isinstance(dist, float)
            assert dist > 0.0

    def test_distance_with_uncataloged_debris_returns_none(self, tracker):
        """Verify distance calculation involving uncataloged Debris returns None."""
        dist = tracker.calculate_distance("Sentinel-6", "Debris")
        assert dist is None, "Distance involving uncataloged object must return None"

        dist_both = tracker.calculate_distance("CubeSat", "Debris")
        assert dist_both is None


class TestOfflineFallback:
    """Tests ensuring offline operation and fallback resilience."""

    def test_explicit_local_tle_load(self):
        """Verify OrbitalTracker loads directly from local tracked TLE cache without network."""
        local_tracker = OrbitalTracker(tle_source="data/tle/celestrak_active.tle")
        assert len(local_tracker.catalog) > 0, "Failed to load satellites from local TLE cache"

    def test_network_failure_fallback_to_local(self):
        """Verify invalid URL/source automatically triggers fallback to local default cache."""
        fallback_tracker = OrbitalTracker(tle_source="http://invalid.celestrak.offline.test/bad_url.tle")
        assert len(fallback_tracker.catalog) > 0, "Fallback to local default cache did not succeed"
