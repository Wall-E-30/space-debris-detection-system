# src/data/tle_pipeline.py
"""
Astrodynamics SGP4 catalog tracking and situational awareness module.
Decoupled catalog awareness layer querying NORAD Two-Line Elements (TLEs)
via Skyfield / SGP4. (DEC-05)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from skyfield.api import load


class OrbitalTracker:
    def __init__(
        self,
        tle_source: Optional[Union[str, Path]] = None,
        reload_remote: bool = False,
    ):
        """
        Initialize the SGP4 Orbital Tracker.

        Args:
            tle_source: Optional path to a local TLE file or URL. If None,
                        defaults to local cache at data/tle/celestrak_active.tle.
            reload_remote: If True, forces downloading fresh TLEs from CelesTrak.
        """
        print("[INFO] Initializing Orbital Tracker...")
        self.ts = load.timescale()

        # 1. Map SPARK-2021 spacecraft classes to real NORAD Catalog IDs.
        # Note: Uncataloged Space Debris and generic CubeSats have NO deterministic single NORAD ID in CelesTrak;
        # SGP4 propagation cannot track uncataloged debris fragments or multi-manifest CubeSats without prior radar/optical
        # orbital state vector determination. The TLE module serves as an independent
        # catalog layer for known spacecraft assets, NOT a detection-to-orbit resolver (DEC-05).
        self.class_to_norad = {
            "AcrimSat": 26032,
            "Aquarius": 37673,
            "Aura": 28376,
            "Calipso": 29108,
            "Cloudsat": 29107,
            "CubeSat": None,       # UNCATALOGED: Multi-manifest CubeSats lack fixed single NORAD ID
            "Jason": 26997,         # Jason-1
            "Sentinel-6": 46984,    # Sentinel-6 Michael Freilich
            "Terra": 25994,
            "TRMM": 25063,
            "Debris": None,         # UNCATALOGED: SGP4 propagation unsupported
        }

        # 2. Determine TLE Source (Local Cache First, Remote Fallback)
        repo_root = Path(__file__).resolve().parent.parent.parent
        default_cache = repo_root / "data" / "tle" / "celestrak_active.tle"
        active_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

        self.satellites = []
        target_source = None

        if tle_source:
            target_source = str(tle_source)
        elif default_cache.exists() and not reload_remote:
            target_source = str(default_cache)
            print(f"[INFO] Loading TLEs from offline cache: {default_cache}")
        else:
            target_source = active_url
            print(f"[INFO] Fetching fresh TLEs from CelesTrak: {active_url}")

        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10.0)  # P3: 10-second network timeout gate
            try:
                self.satellites = load.tle_file(target_source)
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception as exc:
            if target_source != str(default_cache) and default_cache.exists():
                print(f"[WARN] Remote fetch failed ({exc}). Falling back to local cache: {default_cache}")
                self.satellites = load.tle_file(str(default_cache))
            else:
                raise RuntimeError(f"Failed to load orbital TLE catalog from {target_source}: {exc}") from exc

        # Build a fast-lookup dictionary by NORAD ID
        self.catalog = {sat.model.satnum: sat for sat in self.satellites}
        print(f"[INFO] Loaded {len(self.catalog)} active satellites into memory.")

    def get_position(self, detected_class: str, time_utc=None) -> Dict[str, Any]:
        """
        Takes a YOLO detected class string, looks up its TLE, 
        and calculates its XYZ coordinates in km if it is a cataloged asset.
        """
        # Default to "now" if no timestamp is provided (P0: explicit None check for Skyfield scalar Time)
        t = time_utc if time_utc is not None else self.ts.now()

        norad_id = self.class_to_norad.get(detected_class)
        if detected_class in ["Debris", "CubeSat"] or norad_id is None:
            return {
                "status": "uncataloged",
                "error": f"Class '{detected_class}' is uncataloged or has no single fixed NORAD ID in CelesTrak. Direct 2D optical detections cannot determine 3D orbital state vectors without multi-station radar/optical tracklet orbit determination."
            }

        satellite = self.catalog.get(norad_id)

        if not satellite:
            # Fallback if a historical satellite isn't in the active CelesTrak list
            return {"error": f"NORAD ID {norad_id} not found in active TLE catalog."}

        # 3. Propagate the orbit using SGP4
        geocentric = satellite.at(t)
        pos_km = geocentric.position.km

        return {
            "class": detected_class,
            "norad_id": norad_id,
            "time_utc": t.utc_strftime('%Y-%m-%d %H:%M:%S UTC'),
            "position_km": pos_km,
            "velocity_km_s": geocentric.velocity.km_per_s
        }

    def calculate_distance(self, class_a: str, class_b: str) -> Optional[float]:
        """
        Calculates the real-world distance between two verified cataloged satellite classes.
        Note: This is strictly an independent ground-catalog situational awareness utility
        for active catalog assets, NOT a visual conjunction predictor for 2D detections (DEC-05).
        """
        pos_a = self.get_position(class_a)
        pos_b = self.get_position(class_b)

        if "error" in pos_a or "error" in pos_b:
            return None

        # Simple Euclidean distance between the two X,Y,Z vectors
        dist = float(np.linalg.norm(pos_a["position_km"] - pos_b["position_km"]))
        return dist

# --- Test the pipeline locally ---
if __name__ == "__main__":
    tracker = OrbitalTracker()

    # Simulate YOLO finding Terra and Sentinel-6 in orbital field
    print("\n--- Simulating Detections ---")
    obj1 = tracker.get_position("Terra")
    obj2 = tracker.get_position("Sentinel-6")
    cubesat_test = tracker.get_position("CubeSat")
    debris_test = tracker.get_position("Debris")

    print(f"Object 1 (Terra): {obj1}")
    print(f"Object 2 (Sentinel-6): {obj2}")
    print(f"CubeSat Query: {cubesat_test}")
    print(f"Debris Query: {debris_test}")

    if "error" not in obj1 and "error" not in obj2:
        dist = tracker.calculate_distance("Terra", "Sentinel-6")
        print(f"\n[CATALOG AWARENESS] Distance between cataloged assets: {dist:,.2f} km")
