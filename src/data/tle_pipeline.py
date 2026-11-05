# src/data/tle_pipeline.py
from skyfield.api import load
import numpy as np

class OrbitalTracker:
    def __init__(self):
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
        
        # 2. Fetch Active Satellite TLEs from CelesTrak
        print("[INFO] Fetching TLEs from CelesTrak...")
        active_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
        self.satellites = load.tle_file(active_url)
        
        # Build a fast-lookup dictionary by NORAD ID
        self.catalog = {sat.model.satnum: sat for sat in self.satellites}
        print(f"[INFO] Loaded {len(self.catalog)} active satellites into memory.")

    def get_position(self, detected_class, time_utc=None):
        """
        Takes a YOLO detected class string, looks up its TLE, 
        and calculates its XYZ coordinates in km if it is a cataloged asset.
        """
        # Default to "now" if no timestamp is provided
        t = time_utc if time_utc else self.ts.now()
        
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

    def calculate_distance(self, class_a, class_b):
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
        dist = np.linalg.norm(pos_a["position_km"] - pos_b["position_km"])
        return dist

# --- Test the pipeline locally ---
if __name__ == "__main__":
    tracker = OrbitalTracker()
    
    # Simulate YOLO finding AcrimSat and Sentinel-6 in orbital field
    print("\n--- Simulating Detections ---")
    obj1 = tracker.get_position("AcrimSat")
    obj2 = tracker.get_position("Sentinel-6")
    cubesat_test = tracker.get_position("CubeSat")
    debris_test = tracker.get_position("Debris")
    
    print(f"Object 1: {obj1}")
    print(f"Object 2: {obj2}")
    print(f"CubeSat Query: {cubesat_test}")
    print(f"Debris Query: {debris_test}")
    
    if "error" not in obj1 and "error" not in obj2:
        dist = tracker.calculate_distance("AcrimSat", "Sentinel-6")
        print(f"\n[CATALOG AWARENESS] Distance between cataloged assets: {dist:,.2f} km")
