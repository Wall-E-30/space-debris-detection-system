# src/data/tle_pipeline.py
from skyfield.api import load
import numpy as np

class OrbitalTracker:
    def __init__(self):
        print("[INFO] Initializing Orbital Tracker...")
        self.ts = load.timescale()
        
        # 1. Map SPARK-2021 spacecraft classes to real NORAD Catalog IDs.
        # Note: Uncataloged Space Debris has NO deterministic NORAD ID in CelesTrak;
        # SGP4 propagation cannot track uncataloged debris fragments without prior radar/optical
        # orbital state vector determination. The TLE module serves as an independent
        # catalog layer for known spacecraft assets, NOT a detection-to-orbit resolver.
        self.class_to_norad = {
            "AcrimSat": 26032,
            "Aquarius": 37673,
            "Aura": 28376,
            "Calipso": 29108,
            "Cloudsat": 29107,
            "CubeSat": 25544,      # LEO CubeSat/ISS deployment proxy
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
        and calculates its XYZ coordinates in km.
        """
        # Default to "now" if no timestamp is provided
        t = time_utc if time_utc else self.ts.now()
        
        if detected_class == "Debris":
            return {
                "error": "Debris objects are uncataloged and have no deterministic NORAD IDs in CelesTrak. Orbital propagation requires empirical orbit determination."
            }

        norad_id = self.class_to_norad.get(detected_class)
        if not norad_id:
            return {"error": f"Class '{detected_class}' not in mapping dictionary."}
            
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
        Calculates the real-world distance between two detected classes.
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
    debris_test = tracker.get_position("Debris")
    
    print(f"Object 1: {obj1}")
    print(f"Object 2: {obj2}")
    print(f"Debris Query: {debris_test}")
    
    if "error" not in obj1 and "error" not in obj2:
        dist = tracker.calculate_distance("AcrimSat", "Sentinel-6")
        print(f"\n[ALERT] Distance between objects: {dist:,.2f} km")