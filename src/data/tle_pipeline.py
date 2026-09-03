# src/data/tle_pipeline.py
from skyfield.api import load
import numpy as np

class OrbitalTracker:
    def __init__(self):
        print("[INFO] Initializing Orbital Tracker...")
        self.ts = load.timescale()
        
        # 1. Map SPARK classes to real NORAD Catalog IDs
        # Note: Historical/Deep Space probes (Smart1, VenusExpress) use active LEO proxies 
        # here so the SGP4 math actually works for a collision-risk demonstration.
        self.class_to_norad = {
            "Cheops": 44874,
            "LisaPathfinder": 41124, 
            "ObservationSat1": 25544, # ISS Proxy
            "Proba2": 35932,
            "Proba3": 35932,          # Proxy
            "Proba3ocs": 35932,       # Proxy
            "Smart1": 27957,
            "Soho": 23726,
            "VenusExpress": 28899,
            "XMM Newton": 25989
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
    
    # Simulate YOLO finding a Cheops and an ObservationSat1 in the same frame
    print("\n--- Simulating Detections ---")
    obj1 = tracker.get_position("Cheops")
    obj2 = tracker.get_position("ObservationSat1")
    
    print(f"Object 1: {obj1}")
    print(f"Object 2: {obj2}")
    
    if "error" not in obj1 and "error" not in obj2:
        dist = tracker.calculate_distance("Cheops", "ObservationSat1")
        print(f"\n[ALERT] Distance between objects: {dist:,.2f} km")