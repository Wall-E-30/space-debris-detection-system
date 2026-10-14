"""
edge/verify_pi_env.py
---
Testbed Environment Verification Script for Raspberry Pi 4 Model B.
Validates OS architecture, Python version, hardware resources, and installed runtimes.
Supports Day-1 Pi environment gate (EDGE-01, D-03, D-04).
"""

import platform
import sys


def main():
    print("=" * 65)
    print("  RASPBERRY PI 4 EDGE ENVIRONMENT VERIFICATION (EDGE-01)")
    print("=" * 65)

    all_passed = True

    # 1. Python Version Check (>= 3.10)
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        print(f"[PASS] Python version: {py_str} (>= 3.10)")
    else:
        print(f"[FAIL] Python version: {py_str} (Required: >= 3.10)")
        all_passed = False

    # 2. Operating System & Architecture Check
    machine = platform.machine()
    system = platform.system()
    print(f"[*] OS: {system}, Architecture: {machine}")
    if machine in ["aarch64", "arm64"]:
        print(f"[PASS] 64-bit ARM architecture confirmed ({machine})")
    else:
        print(f"[INFO] Running on non-ARM host ({machine}). On Pi 4, expected: aarch64.")

    # 3. Memory & Hardware Checks (via psutil)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024.0 ** 3)
        cores = psutil.cpu_count(logical=True)
        print(f"[PASS] psutil installed. CPU Cores: {cores}, Total RAM: {ram_gb:.2f} GB")
    except ImportError:
        print("[FAIL] psutil is not installed. Run: pip install psutil")
        all_passed = False

    # 4. Core Edge Runtime Dependencies
    modules = ["numpy", "cv2", "onnxruntime"]
    for mod in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"[PASS] {mod} installed ({ver})")
        except ImportError:
            print(f"[FAIL] {mod} is NOT installed.")
            all_passed = False

    # 5. Optional NCNN Runtime Check
    try:
        import ncnn
        print(f"[PASS] ncnn installed (NCNN Python bindings available)")
    except ImportError:
        print("[WARN] ncnn Python bindings not installed. (Optional: ONNX Runtime will serve as fallback per DEC-01 / D-04)")

    # -------------------------------------------------------------
    # Persistent Environment Verification Archiving (outputs/env_checks/)
    # -------------------------------------------------------------
    import datetime, json
    from pathlib import Path
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path("outputs") / "env_checks"
    archive_dir.mkdir(parents=True, exist_ok=True)

    check_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": py_str,
        "system": system,
        "machine": machine,
        "passed": all_passed,
    }

    timestamped_file = archive_dir / f"env_check_{now_str}.json"
    latest_file = archive_dir / "latest.json"

    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(check_record, f, indent=2)

    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(check_record, f, indent=2)

    print(f"[OK] Environment check archived to: {timestamped_file.resolve()}")
    print("=" * 65)
    if all_passed:
        print("[OK] Core edge environment verification PASSED.")
        sys.exit(0)
    else:
        print("[ERROR] Core edge environment verification FAILED. Please review missing packages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
