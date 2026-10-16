"""
scripts/generate_phase1_report_pdf.py
---
Automated PDF Report Generator for Phase 1: Foundations, Evaluation Harness & Baseline Findings.
Reflects strict engineering rigor, realistic hardware proxy labeling, honest metric presentation,
and explicit disclosure of dataset class scope (10 spacecraft classes, 0 debris instances).
"""

import os
import subprocess
import sys
from pathlib import Path

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 1 Engineering Report — Space Object Detection System</title>
<style>
  @page {
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-right {
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8.5pt;
      color: #64748b;
    }
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.45;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
  }

  .header {
    border-bottom: 2px solid #0f172a;
    padding-bottom: 10px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }

  .title-group h1 {
    font-size: 17pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 3px 0;
    letter-spacing: -0.5px;
  }

  .title-group h2 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #2563eb;
    margin: 0 0 3px 0;
  }

  .title-group .meta {
    font-size: 8pt;
    color: #64748b;
  }

  .badge-container {
    text-align: right;
  }

  .badge {
    display: inline-block;
    padding: 3px 8px;
    font-size: 8pt;
    font-weight: 700;
    border-radius: 4px;
    margin-bottom: 3px;
  }

  .badge-success {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
  }

  .badge-warning {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
  }

  .scope-banner {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    padding: 8px 12px;
    margin-bottom: 14px;
    border-radius: 0 4px 4px 0;
    font-size: 8.5pt;
    color: #92400e;
  }

  .scope-banner strong {
    color: #78350f;
  }

  /* Metric Cards */
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 14px;
  }

  .stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 10px;
    border-left: 4px solid #2563eb;
  }

  .stat-card.amber {
    border-left-color: #f59e0b;
  }

  .stat-card.blue {
    border-left-color: #2563eb;
  }

  .stat-card.slate {
    border-left-color: #64748b;
  }

  .stat-title {
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 2px;
  }

  .stat-value {
    font-size: 14pt;
    font-weight: 800;
    color: #0f172a;
  }

  .stat-desc {
    font-size: 7.5pt;
    color: #475569;
    margin-top: 1px;
  }

  h3 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 12px;
    margin-bottom: 6px;
  }

  p {
    margin: 0 0 6px 0;
    text-align: justify;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin-bottom: 10px;
    page-break-inside: avoid;
  }

  th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 5px 6px;
    border: 1px solid #cbd5e1;
  }

  td {
    padding: 4px 6px;
    border: 1px solid #e2e8f0;
    color: #334155;
  }

  tr:nth-child(even) {
    background-color: #f8fafc;
  }

  .tag-pass {
    display: inline-block;
    padding: 1px 5px;
    font-size: 7pt;
    font-weight: 700;
    color: #166534;
    background-color: #bbf7d0;
    border-radius: 3px;
  }

  .tag-pending {
    display: inline-block;
    padding: 1px 5px;
    font-size: 7pt;
    font-weight: 700;
    color: #9a3412;
    background-color: #ffedd5;
    border-radius: 3px;
  }

  .callout {
    background: #f1f5f9;
    border-left: 3px solid #64748b;
    padding: 6px 10px;
    margin-bottom: 10px;
    font-size: 8pt;
    color: #334155;
    border-radius: 0 4px 4px 0;
  }

  .code-block {
    background: #0f172a;
    color: #f8fafc;
    padding: 6px 10px;
    border-radius: 4px;
    font-family: Consolas, Monaco, monospace;
    font-size: 7.5pt;
    margin-bottom: 10px;
    white-space: pre-wrap;
    line-height: 1.35;
  }

  .page-break {
    page-break-before: always;
  }

  .footer-note {
    font-size: 7.5pt;
    color: #94a3b8;
    text-align: center;
    border-top: 1px solid #e2e8f0;
    padding-top: 6px;
    margin-top: 14px;
  }
</style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="title-group">
      <h1>Space Object Detection System</h1>
      <h2>Phase 1 Engineering Report & Baseline Findings</h2>
      <div class="meta">
        <strong>ISRO Problem Statement #05</strong> | Phase 01: Foundations & Day-1 Audit | Date: October 16, 2026
      </div>
    </div>
    <div class="badge-container">
      <div class="badge badge-success">&#10003; Phase 1 Complete</div><br>
      <div class="badge badge-warning">Physical Pi 4 Gate: Pending</div>
    </div>
  </div>

  <!-- Critical Dataset Scope Disclosure -->
  <div class="scope-banner">
    <strong>CRITICAL DATASET SCOPE DISCLOSURE:</strong> The project data currently contains <strong>10 spacecraft classes and ZERO debris annotations</strong> (confirmed by 80,000 label audit). This is strictly an edge-deployable multi-spacecraft recognition system. Neither the codebase, metrics, nor reports claim debris detection performance until real debris annotations are acquired.
  </div>

  <!-- Metric Cards -->
  <div class="grid-3">
    <div class="stat-card blue">
      <div class="stat-title">Headline Accuracy (Val)</div>
      <div class="stat-value">94.13%</div>
      <div class="stat-desc">mAP@0.5:0.95 (mAP@0.5 = 99.50% saturated, see Caveats)</div>
    </div>
    <div class="stat-card amber">
      <div class="stat-title">Latency: Laptop CPU Proxy</div>
      <div class="stat-value">85.94 ms</div>
      <div class="stat-desc">i5-11400H 4-thread proxy (Real Pi target: &le; 1000 ms, PENDING)</div>
    </div>
    <div class="stat-card slate">
      <div class="stat-title">SPARK Label Audit</div>
      <div class="stat-value">80,000</div>
      <div class="stat-desc">Labels audited across 10 classes (0 debris rows, DEC-04)</div>
    </div>
  </div>

  <!-- Section 1: Executive Summary & Objective -->
  <h3>1. Executive Summary & Objective Status</h3>
  <p>
    Phase 1 established the evaluation infrastructure and resolved critical dataset ambiguities for the Edge-Deployable Space Object Detection System. Prior to Phase 1, the evaluation harness was an empty 0-byte stub, SPARK dataset class composition was unverified, and edge benchmarking lacked standardization.
  </p>
  <p>
    <strong>Core Outcomes:</strong>
    1. Built standalone evaluation harness (<code>src/eval.py</code>) writing standardized <code>metrics.json</code> artifacts.<br>
    2. Audited 80,000 SPARK annotations, discovering that generic debris does not exist in the dataset, locking <strong>Decision DEC-04 (Branch B: 10-Class Spacecraft Recognition)</strong>.<br>
    3. Refactored <code>edge/benchmark.py</code> to execute &ge;10 warmup and &ge;100 timed runs with peak RSS memory profiling.<br>
    4. Evaluated baseline inference across hardware tiers, establishing proxy metrics while acknowledging that <strong>the Thursday milestone ("model running on physical Raspberry Pi 4") remains PENDING</strong>.
  </p>

  <!-- Section 2: Hardware & Execution Matrix -->
  <h3>2. Hardware Tiers & Proxy Attribution</h3>
  <p>
    To ensure scientific honesty, all benchmark measurements are attributed to their exact hardware execution environments without conflating proxy numbers with physical edge results:
  </p>
  <table>
    <thead>
      <tr>
        <th style="width: 22%;">Dimension</th>
        <th style="width: 38%;">Compute Workstation (Quadro GV100)</th>
        <th style="width: 40%;">Host Laptop (CPU Proxy & RTX 2050)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Role</strong></td>
        <td><strong>ML Lead</strong> (Deep Training & Export)</td>
        <td><strong>Data & Edge Lead</strong> (Audits, Augment, Local Proxy)</td>
      </tr>
      <tr>
        <td><strong>Hardware</strong></td>
        <td>Intel Xeon / High-Core + NVIDIA Quadro GV100 (32 GB)</td>
        <td>Intel Core i5-11400H (6C/12T) + NVIDIA RTX 2050 (4 GB)</td>
      </tr>
      <tr>
        <td><strong>Edge Status</strong></td>
        <td>Compute Server (Training Only)</td>
        <td><strong>Host CPU Proxy</strong> (Simulating 4 threads, NOT a Pi 4)</td>
      </tr>
      <tr>
        <td><strong>Physical Pi 4</strong></td>
        <td>N/A</td>
        <td><strong>Pending Hardware Deployment</strong> (Target: &le; 1000 ms per PRD)</td>
      </tr>
    </tbody>
  </table>

  <!-- Section 3: Standalone Evaluation & Metric Integrity -->
  <h3>3. Evaluation Harness & Metric Analysis (<code>src/eval.py</code>)</h3>
  <p>
    The standalone CLI <code>src/eval.py</code> evaluates checkpoints independently of training scripts. On the initial 50-epoch SPARK model, the evaluation yielded:
  </p>
  <div class="code-block">--------------------------------------------------
  STANDALONE EVALUATION SUMMARY (SPARK 10-Class)
--------------------------------------------------
  mAP@0.5:0.95: 0.9413  (94.13% — HEADLINE METRIC)
  mAP@0.5:      0.9950  (99.50% — Saturated)
  Precision:    0.9996  (99.96%)
  Recall:       0.9986  (99.86%)
  GPU Latency:  18.50 ms (RTX 2050) / 3.44 ms (GV100)
  Resolution:   1024x1024 (Legacy Smoke Test Size)
--------------------------------------------------</div>

  <div class="callout">
    <strong>Validation Integrity & Inflation Caveats:</strong><br>
    &bull; <strong>Metric Saturation:</strong> At mAP@0.5 = 0.9950, the metric is near-ceiling and cannot show meaningful differences during INT8 quantization or 416 px tradeoff analysis. <strong>mAP@0.5:0.95 (94.13%) must serve as the primary headline metric.</strong><br>
    &bull; <strong>Sequence Leakage Risk:</strong> The current SPARK train/val split is likely random per-image rather than trajectory-aware. If adjacent video frames from the same orbital sequence appear in both splits, metrics are artificially optimistic. Phase 2 must enforce strict trajectory-aware splitting.<br>
    &bull; <strong>Validation Selection Bias:</strong> <code>best.pt</code> was checkpointed on the same validation set it was evaluated on. A frozen, held-out test split must be evaluated before claiming production generalization.
  </div>

  <!-- Page Break -->
  <div class="page-break"></div>

  <!-- Section 4: Day-1 SPARK Annotation Audit -->
  <h3>4. Day-1 SPARK Dataset Annotation Audit (DEC-04)</h3>
  <p>
    Using <code>scripts/audit_spark_annotations.py</code>, we parsed all 80,000 ground-truth labels across <code>train.csv</code> and <code>val.csv</code>:
  </p>
  <table>
    <thead>
      <tr>
        <th>Class Name</th>
        <th>Train Count</th>
        <th>Validation Count</th>
        <th>Total Count</th>
        <th>Debris Flag</th>
        <th>Audit Finding</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Cheops</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>LisaPathfinder</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>ObservationSat1</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>Proba2</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>Proba3</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>Proba3ocs</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>Smart1</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>Soho</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>VenusExpress</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr><td>XMM Newton</td><td>6,000</td><td>2,000</td><td>8,000</td><td>No</td><td>Spacecraft</td></tr>
      <tr style="font-weight: 700; background: #e2e8f0;">
        <td>TOTALS</td><td>60,000</td><td>20,000</td><td>80,000</td><td>0 (0.00%)</td><td>10-Class Lock (DEC-04)</td>
      </tr>
    </tbody>
  </table>
  <div class="callout">
    <strong>Phase 2 Subsampling Plan is Stale:</strong> The earlier plan called for 2:1 majority-class subsampling under the assumption of severe imbalance between spacecraft and debris. Because all 10 spacecraft classes have exactly 8,000 images each, there is <strong>zero class imbalance</strong>. With the 32 GB Quadro GV100, training on the full 60k training split at 640 px is practical; artificial subsampling is unnecessary.
  </div>

  <!-- Section 5: Benchmark Dissection -->
  <h3>5. Latency Benchmark Analysis: Host CPU Proxy vs Edge Reality</h3>
  <p>
    We profiled ONNX Runtime across thread counts and resolutions. The comparison below clarifies previous confusion between 1-thread p95 latency and 4-thread mean latency:
  </p>
  <table>
    <thead>
      <tr>
        <th>Model / Resolution</th>
        <th>Hardware Tier</th>
        <th>Threads</th>
        <th>Mean Latency</th>
        <th>p95 Latency</th>
        <th>FPS</th>
        <th>Peak RSS</th>
        <th>Target Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>SwissCube (640x640)</td>
        <td>Laptop CPU Proxy (i5-11400H)</td>
        <td>1 Thread</td>
        <td>170.92 ms</td>
        <td>219.49 ms</td>
        <td>5.9</td>
        <td>118.6 MB</td>
        <td>Reference (Explains "210 ms")</td>
      </tr>
      <tr>
        <td>SwissCube (640x640)</td>
        <td>Laptop CPU Proxy (i5-11400H)</td>
        <td>4 Threads</td>
        <td>46.44 ms</td>
        <td>54.69 ms</td>
        <td>21.5</td>
        <td>121.8 MB</td>
        <td>Proxy Baseline</td>
      </tr>
      <tr>
        <td>SPARK (1024x1024)</td>
        <td>Laptop CPU Proxy (i5-11400H)</td>
        <td>1 Thread</td>
        <td>182.38 ms</td>
        <td>191.49 ms</td>
        <td>5.5</td>
        <td>204.7 MB</td>
        <td>Legacy 1024 px Single-Thread</td>
      </tr>
      <tr style="background: #eff6ff; font-weight: 700;">
        <td>SPARK (1024x1024)</td>
        <td>Laptop CPU Proxy (i5-11400H)</td>
        <td>4 Threads</td>
        <td>85.94 ms</td>
        <td>97.62 ms</td>
        <td>11.6</td>
        <td>207.1 MB</td>
        <td><strong>Laptop Proxy (NOT a Pi 4)</strong></td>
      </tr>
      <tr style="background: #fef2f2;">
        <td>Physical Pi 4 (Cortex-A72)</td>
        <td>Raspberry Pi 4B (4 GB)</td>
        <td>4 Threads</td>
        <td><em>Pending Run</em></td>
        <td><em>Pending Run</em></td>
        <td><em>--</em></td>
        <td><em>--</em></td>
        <td><strong>PRD Gate: &le; 1000 ms (PENDING)</strong></td>
      </tr>
    </tbody>
  </table>
  <div class="callout">
    <strong>Resolution Realignment (DEC-06):</strong> The 1024 px model was an early smoke-test configuration. The primary production model must be trained and benchmarked at <strong>640x640 px</strong> per DEC-06 so that accuracy, memory, and edge latency are evaluated on identical input tensors.
  </div>

  <!-- Section 6: Phase 1 Acceptance Status -->
  <h3>6. Phase 1 Deliverable Status & Phase 2 Action Plan</h3>
  <table>
    <thead>
      <tr>
        <th>Workstream</th>
        <th>Deliverable</th>
        <th>Demonstrated Reality</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Evaluation CLI</td>
        <td><code>src/eval.py</code></td>
        <td>Standalone metric calculation; exports standardized JSON artifacts.</td>
        <td><span class="tag-pass">Complete</span></td>
      </tr>
      <tr>
        <td>Taxonomy Audit</td>
        <td><code>scripts/audit_spark_annotations.py</code></td>
        <td>80k labels audited; confirmed 0 debris; DEC-04 locked.</td>
        <td><span class="tag-pass">Complete</span></td>
      </tr>
      <tr>
        <td>Augmentation</td>
        <td><code>src/data/augment.py</code></td>
        <td>Photometric jitter, blur, noise with ETA tracking; 4,030 images generated.</td>
        <td><span class="tag-pass">Complete</span></td>
      </tr>
      <tr>
        <td>Benchmark Protocol</td>
        <td><code>edge/benchmark.py</code></td>
        <td>&ge;10 warmup, &ge;100 runs, thread control, peak RSS profiling.</td>
        <td><span class="tag-pass">Complete</span></td>
      </tr>
      <tr>
        <td>Physical Edge Run</td>
        <td>Model running on physical Pi 4</td>
        <td>Measured on laptop CPU proxy; physical deployment to Pi 4 pending.</td>
        <td><span class="tag-pending">Pending Physical Run</span></td>
      </tr>
    </tbody>
  </table>

  <p><strong>Phase 2 Action Plan:</strong></p>
  <ul>
    <li><strong>Retrain at 640 px (DEC-06):</strong> Retire 1024 px as legacy; train primary YOLOv8-Nano on full 60k training split at 640 px.</li>
    <li><strong>Trajectory-Aware Splitting:</strong> Prevent sequence leakage by ensuring all frames of any sequence reside entirely in train or validation.</li>
    <li><strong>Carve-out Frozen Test Split:</strong> Create an untouched test split to report honest, uninflated mAP@0.5:0.95.</li>
    <li><strong>Physical Pi 4 Deployment:</strong> Transfer 640 px ONNX/NCNN exports to physical Raspberry Pi 4 and measure actual ARM latency against the PRD target (&le; 1000 ms).</li>
  </ul>

  <div class="footer-note">
    Space Object Detection System &bull; ISRO Problem Statement #05 &bull; Phase 1 Technical Record &bull; Updated October 16, 2026
  </div>

</body>
</html>
"""


def find_browser_executable() -> str:
    """Locate Microsoft Edge or Google Chrome executable on Windows."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def main():
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_path = docs_dir / "phase1_report.html"
    pdf_path = docs_dir / "Phase_1_Comprehensive_Engineering_Report.pdf"

    # 1. Write the styled HTML document
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print(f"[OK] Report HTML written to: {html_path.resolve()}")

    # 2. Convert to PDF using Headless Edge / Chrome
    browser_exe = find_browser_executable()
    if browser_exe:
        print(f"[INFO] Found browser at: {browser_exe}")
        print("Rendering publication-quality PDF via headless browser...")
        cmd = [
            browser_exe,
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            f"--print-to-pdf={pdf_path.resolve()}",
            str(html_path.resolve()),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            print(f"\n==================================================================")
            print(f" [SUCCESS] Phase 1 PDF Report regenerated successfully!")
            print(f" PDF Location: {pdf_path.resolve()}")
            print(f" PDF Size:     {pdf_path.stat().st_size / 1024:.1f} KB")
            print(f"==================================================================\n")
            return
        else:
            print(f"[warn] Headless browser conversion returned code {res.returncode}. Output: {res.stderr}")

    print(f"\n[NOTE] HTML export saved to: {html_path.resolve()}\n")


if __name__ == "__main__":
    main()
