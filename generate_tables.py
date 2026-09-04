"""
Generate paper-style comparison tables after training ViPA-Fi.
Reproduces Table 1 (SOTA comparison) and Table 2 (Per-joint MPJPE) from GraphPose-Fi paper,
with ViPA-Fi results added.

Usage:
    python generate_tables.py                          # Auto-find results
    python generate_tables.py --logs_dir logs           # Specify logs directory
"""

import os
import json
import argparse
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Baseline results from GraphPose-Fi paper (ICASSP 2026), Table 1
# ═══════════════════════════════════════════════════════════════════

BASELINES_TABLE1 = {
    "Setting 1 (Random Split)": {
        "MetaFi++ [15]":   {"PCK@10": 24.7, "PCK@20": 56.1, "PCK@30": 72.9, "PCK@40": 82.5, "PCK@50": 88.1, "MPJPE": 174.5, "PA-MPJPE": 112.9},
        "HPE-Li [17]":     {"PCK@10": 26.4, "PCK@20": 56.4, "PCK@30": 72.6, "PCK@40": 82.0, "PCK@50": 87.8, "MPJPE": 172.6, "PA-MPJPE": 102.0},
        "DT-Pose [24]":    {"PCK@10": 26.3, "PCK@20": 57.2, "PCK@30": 74.1, "PCK@40": 83.7, "PCK@50": 88.9, "MPJPE": 168.0, "PA-MPJPE": 102.4},
        "GraphPose-Fi":    {"PCK@10": 33.3, "PCK@20": 61.1, "PCK@30": 75.6, "PCK@40": 83.9, "PCK@50": 89.3, "MPJPE": 160.6, "PA-MPJPE": 105.0},
    },
    "Setting 2 (Cross-Subject)": {
        "MetaFi++ [15]":   {"PCK@10":  9.9, "PCK@20": 40.3, "PCK@30": 64.7, "PCK@40": 79.0, "PCK@50": 86.9, "MPJPE": 214.8, "PA-MPJPE": 118.8},
        "HPE-Li [17]":     {"PCK@10": 11.1, "PCK@20": 40.4, "PCK@30": 62.6, "PCK@40": 75.9, "PCK@50": 84.3, "MPJPE": 221.4, "PA-MPJPE": 104.4},
        "DT-Pose [24]":    {"PCK@10": 11.6, "PCK@20": 40.2, "PCK@30": 62.1, "PCK@40": 76.1, "PCK@50": 84.8, "MPJPE": 221.1, "PA-MPJPE": 105.8},
        "GraphPose-Fi":    {"PCK@10": 13.1, "PCK@20": 44.2, "PCK@30": 66.4, "PCK@40": 78.8, "PCK@50": 86.3, "MPJPE": 210.5, "PA-MPJPE": 105.5},
    },
    "Setting 3 (Cross-Environment)": {
        "MetaFi++ [15]":   {"PCK@10":  1.4, "PCK@20":  9.6, "PCK@30": 23.1, "PCK@40": 40.5, "PCK@50": 57.3, "MPJPE": 341.8, "PA-MPJPE": 108.8},
        "HPE-Li [17]":     {"PCK@10":  0.5, "PCK@20":  5.8, "PCK@30": 18.1, "PCK@40": 35.6, "PCK@50": 52.3, "MPJPE": 361.1, "PA-MPJPE": 104.4},
        "DT-Pose [24]":    {"PCK@10":  0.8, "PCK@20":  7.9, "PCK@30": 23.7, "PCK@40": 43.5, "PCK@50": 61.0, "MPJPE": 326.9, "PA-MPJPE": 104.7},
        "GraphPose-Fi":    {"PCK@10":  2.7, "PCK@20": 12.9, "PCK@30": 29.2, "PCK@40": 49.6, "PCK@50": 67.2, "MPJPE": 302.7, "PA-MPJPE": 103.0},
    },
}

# ═══════════════════════════════════════════════════════════════════
# Baseline per-joint MPJPE from Table 2 (P1-S1)
# ═══════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "Bot Torso", "L.Hip", "L.Knee", "L.Foot", "R.Hip", "R.Knee", "R.Foot",
    "Center Torso", "Upper Torso", "Neck Base", "Center Head",
    "R.Shoulder", "R.Elbow", "R.Hand", "L.Shoulder", "L.Elbow", "L.Hand"
]

BASELINES_TABLE2 = {
    "MetaFi++ [15]": [116.4, 119.8, 114.3, 112.8, 116.8, 111.9, 115.3, 117.4, 139.7, 166.1, 168.9, 154.5, 257.8, 372.7, 151.9, 252.6],
    "HPE-Li [17]":   [108.3, 111.6, 111.1, 109.6, 114.7, 113.1, 116.2, 117.7, 142.0, 165.6, 166.0, 153.4, 260.7, 381.2, 148.3, 244.0],
    "DT-Pose [24]":  [105.8, 109.5, 111.1, 110.3, 112.5, 113.6, 119.4, 116.3, 141.5, 163.7, 167.6, 153.4, 246.0, 359.7, 149.4, 230.4],
    "GraphPose-Fi":  [ 93.9, 100.2,  99.7, 102.3, 101.2,  99.6, 107.0, 100.4, 123.7, 150.9, 150.8, 141.0, 251.7, 360.4, 137.4, 244.6],
}

SETTING_TO_CONFIG = {
    "protocol1-s1": "Setting 1 (Random Split)",
    "protocol1-s2": "Setting 2 (Cross-Subject)",
    "protocol1-s3": "Setting 3 (Cross-Environment)",
}


def load_vipafi_results(logs_dir="logs"):
    """Load ViPA-Fi results from best_metrics.json files."""
    results = {}
    for setting_key, setting_name in SETTING_TO_CONFIG.items():
        search_paths = [
            os.path.join(logs_dir, "mmfi-csi", setting_key, "pose_scratch"),
        ]
        for search_path in search_paths:
            if not os.path.isdir(search_path):
                continue
            for exp_dir in os.listdir(search_path):
                metrics_file = os.path.join(search_path, exp_dir, "best_metrics.json")
                if os.path.isfile(metrics_file):
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                    bv = data.get("best_values", {})
                    results[setting_name] = {
                        "MPJPE": bv.get("mpjpe"),
                        "PA-MPJPE": bv.get("pampjpe"),
                        "PCK@10": bv.get("pck@10"),
                        "PCK@20": bv.get("pck@20"),
                        "PCK@30": bv.get("pck@30"),
                        "PCK@40": bv.get("pck@40"),
                        "PCK@50": bv.get("pck@50"),
                        "extremity_mpjpe": bv.get("extremity_mpjpe"),
                        "mean_visibility": bv.get("mean_visibility"),
                        "mean_uncertainty": bv.get("mean_uncertainty"),
                        "per_joint_mpjpe": bv.get("per_joint_mpjpe"),
                    }
                    print(f"  [LOADED] {setting_name} from {metrics_file}")
                    break
    return results


def fmt(v, bold=False):
    """Format a numeric value."""
    if v is None:
        return "—"
    s = f"{v:.1f}"
    return f"**{s}**" if bold else s


def generate_table1(vipafi_results):
    """Generate Table 1: SOTA comparison across 3 settings."""
    lines = []
    lines.append("## Table 1. State-of-the-art performance comparisons on MM-Fi (Protocol 1)")
    lines.append("")
    lines.append("↑ higher is better; ↓ lower is better. **Best** values are bolded.")
    lines.append("")

    metric_keys = ["PCK@10", "PCK@20", "PCK@30", "PCK@40", "PCK@50", "MPJPE", "PA-MPJPE"]
    higher_better = {"PCK@10", "PCK@20", "PCK@30", "PCK@40", "PCK@50"}

    for setting_name in ["Setting 1 (Random Split)", "Setting 2 (Cross-Subject)", "Setting 3 (Cross-Environment)"]:
        lines.append(f"### {setting_name}")
        lines.append("")
        lines.append("| Method | PCK@10↑ | PCK@20↑ | PCK@30↑ | PCK@40↑ | PCK@50↑ | MPJPE↓ | PA-MPJPE↓ |")
        lines.append("|--------|---------|---------|---------|---------|---------|--------|-----------|")

        methods = dict(BASELINES_TABLE1[setting_name])
        if setting_name in vipafi_results:
            methods["**ViPA-Fi (Ours)**"] = vipafi_results[setting_name]

        # Find best for each metric
        best = {}
        for mk in metric_keys:
            vals = [m.get(mk) for m in methods.values() if m.get(mk) is not None]
            if vals:
                best[mk] = max(vals) if mk in higher_better else min(vals)

        for method_name, metrics in methods.items():
            cells = []
            for mk in metric_keys:
                v = metrics.get(mk)
                is_best = v is not None and mk in best and abs(v - best[mk]) < 0.05
                cells.append(fmt(v, bold=is_best))
            lines.append(f"| {method_name} | {' | '.join(cells)} |")

        lines.append("")

        # Show improvement
        if setting_name in vipafi_results:
            vr = vipafi_results[setting_name]
            gr = BASELINES_TABLE1[setting_name]["GraphPose-Fi"]
            if vr.get("MPJPE") is not None:
                diff = gr["MPJPE"] - vr["MPJPE"]
                pct = diff / gr["MPJPE"] * 100
                emoji = "✅" if diff > 0 else "⚠️"
                lines.append(f"> {emoji} MPJPE vs GraphPose-Fi: **{-diff:+.1f} mm ({-pct:+.1f}%)**")
                lines.append("")

    return "\n".join(lines)


def generate_table2(vipafi_results):
    """Generate Table 2: Per-joint MPJPE comparisons (P1-S1)."""
    lines = []
    lines.append("## Table 2. Per-joint MPJPE (mm) comparisons on MM-Fi (P1–S1)")
    lines.append("")

    has_vipafi = "Setting 1 (Random Split)" in vipafi_results
    vipafi_pj = None
    if has_vipafi:
        vipafi_pj = vipafi_results["Setting 1 (Random Split)"].get("per_joint_mpjpe")

    if has_vipafi and vipafi_pj:
        lines.append("| Joint | MetaFi++ | HPE-Li | DT-Pose | GraphPose-Fi | **ViPA-Fi** |")
        lines.append("|-------|----------|--------|---------|-------------|------------|")
    else:
        lines.append("| Joint | MetaFi++ | HPE-Li | DT-Pose | GraphPose-Fi |")
        lines.append("|-------|----------|--------|---------|-------------|")

    extremity_ids = {12, 13, 15}  # R.Elbow, R.Hand, L.Elbow

    for i in range(16):  # Paper shows 16 joints
        joint_name = JOINT_NAMES[i]
        
        all_vals = []
        baseline_vals = []
        for method in ["MetaFi++ [15]", "HPE-Li [17]", "DT-Pose [24]", "GraphPose-Fi"]:
            v = BASELINES_TABLE2[method][i]
            baseline_vals.append(v)
            if v is not None:
                all_vals.append(v)

        vf_val = None
        if has_vipafi and vipafi_pj and joint_name in vipafi_pj:
            vf_val = vipafi_pj[joint_name]
            all_vals.append(vf_val)

        best_val = min(all_vals) if all_vals else None

        cells = []
        for v in baseline_vals:
            is_best = v is not None and best_val is not None and abs(v - best_val) < 0.05
            cells.append(fmt(v, bold=is_best))

        if has_vipafi and vipafi_pj:
            is_best = vf_val is not None and best_val is not None and abs(vf_val - best_val) < 0.05
            cells.append(fmt(vf_val, bold=is_best))

        jn = f"* {joint_name}" if i in extremity_ids else joint_name
        lines.append(f"| {jn} | {' | '.join(cells)} |")

    lines.append("")

    # Extremity summary
    if has_vipafi and vipafi_pj:
        lines.append("### Extremity MPJPE Summary")
        lines.append("")
        lines.append("| Joint Group | GraphPose-Fi | ViPA-Fi | Δ |")
        lines.append("|------------|-------------|---------|---|")

        ext_names = ["R.Elbow", "R.Hand", "L.Elbow"]
        ext_ids = [12, 13, 15]
        gf_vals = [BASELINES_TABLE2["GraphPose-Fi"][i] for i in ext_ids]
        vf_vals = [vipafi_pj.get(n) for n in ext_names]

        for name, gf, vf in zip(ext_names, gf_vals, vf_vals):
            if gf is not None and vf is not None:
                diff = vf - gf
                emoji = "✅" if diff < 0 else "⚠️"
                lines.append(f"| {name} | {gf:.1f} | {vf:.1f} | {emoji} {diff:+.1f} |")

        gf_mean = np.mean([v for v in gf_vals if v])
        vf_mean = np.mean([v for v in vf_vals if v])
        if gf_mean and vf_mean:
            diff = vf_mean - gf_mean
            emoji = "✅" if diff < 0 else "⚠️"
            lines.append(f"| **Mean** | **{gf_mean:.1f}** | **{vf_mean:.1f}** | {emoji} **{diff:+.1f}** |")
        lines.append("")

    return "\n".join(lines)


def generate_table_novel(vipafi_results):
    """Generate novel metrics table unique to ViPA-Fi."""
    lines = []
    lines.append("## Table 5. ViPA-Fi Novel Metrics")
    lines.append("")
    lines.append("*Metrics not available in GraphPose-Fi or other baselines.*")
    lines.append("")
    lines.append("| Setting | Extremity MPJPE ↓ | Mean Visibility | Mean Uncertainty |")
    lines.append("|---------|------------------|-----------------|------------------|")

    for setting_name in ["Setting 1 (Random Split)", "Setting 2 (Cross-Subject)", "Setting 3 (Cross-Environment)"]:
        short = setting_name.split("(")[1].rstrip(")")
        if setting_name in vipafi_results:
            vr = vipafi_results[setting_name]
            ext = f"{vr['extremity_mpjpe']:.1f} mm" if vr.get("extremity_mpjpe") else "—"
            vis = f"{vr['mean_visibility']:.4f}" if vr.get("mean_visibility") else "—"
            unc = f"{vr['mean_uncertainty']:.1f} mm" if vr.get("mean_uncertainty") else "—"
            lines.append(f"| {short} | {ext} | {vis} | {unc} |")
        else:
            lines.append(f"| {short} | *pending* | *pending* | *pending* |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper comparison tables")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Logs directory")
    parser.add_argument("--output", type=str, default="reports/comparison_tables.md", help="Output file")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Paper Comparison Tables")
    print("=" * 60)

    print("\nLoading ViPA-Fi results...")
    vipafi_results = load_vipafi_results(args.logs_dir)

    if not vipafi_results:
        print("\n[WARNING] No ViPA-Fi results found yet.")
        print(f"  Searched: {args.logs_dir}/mmfi-csi/protocol1-s*/pose_scratch/*/best_metrics.json")
        print("  Tables will show baseline-only. Re-run after training.")
    else:
        print(f"\n  Found results for {len(vipafi_results)} setting(s)")

    output_lines = []
    output_lines.append("# ViPA-Fi vs GraphPose-Fi — Paper Comparison Tables\n")
    output_lines.append(generate_table1(vipafi_results))
    output_lines.append("\n---\n")
    output_lines.append(generate_table2(vipafi_results))
    output_lines.append("\n---\n")
    output_lines.append(generate_table_novel(vipafi_results))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n[OK] Tables saved to: {args.output}")
    print("\n" + "=" * 80)
    print("\n".join(output_lines))
    print("=" * 80)


if __name__ == "__main__":
    main()
