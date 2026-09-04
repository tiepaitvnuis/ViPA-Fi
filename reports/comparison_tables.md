# ViPA-Fi vs GraphPose-Fi — Paper Comparison Tables

## Table 1. State-of-the-art performance comparisons on MM-Fi (Protocol 1)

↑ higher is better; ↓ lower is better. **Best** values are bolded.

### Setting 1 (Random Split)

| Method | PCK@10↑ | PCK@20↑ | PCK@30↑ | PCK@40↑ | PCK@50↑ | MPJPE↓ | PA-MPJPE↓ |
|--------|---------|---------|---------|---------|---------|--------|-----------|
| MetaFi++ [15] | 24.7 | 56.1 | 72.9 | 82.5 | 88.1 | 174.5 | 112.9 |
| HPE-Li [17] | 26.4 | 56.4 | 72.6 | 82.0 | 87.8 | 172.6 | **102.0** |
| DT-Pose [24] | 26.3 | 57.2 | 74.1 | 83.7 | 88.9 | 168.0 | 102.4 |
| GraphPose-Fi | **33.3** | **61.1** | **75.6** | **83.9** | **89.3** | **160.6** | 105.0 |

### Setting 2 (Cross-Subject)

| Method | PCK@10↑ | PCK@20↑ | PCK@30↑ | PCK@40↑ | PCK@50↑ | MPJPE↓ | PA-MPJPE↓ |
|--------|---------|---------|---------|---------|---------|--------|-----------|
| MetaFi++ [15] | 9.9 | 40.3 | 64.7 | **79.0** | **86.9** | 214.8 | 118.8 |
| HPE-Li [17] | 11.1 | 40.4 | 62.6 | 75.9 | 84.3 | 221.4 | **104.4** |
| DT-Pose [24] | 11.6 | 40.2 | 62.1 | 76.1 | 84.8 | 221.1 | 105.8 |
| GraphPose-Fi | **13.1** | **44.2** | **66.4** | 78.8 | 86.3 | **210.5** | 105.5 |

### Setting 3 (Cross-Environment)

| Method | PCK@10↑ | PCK@20↑ | PCK@30↑ | PCK@40↑ | PCK@50↑ | MPJPE↓ | PA-MPJPE↓ |
|--------|---------|---------|---------|---------|---------|--------|-----------|
| MetaFi++ [15] | 1.4 | 9.6 | 23.1 | 40.5 | 57.3 | 341.8 | 108.8 |
| HPE-Li [17] | 0.5 | 5.8 | 18.1 | 35.6 | 52.3 | 361.1 | 104.4 |
| DT-Pose [24] | 0.8 | 7.9 | 23.7 | 43.5 | 61.0 | 326.9 | 104.7 |
| GraphPose-Fi | **2.7** | **12.9** | **29.2** | **49.6** | **67.2** | **302.7** | **103.0** |


---

## Table 2. Per-joint MPJPE (mm) comparisons on MM-Fi (P1–S1)

| Joint | MetaFi++ | HPE-Li | DT-Pose | GraphPose-Fi |
|-------|----------|--------|---------|-------------|
| Bot Torso | 116.4 | 108.3 | 105.8 | **93.9** |
| L.Hip | 119.8 | 111.6 | 109.5 | **100.2** |
| L.Knee | 114.3 | 111.1 | 111.1 | **99.7** |
| L.Foot | 112.8 | 109.6 | 110.3 | **102.3** |
| R.Hip | 116.8 | 114.7 | 112.5 | **101.2** |
| R.Knee | 111.9 | 113.1 | 113.6 | **99.6** |
| R.Foot | 115.3 | 116.2 | 119.4 | **107.0** |
| Center Torso | 117.4 | 117.7 | 116.3 | **100.4** |
| Upper Torso | 139.7 | 142.0 | 141.5 | **123.7** |
| Neck Base | 166.1 | 165.6 | 163.7 | **150.9** |
| Center Head | 168.9 | 166.0 | 167.6 | **150.8** |
| R.Shoulder | 154.5 | 153.4 | 153.4 | **141.0** |
| ⭐ R.Elbow | 257.8 | 260.7 | **246.0** | 251.7 |
| ⭐ R.Hand | 372.7 | 381.2 | **359.7** | 360.4 |
| L.Shoulder | 151.9 | 148.3 | 149.4 | **137.4** |
| ⭐ L.Elbow | 252.6 | 244.0 | **230.4** | 244.6 |


---

## Table 5. ViPA-Fi Novel Metrics

*Metrics not available in GraphPose-Fi or other baselines.*

| Setting | Extremity MPJPE ↓ | Mean Visibility | Mean Uncertainty |
|---------|------------------|-----------------|------------------|
| Random Split | *pending* | *pending* | *pending* |
| Cross-Subject | *pending* | *pending* | *pending* |
| Cross-Environment | *pending* | *pending* | *pending* |
