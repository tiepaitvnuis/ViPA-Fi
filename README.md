# ViPA-Fi

Visibility-Aware Probabilistic Skeleton Estimation for WiFi-based 3D Human Pose Estimation.

## Introduction

ViPA-Fi estimates 3D human pose from WiFi CSI while explicitly modeling joint-level uncertainty and visibility. The model keeps the CSI encoder and graph pose decoder structure, then adds:

- skeleton evidence aggregation for CSI-to-joint feature construction
- joint uncertainty and visibility prediction
- visibility-aware residual refinement for difficult limb and extremity joints
- optional CSI perturbation consistency for robustness experiments

The default training path is resource-conscious: consistency augmentation is disabled by default because it requires an additional forward pass.

## Requirements

The code is developed and tested under:

- Python 3.10.16
- PyTorch 2.7.0
- CUDA 12.8

## Dataset

We use the MM-Fi dataset. Please request access and follow the official instructions: [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi). Place the downloaded data under your dataset root and set the path in the config:

```yaml
dataset_root:
```

## Train

Run ViPA-Fi on MM-Fi Protocol 1 Setting 1:

```shell
python train.py --config_file config/mmfi/pose_config_p1s1.yaml
```

Explicit resource-efficient configuration:

```shell
python train.py --config_file config/mmfi/pose_config_p1s1.yaml --experiment_name vipafi --agg_mode joint_attn --ext_loss_weight 0.35 --coarse_loss_weight 0.25 --nll_loss_weight 0.10 --visibility_loss_weight 0.02 --consistency_weight 0
```

Optional robustness run with CSI perturbation consistency:

```shell
python train.py --config_file config/mmfi/pose_config_p1s3.yaml --experiment_name vipafi_consistency --consistency_weight 0.03
```

## Outputs

The training script logs:

- MPJPE, PA-MPJPE and PCK
- per-joint MPJPE
- extremity MPJPE for elbows and hands
- mean predicted visibility
- mean predicted uncertainty

Best metrics are saved to `best_metrics.json` under the experiment log directory.

## Notes

The current implementation builds on public pose-decoding components and keeps ablation flags available for controlled experiments. Use `--disable_refinement`, `--agg_mode attn2`, and zero auxiliary loss weights when comparing against a simpler baseline.
