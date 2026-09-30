"""
Fine-tuning script for ViPA-Fi.
Strategy:
  1. Load the best checkpoint from vipafi_v2 / vipafi_v2_resume
  2. Freeze the backbone (ResNet encoder) to prevent catastrophic forgetting
  3. Use 3x higher extremity loss weight to focus on hand/elbow joints
  4. Constant low learning rate (no warmup, no cosine decay)
  5. Test-Time Augmentation (TTA) during evaluation for free improvement
"""

import os
import argparse
import json
import math
import yaml
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from feeder.mmfi import make_dataset, make_dataloader
from utils import *
from model.model import ViPAFiNet, _weights_init


def joint_distance_loss(pred, target, joint_ids=None):
    if joint_ids is not None:
        pred = pred[:, joint_ids, :]
        target = target[:, joint_ids, :]
    return torch.mean(torch.norm(pred - target, dim=-1))


def probabilistic_pose_loss(pred, target, log_var, joint_ids=None, scale=100.0):
    if joint_ids is not None:
        pred = pred[:, joint_ids, :]
        target = target[:, joint_ids, :]
        log_var = log_var[:, joint_ids, :]
    sq_error = ((pred - target) / scale).pow(2)
    return 0.5 * torch.mean(torch.exp(-log_var) * sq_error + log_var)


def visibility_calibration_loss(pred, target, visibility, joint_ids=None, scale=180.0):
    if joint_ids is not None:
        pred = pred[:, joint_ids, :]
        target = target[:, joint_ids, :]
        visibility = visibility[:, joint_ids, :]
    joint_error = torch.norm(pred.detach() - target, dim=-1, keepdim=True)
    visibility_target = torch.exp(-joint_error / scale).clamp(min=0.05, max=0.95)
    return F.binary_cross_entropy(visibility.clamp(min=1e-4, max=1.0 - 1e-4), visibility_target)


def joint_weighted_distance_loss(pred, target, weights):
    """Weighted per-joint loss: joints with higher weight get more attention."""
    per_joint = torch.norm(pred - target, dim=-1)  # [B, J]
    weighted = per_joint * weights.unsqueeze(0)     # [B, J]
    return torch.mean(weighted)


def tta_predict(model, csi_data, num_aug=5, noise_std=0.01):
    """
    Test-Time Augmentation: run multiple slightly noisy forward passes
    and average the predictions. Free improvement without retraining.
    """
    preds = []
    with torch.no_grad():
        # Original (clean) prediction
        pred_clean, _, aux_clean = model(csi_data, return_aux=True)
        preds.append(pred_clean)

        # Augmented predictions
        for _ in range(num_aug - 1):
            noise = torch.randn_like(csi_data) * noise_std
            csi_noisy = torch.clamp(csi_data + noise, 0.0, 1.0)
            pred_aug, _, _ = model(csi_noisy, return_aux=True)
            preds.append(pred_aug)

    # Average all predictions
    avg_pred = torch.stack(preds, dim=0).mean(dim=0)
    return avg_pred, aux_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ViPA-Fi Fine-tuning")
    parser.add_argument("--config_file", type=str, default="config/mmfi/pose_config_p1s1.yaml")
    parser.add_argument("--experiment_name", type=str, default="vipafi_v2_finetune")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint to fine-tune from")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Constant low learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--total_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_device_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ext_loss_weight", type=float, default=1.0, help="3x higher than default 0.35")
    parser.add_argument("--nll_loss_weight", type=float, default=0.10)
    parser.add_argument("--visibility_loss_weight", type=float, default=0.02)
    parser.add_argument("--uncertainty_scale", type=float, default=100.0)
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze ResNet backbone")
    parser.add_argument("--no_freeze_backbone", action="store_true", default=False)
    parser.add_argument("--tta_aug", type=int, default=5, help="Number of TTA augmentations during eval")
    parser.add_argument("--tta_noise", type=float, default=0.01, help="TTA noise std")
    parser.add_argument("--use_joint_weights", action="store_true", default=True, help="Use per-joint weighted loss")

    args = parser.parse_args()
    if args.no_freeze_backbone:
        args.freeze_backbone = False

    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    setup_seed(args.seed)
    dataset_root = os.path.expanduser(config['dataset_root'])
    logs_path = os.path.join('logs', config['dataset_name'], config['setting'], 'pose_scratch', args.experiment_name)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    steps_per_update = batch_size // load_batch_size

    print(f"=== ViPA-Fi Fine-tuning ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Learning Rate: {args.learning_rate} (constant)")
    print(f"Ext Loss Weight: {args.ext_loss_weight}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"TTA Augmentations: {args.tta_aug}")
    print(f"Batch Size: {batch_size} (GPU load: {load_batch_size})")

    # Load dataset
    if config['dataset_name'] == 'mmfi-csi':
        train_dataset, val_dataset = make_dataset(config['training_semi'], dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size, num_workers=args.num_workers)
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    writer = SummaryWriter(logs_path)

    # Build model and load checkpoint
    model = ViPAFiNet(
        num_keypoints=17, num_coor=3,
        num_person=config['num_person'],
        dataset=config['dataset_name'],
        pretrained_weights=False,
        num_layers=4,
        agg_mode='joint_attn',
        use_refinement=True,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully!")

    # Freeze backbone if requested
    if args.freeze_backbone:
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(key in name for key in ['f_conv1', 'f_bn1', 'f_layer1', 'f_layer2', 'f_layer3', 'f_layer4', 'bn2']):
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} backbone parameters")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # Per-joint loss weights: higher weight for extremity joints
    # Joint order: [Bot_Torso, L.Hip, L.Knee, L.Foot, R.Hip, R.Knee, R.Foot,
    #               Center_Torso, Upper_Torso, Neck_Base, Center_Head,
    #               R.Shoulder, R.Elbow, R.Hand, L.Shoulder, L.Elbow, L.Hand]
    if args.use_joint_weights:
        joint_weights = torch.ones(17, device=device)
        # Elbows get 2x weight
        joint_weights[12] = 2.0  # R.Elbow
        joint_weights[15] = 2.0  # L.Elbow
        # Hands get 3x weight  
        joint_weights[13] = 3.0  # R.Hand
        joint_weights[16] = 3.0  # L.Hand
        # Shoulders get 1.5x weight
        joint_weights[11] = 1.5  # R.Shoulder
        joint_weights[14] = 1.5  # L.Shoulder
        joint_weights = joint_weights / joint_weights.mean()  # Normalize so mean=1
        print(f"Using per-joint weighted loss")
    else:
        joint_weights = None

    # Optimizer: constant LR, only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # NO scheduler -- constant LR throughout fine-tuning

    # Save paths
    weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], 'pose_scratch', args.experiment_name, args.experiment_name)
    os.makedirs(weights_path, exist_ok=True)

    joint_names = [
        "Bot Torso", "L.Hip", "L.Knee", "L.Foot", "R.Hip", "R.Knee", "R.Foot",
        "Center Torso", "Upper Torso", "Neck Base", "Center Head",
        "R.Shoulder", "R.Elbow", "R.Hand", "L.Shoulder", "L.Elbow", "L.Hand"
    ]
    extremity_joint_ids = [12, 13, 15, 16]

    best_val_mpjpe = float('inf')
    best_val_pampjpe = float('inf')
    best_val_pck = [0 for _ in range(5)]
    pck_order = [50, 40, 30, 20, 10]
    best_per_joint_mpjpe = None
    best_extremity_mpjpe = None

    print(f"\n{'='*60}")
    print(f"Starting fine-tuning for {args.total_epoch} epochs...")
    print(f"{'='*60}\n")

    optim.zero_grad()
    step_count = 0

    for epoch in range(1, args.total_epoch + 1):
        model.train()
        losses, pose_losses, ext_losses, nll_losses, vis_losses = [], [], [], [], []
        mpjpe_list = []
        optim.zero_grad()
        start_time = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            csi_data = batch_data['input_wifi-csi'].unsqueeze(-1).to(device)
            pose_gt = batch_data['output'].to(device)
            predicted_pose, features, aux = model(csi_data, return_aux=True)

            # Main pose loss (optionally weighted per-joint)
            if joint_weights is not None:
                loss_pose = joint_weighted_distance_loss(predicted_pose, pose_gt, joint_weights)
            else:
                loss_pose = joint_distance_loss(predicted_pose, pose_gt)

            loss_ext = joint_distance_loss(predicted_pose, pose_gt, extremity_joint_ids)
            loss_nll = probabilistic_pose_loss(predicted_pose, pose_gt, aux["log_var"], scale=args.uncertainty_scale)
            loss_visibility = visibility_calibration_loss(predicted_pose, pose_gt, aux["visibility"])

            total_loss = (
                loss_pose
                + args.ext_loss_weight * loss_ext
                + args.nll_loss_weight * loss_nll
                + args.visibility_loss_weight * loss_visibility
            )

            mpjpe, _, _, _ = calculate_error(predicted_pose.detach().cpu().numpy(), pose_gt.detach().cpu().numpy(), align=False)
            mpjpe_list += mpjpe.tolist()
            losses.append(total_loss.item())
            pose_losses.append(loss_pose.item())
            ext_losses.append(loss_ext.item())
            nll_losses.append(loss_nll.item())
            vis_losses.append(loss_visibility.item())

            total_loss = total_loss / steps_per_update
            total_loss.backward()
            step_count += 1

            if (step_count % steps_per_update == 0) or (batch_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)  # Tighter clipping for fine-tuning
                optim.step()
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        avg_train_loss = sum(losses) / len(losses)
        avg_pose_loss = sum(pose_losses) / len(pose_losses)
        avg_ext_loss = sum(ext_losses) / len(ext_losses)
        avg_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        epoch_time = time.time() - start_time

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Pose', avg_pose_loss, epoch)
        writer.add_scalar('Loss/Train_Extremity', avg_ext_loss, epoch)
        print(f'Epoch {epoch}/{args.total_epoch} | Loss: {avg_train_loss:.4f} | Pose: {avg_pose_loss:.4f} | Ext: {avg_ext_loss:.4f} | Train MPJPE: {avg_mpjpe:.4f} | Time: {epoch_time:.2f}s')

        # Validation (with TTA)
        model.eval()
        mpjpe_list, pampjpe_list = [], []
        mpjpe_joints_list = []
        pck_iter = [[] for _ in range(5)]
        visibility_list, uncertainty_list = [], []

        # Also run standard eval (no TTA) for comparison
        mpjpe_list_noTTA = []
        mpjpe_joints_list_noTTA = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                val_csi = batch_data['input_wifi-csi'].unsqueeze(-1).to(device)
                val_gt = batch_data['output'].to(device)

                # Standard eval (no TTA)
                pred_noTTA, _, aux_noTTA = model(val_csi, return_aux=True)
                mpjpe_nt, _, mpjpe_joints_nt, _ = calculate_error(pred_noTTA.detach().cpu().numpy(), val_gt.detach().cpu().numpy(), align=False)
                mpjpe_list_noTTA += mpjpe_nt.tolist()
                mpjpe_joints_list_noTTA.append(mpjpe_joints_nt)

                # TTA eval
                pred_tta, aux_tta = tta_predict(model, val_csi, num_aug=args.tta_aug, noise_std=args.tta_noise)

                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(pred_tta.permute(0,2,1).detach().cpu().numpy(), val_gt.permute(0,2,1).detach().cpu().numpy(), percentage, align=False, dataset=config['dataset_name']))

                mpjpe, pampjpe, mpjpe_joints, _ = calculate_error(pred_tta.detach().cpu().numpy(), val_gt.detach().cpu().numpy(), align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()
                mpjpe_joints_list.append(mpjpe_joints)
                visibility_list.append(aux_tta["visibility"].detach().cpu().numpy())
                uncertainty_list.append((torch.exp(0.5 * aux_tta["log_var"]) * args.uncertainty_scale).mean(dim=-1).detach().cpu().numpy())

        # Compute metrics
        avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
        avg_val_mpjpe_joints = np.mean(np.stack(mpjpe_joints_list, axis=0), axis=0)
        avg_val_extremity = float(np.mean(avg_val_mpjpe_joints[extremity_joint_ids]))
        avg_noTTA_mpjpe = sum(mpjpe_list_noTTA) / len(mpjpe_list_noTTA)

        pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]

        print(f'  [No TTA] Val MPJPE: {avg_noTTA_mpjpe:.5f}')
        print(f'  [TTA x{args.tta_aug}] Val MPJPE: {avg_val_mpjpe:.5f}, PA-MPJPE: {avg_val_pampjpe:.5f}')
        print(f'  Extremity MPJPE: {avg_val_extremity:.5f}')
        print(f'  PCK@50: {pck_overall[0]:.2f}, PCK@20: {pck_overall[3]:.2f}, PCK@10: {pck_overall[4]:.2f}')

        # Per-joint details
        print('  --- Per Joint (TTA) ---')
        for i, j_name in enumerate(joint_names):
            marker = " <--" if i in extremity_joint_ids else ""
            print(f'    {j_name}: {avg_val_mpjpe_joints[i]:.5f}{marker}')

        # Log to TensorBoard
        writer.add_scalar('MPJPE/Val_NoTTA', avg_noTTA_mpjpe, epoch)
        writer.add_scalar('MPJPE/Val_TTA', avg_val_mpjpe, epoch)
        writer.add_scalar('MPJPE/Val_Extremity', avg_val_extremity, epoch)
        for i, j_name in enumerate(joint_names):
            writer.add_scalar(f'MPJPE_Joint/{j_name.replace(" ", "_")}', avg_val_mpjpe_joints[i], epoch)

        # Save best model
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            best_per_joint_mpjpe = avg_val_mpjpe_joints.tolist()
            best_extremity_mpjpe = avg_val_extremity
            torch.save(model.state_dict(), f'{weights_path}/best_mpjpe.pt')
            print(f'  ** New best MPJPE: {best_val_mpjpe:.5f} -- saved!')
        if avg_val_pampjpe < best_val_pampjpe:
            best_val_pampjpe = avg_val_pampjpe
            torch.save(model.state_dict(), f'{weights_path}/best_pampjpe.pt')
        for idx, pck_value in enumerate(pck_overall):
            if pck_value > best_val_pck[idx]:
                best_val_pck[idx] = pck_value
                torch.save(model.state_dict(), f'{weights_path}/best_pck{pck_order[idx]}.pt')

        torch.cuda.empty_cache()

    # Final Summary
    print('\n' + '='*80)
    print(f'FINE-TUNING COMPLETE')
    print(f'Best MPJPE (TTA): {best_val_mpjpe:.5f}')
    print(f'Best PA-MPJPE: {best_val_pampjpe:.5f}')
    if best_per_joint_mpjpe:
        print('\nBest Per-Joint MPJPE:')
        for i, j_name in enumerate(joint_names):
            print(f'  {j_name}: {best_per_joint_mpjpe[i]:.5f}')
    print(f'\nBest Extremity MPJPE: {best_extremity_mpjpe:.5f}')
    for idx in range(5):
        print(f'Best PCK@{pck_order[idx]}: {best_val_pck[idx]:.4f}')
    print('='*80)

    # Save results
    results = {
        "best_mpjpe": best_val_mpjpe,
        "best_pampjpe": best_val_pampjpe,
        "best_extremity_mpjpe": best_extremity_mpjpe,
        "best_per_joint_mpjpe": dict(zip(joint_names, best_per_joint_mpjpe)) if best_per_joint_mpjpe else None,
        "best_pck": {f"pck@{pck_order[i]}": best_val_pck[i] for i in range(5)},
        "config": {
            "checkpoint": args.checkpoint,
            "learning_rate": args.learning_rate,
            "ext_loss_weight": args.ext_loss_weight,
            "freeze_backbone": args.freeze_backbone,
            "tta_aug": args.tta_aug,
            "total_epoch": args.total_epoch,
        }
    }
    with open(os.path.join(logs_path, "finetune_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {logs_path}/finetune_results.json")

    writer.close()
