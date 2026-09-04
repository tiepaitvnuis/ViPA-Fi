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


def make_csi_perturbation(csi, noise_std=0.02, amp_scale=0.05, freq_drop_prob=0.08, ant_drop_prob=0.10):
    B, A, S, T, C = csi.shape
    view = csi * (1.0 + amp_scale * torch.randn(B, A, 1, 1, 1, device=csi.device, dtype=csi.dtype))
    view = view + noise_std * torch.randn_like(view)

    freq_keep = (torch.rand(B, 1, S, 1, 1, device=csi.device, dtype=csi.dtype) > freq_drop_prob).to(csi.dtype)
    ant_keep = (torch.rand(B, A, 1, 1, 1, device=csi.device, dtype=csi.dtype) > ant_drop_prob).to(csi.dtype)
    attenuation = 0.5 + 0.5 * freq_keep * ant_keep
    view = view * attenuation
    return torch.clamp(view, 0.0, 1.0)


def representation_consistency_loss(clean_features, aug_features):
    clean_features = F.normalize(clean_features.detach(), dim=-1)
    aug_features = F.normalize(aug_features, dim=-1)
    return F.mse_loss(aug_features, clean_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ViPA-Fi Training")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default="config/mmfi/pose_config_p1s1.yaml")
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default="vipafi")
    parser.add_argument("--tag", type=str, help="", default="")
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.02)
    parser.add_argument("--total_epoch", type=int, help="Total epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--max_device_batch_size", type=int, help="Max device batch size", default=64)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_frames", type=int, help="Number of frames", default=1)
    parser.add_argument("--dropout", type=float, help="Dropout rate", default=0.1)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader (0=main process only, avoids Windows page file issues)")
    parser.add_argument("--pretrained_weights", action="store_true", help="Use pretrained weights")
    parser.add_argument("--graattention_layers", type=int, default=4, help="Number of layers in the graph attention network")
    parser.add_argument("--agg_mode", type=str, default="joint_attn", choices=["mean", "attn2", "mhsa", "joint_attn"], help="Aggregation mode")
    parser.add_argument("--disable_refinement", action="store_true", help="Disable hierarchical extremity refinement")
    parser.add_argument("--ext_loss_weight", type=float, default=0.35, help="Weight for elbow/hand extremity loss")
    parser.add_argument("--coarse_loss_weight", type=float, default=0.25, help="Auxiliary weight for coarse pose supervision")
    parser.add_argument("--nll_loss_weight", type=float, default=0.10, help="Weight for uncertainty-aware Gaussian NLL")
    parser.add_argument("--uncertainty_scale", type=float, default=100.0, help="Coordinate scale used to stabilize uncertainty NLL")
    parser.add_argument("--visibility_loss_weight", type=float, default=0.02, help="Weight for visibility-error calibration")
    parser.add_argument("--consistency_weight", type=float, default=0.0, help="Optional consistency weight. Keep 0 for resource-efficient main training")
    parser.add_argument("--csi_noise_std", type=float, default=0.02, help="Gaussian noise std for CSI perturbation")
    parser.add_argument("--csi_amp_scale", type=float, default=0.05, help="Amplitude scale jitter for CSI perturbation")
    parser.add_argument("--csi_freq_drop_prob", type=float, default=0.08, help="Frequency-local attenuation probability")
    parser.add_argument("--csi_ant_drop_prob", type=float, default=0.10, help="Antenna/link attenuation probability")
    parser.add_argument("--description", type=str, default="", help="Description of the experiment")

    args = parser.parse_args()
    exp_tag = f"{args.experiment_name}_{args.tag}" if args.tag else args.experiment_name
    
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    setup_seed(args.seed)
    dataset_root = os.path.expanduser(config['dataset_root'])
    if args.tag == 'debug':
        logs_path = os.path.join('logs_debug', config['dataset_name'], config['setting'], 'pose_scratch', exp_tag)
    else:
        logs_path = os.path.join('logs', config['dataset_name'], config['setting'], 'pose_scratch', exp_tag)

    total_epochs = args.total_epoch
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    print(f"Effective Batch Size: {batch_size}")
    print(f"GPU Load Batch Size: {load_batch_size}")
    print(f"Gradient Accumulation Steps: {steps_per_update}")

    # load dataset
    if config['dataset_name'] == 'mmfi-csi':
        print(f"Loading MM-FI dataset from: {dataset_root}")
        train_dataset, val_dataset = make_dataset(config['training_semi'], dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size, num_workers=args.num_workers)
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size, num_workers=args.num_workers)
    else:
        print('No dataset!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    writer = SummaryWriter(logs_path)

    print('*'*20+'   Training from Scratch  '+'*'*20)
    print('*'*20+'  '+config['dataset_name']+','+config['setting']+','+args.experiment_name+'   '+'*'*20)
    if config['dataset_name'] == 'mmfi-csi':
        model = ViPAFiNet(
            num_keypoints=17,
            num_coor=3,
            num_person=config['num_person'],
            dataset=config['dataset_name'],
            pretrained_weights=args.pretrained_weights,
            num_layers=args.graattention_layers,
            agg_mode=args.agg_mode,
            use_refinement=not args.disable_refinement,
        ).to(device)
        if not args.pretrained_weights:
            model.apply(_weights_init)
        warmup_epochs = 5
        def lr_lambda(cur_epoch):
            if cur_epoch < warmup_epochs:
                return (cur_epoch + 1) / warmup_epochs
            progress = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        optim = torch.optim.AdamW(param_groups_simple(model, lr=args.learning_rate, wd=args.weight_decay), lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        print("Initializing ViPA-Fi model...")

        # save model
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], 'pose_scratch', args.experiment_name, exp_tag)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

    optim.zero_grad()
    step_count = 0
    best_val_mpjpe = float('inf')
    best_val_pampjpe = float('inf')
    best_val_mpjpe_align = 1
    best_val_pampjpe_align = 1
    best_val_pck = [0 for _ in range(5)]
    best_val_pck_align = [0 for _ in range(5)]
    pck_order = [50, 40, 30, 20, 10]
    best_weights = {
        "mpjpe": None,
        "pampjpe": None,
        "pck@50": None,
        "pck@40": None,
        "pck@30": None,
        "pck@20": None,
        "pck@10": None,
    }
    joint_names = [
        "Bot Torso", "L.Hip", "L.Knee", "L.Foot", "R.Hip", "R.Knee", "R.Foot",
        "Center Torso", "Upper Torso", "Neck Base", "Center Head",
        "R.Shoulder", "R.Elbow", "R.Hand", "L.Shoulder", "L.Elbow", "L.Hand"
    ]
    extremity_joint_ids = [12, 13, 15, 16]
    best_per_joint_mpjpe = None
    best_extremity_mpjpe = None
    best_val_visibility = None
    best_val_uncertainty = None

    for epoch in range(1, args.total_epoch+1):
        model.train()
        epoch_train_losses = []
        losses = []
        pose_losses = []
        ext_losses = []
        coarse_losses = []
        nll_losses = []
        visibility_losses = []
        cons_losses = []
        optim.zero_grad()
        start_time = time.time()
        
        mpjpe_list = []
       
        pck_iter = [[] for _ in range(5)]
        pck_align_iter = [[] for _ in range(5)]
        for batch_idx, batch_data in enumerate(train_loader):
            csi_data = batch_data['input_wifi-csi'].unsqueeze(-1)
            csi_data = csi_data.to(device) 
            pose_gt = batch_data['output'].to(device)
            predicted_pose, features, aux = model(csi_data, return_aux=True)

            loss_pose = joint_distance_loss(predicted_pose, pose_gt)
            loss_ext = joint_distance_loss(predicted_pose, pose_gt, extremity_joint_ids) if args.ext_loss_weight > 0 else predicted_pose.new_tensor(0.0)
            loss_coarse = joint_distance_loss(aux["coarse_pose"], pose_gt) if args.coarse_loss_weight > 0 else predicted_pose.new_tensor(0.0)
            loss_nll = probabilistic_pose_loss(predicted_pose, pose_gt, aux["log_var"], scale=args.uncertainty_scale) if args.nll_loss_weight > 0 else predicted_pose.new_tensor(0.0)
            loss_visibility = visibility_calibration_loss(predicted_pose, pose_gt, aux["visibility"]) if args.visibility_loss_weight > 0 else predicted_pose.new_tensor(0.0)
            loss_cons = predicted_pose.new_tensor(0.0)
            if args.consistency_weight > 0:
                csi_aug = make_csi_perturbation(
                    csi_data,
                    noise_std=args.csi_noise_std,
                    amp_scale=args.csi_amp_scale,
                    freq_drop_prob=args.csi_freq_drop_prob,
                    ant_drop_prob=args.csi_ant_drop_prob,
                )
                aug_predicted_pose, aug_features, _ = model(csi_aug, return_aux=True)
                loss_cons = representation_consistency_loss(features, aug_features)
                loss_cons = loss_cons + 0.1 * F.smooth_l1_loss(aug_predicted_pose, predicted_pose.detach())

            total_loss = (
                loss_pose
                + args.ext_loss_weight * loss_ext
                + args.coarse_loss_weight * loss_coarse
                + args.nll_loss_weight * loss_nll
                + args.visibility_loss_weight * loss_visibility
                + args.consistency_weight * loss_cons
            )
            mpjpe, _, _, _ = calculate_error(predicted_pose.detach().cpu().numpy(), pose_gt.detach().cpu().numpy(), align=False)
            mpjpe_list += mpjpe.tolist()
            losses.append(total_loss.item())
            pose_losses.append(loss_pose.item())
            ext_losses.append(loss_ext.item())
            coarse_losses.append(loss_coarse.item())
            nll_losses.append(loss_nll.item())
            visibility_losses.append(loss_visibility.item())
            cons_losses.append(loss_cons.item())

            total_loss = total_loss / steps_per_update
            total_loss.backward()
            step_count += 1

            if (step_count % steps_per_update == 0) or (batch_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_( (p for p in model.parameters() if p.requires_grad), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
        scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_pose_loss = sum(pose_losses) / len(pose_losses)
        avg_ext_loss = sum(ext_losses) / len(ext_losses)
        avg_coarse_loss = sum(coarse_losses) / len(coarse_losses)
        avg_nll_loss = sum(nll_losses) / len(nll_losses)
        avg_visibility_loss = sum(visibility_losses) / len(visibility_losses)
        avg_cons_loss = sum(cons_losses) / len(cons_losses)
        avg_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        epoch_time = time.time() - start_time
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Pose', avg_pose_loss, epoch)
        writer.add_scalar('Loss/Train_Extremity', avg_ext_loss, epoch)
        writer.add_scalar('Loss/Train_Coarse', avg_coarse_loss, epoch)
        writer.add_scalar('Loss/Train_NLL', avg_nll_loss, epoch)
        writer.add_scalar('Loss/Train_Visibility', avg_visibility_loss, epoch)
        writer.add_scalar('Loss/Train_Consistency', avg_cons_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        print(f'Epoch {epoch}/{total_epochs} | Train Loss: {avg_train_loss:.4f} | Pose: {avg_pose_loss:.4f} | Ext: {avg_ext_loss:.4f} | NLL: {avg_nll_loss:.4f} | Vis: {avg_visibility_loss:.4f} | Cons: {avg_cons_loss:.4f} | Train mpjpe: {avg_mpjpe:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')

        # Validation
        model.eval()
        epoch_val_losses = []
        all_val_preds = []
        all_val_gts = []
        with torch.no_grad():
            losses = []
            mpjpe_list = []
            pampjpe_list = []
            
            pck_iter = [[] for _ in range(5)]
            pck_align_iter = [[] for _ in range(5)]
            subject_mpjpe = {}
            mpjpe_joints_list = []
            visibility_list = []
            uncertainty_list = []
            for batch_idx, batch_data in enumerate(val_loader):
                val_csi_data = batch_data['input_wifi-csi'].unsqueeze(-1)
                val_csi_data = val_csi_data.to(device)  
                val_pose_gt = batch_data['output'].to(device)
                predicted_val_pose, _, val_aux = model(val_csi_data, return_aux=True)               
                loss = torch.mean(torch.norm(predicted_val_pose-val_pose_gt, dim=-1))
                # calculate the pck, mpjpe, pampjpe
                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(predicted_val_pose.permute(0,2,1).detach().cpu().numpy(), val_pose_gt.permute(0,2,1).detach().cpu().numpy(), percentage, align=False, dataset=config['dataset_name']))
                mpjpe, pampjpe, mpjpe_joints, pampjpe_joints = calculate_error(predicted_val_pose.detach().cpu().numpy(), val_pose_gt.detach().cpu().numpy(), align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()
                mpjpe_joints_list.append(mpjpe_joints)
                visibility_list.append(val_aux["visibility"].detach().cpu().numpy())
                uncertainty_list.append((torch.exp(0.5 * val_aux["log_var"]) * args.uncertainty_scale).mean(dim=-1).detach().cpu().numpy())
                losses.append(loss.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
            avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
            avg_val_mpjpe_joints = np.mean(np.stack(mpjpe_joints_list, axis=0), axis=0)
            avg_val_extremity_mpjpe = float(np.mean(avg_val_mpjpe_joints[extremity_joint_ids]))
            avg_val_visibility = float(np.mean(np.concatenate(visibility_list, axis=0)))
            avg_val_uncertainty = float(np.mean(np.concatenate(uncertainty_list, axis=0)))
            if config['dataset_name'] == 'mmfi-csi':
                pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]           
        print(f'In epoch {epoch}, test losss: {avg_val_loss}')
        print(f'test mpjpe: {avg_val_mpjpe}, test pa-mpjpe: {avg_val_pampjpe}, test pck50: {pck_overall[0]}, test pck40: {pck_overall[1]}, test pck30: {pck_overall[2]}, test pck20: {pck_overall[3]}, test pck10: {pck_overall[4]}.')
        print(f'test extremity mpjpe (elbows/hands): {avg_val_extremity_mpjpe}')
        print(f'test mean visibility: {avg_val_visibility}, test mean uncertainty: {avg_val_uncertainty}')

        ''' save model '''
        # save mpjpe pampjpe
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            best_per_joint_mpjpe = avg_val_mpjpe_joints.tolist()
            best_extremity_mpjpe = avg_val_extremity_mpjpe
            best_weights["mpjpe"] = f'{weights_path}/pose_mpjpe.pt'
            best_val_visibility = avg_val_visibility
            best_val_uncertainty = avg_val_uncertainty
            torch.save(model.state_dict(), '{}/pose_mpjpe.pt'.format(weights_path)) 
        if avg_val_pampjpe < best_val_pampjpe:
            best_val_pampjpe = avg_val_pampjpe
            best_weights["pampjpe"] = f'{weights_path}/pose_pampjpe.pt'
            torch.save(model.state_dict(), '{}/pose_pampjpe.pt'.format(weights_path)) 
        for idx, pck_value in enumerate(pck_overall):
            if pck_value > best_val_pck[idx]:
                best_val_pck[idx] = pck_value
                pck_tag = pck_order[idx]
                best_weights[f"pck@{pck_tag}"] = f'{weights_path}/pose_pck{pck_tag}.pt'
                torch.save(model.state_dict(), '{}/pose_pck{}.pt'.format(weights_path, pck_order[idx]))
        
        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=epoch)
        writer.add_scalar('MPJPE/Val_Extremity', avg_val_extremity_mpjpe, epoch)
        writer.add_scalar('Visibility/Val_Mean', avg_val_visibility, epoch)
        writer.add_scalar('Uncertainty/Val_Mean', avg_val_uncertainty, epoch)
        for i, j_name in enumerate(joint_names):
            writer.add_scalar(f'MPJPE_Joint/{j_name.replace(" ", "_")}', avg_val_mpjpe_joints[i], epoch)
            
        val_errors_flat = np.concatenate(mpjpe_joints_list, axis=0).flatten()  # (num_batches, num_joints) -> flat
        val_uncerts_stacked = np.concatenate(uncertainty_list, axis=0)  # (total_samples, num_joints)
        # Average uncertainty per joint across all samples to match mpjpe_joints shape
        # mpjpe_joints_list has shape (num_batches, num_joints) - already batch-averaged
        # uncertainty_list has shape (num_batches, batch_size, num_joints) - per sample
        # We need to average uncertainty per batch to match
        val_uncerts_per_batch = np.stack([np.mean(u, axis=0) for u in uncertainty_list], axis=0).flatten()
        if val_errors_flat.shape == val_uncerts_per_batch.shape:
            val_uncert_corr = np.corrcoef(val_errors_flat, val_uncerts_per_batch)[0, 1]
        else:
            val_uncert_corr = 0.0  # fallback if shapes still don't match
        writer.add_scalar('Uncertainty/Val_Correlation', val_uncert_corr, epoch)
        torch.cuda.empty_cache()

    print('*'*100)
    print(f'Best mpjpe: {best_val_mpjpe}') 
    print(f'Best pa-mpjpe: {best_val_pampjpe}')  
    for idx, pck_value in enumerate(best_val_pck):
        print(f'Best pck{pck_order[idx]}: {pck_value}')  
    print('*'*100)
    # ─── Write TensorBoard Scalar ────────────────────────
    writer.add_scalar('Best/MPJPE', best_val_mpjpe, global_step=0)
    writer.add_scalar('Best/PA-MPJPE', best_val_pampjpe, global_step=0)
    for idx, pck_value in enumerate(best_val_pck):
        writer.add_scalar(f'Best/PCK@{pck_order[idx]}', pck_value, global_step=0)

    # ─── Write TensorBoard Text (Summary) ─────────────────────
    summary_text = f"Best MPJPE: {best_val_mpjpe:.5f} mm\nBest PA-MPJPE: {best_val_pampjpe:.5f} mm\n"
    if best_extremity_mpjpe is not None:
        summary_text += f"Best-run Extremity MPJPE: {best_extremity_mpjpe:.5f} mm\n"
    if best_val_visibility is not None and best_val_uncertainty is not None:
        summary_text += f"Best-run Mean Visibility: {best_val_visibility:.5f}\n"
        summary_text += f"Best-run Mean Uncertainty: {best_val_uncertainty:.5f}\n"
    summary_text += "\n".join([f"PCK@{pck_order[idx]}: {pck_value:.4f}" for idx, pck_value in enumerate(best_val_pck)])
    writer.add_text("Eval/Best_Results", summary_text, global_step=0)
    args_save_path = os.path.join(logs_path, "args.json")
    args_dict = vars(args)
    log_config(config, logs_path, writer)
    os.makedirs(os.path.dirname(args_save_path), exist_ok=True)
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    best_json = {
        "best_values": {
            "mpjpe": best_val_mpjpe,
            "pampjpe": best_val_pampjpe,
            "pck@50": best_val_pck[0],
            "pck@40": best_val_pck[1],
            "pck@30": best_val_pck[2],
            "pck@20": best_val_pck[3],
            "pck@10": best_val_pck[4],
            "extremity_mpjpe": best_extremity_mpjpe,
            "mean_visibility": best_val_visibility,
            "mean_uncertainty": best_val_uncertainty,
            "per_joint_mpjpe": dict(zip(joint_names, best_per_joint_mpjpe)) if best_per_joint_mpjpe is not None else None,
        },
        "weights": best_weights,
        "paths": {
            "logs_path": logs_path,
            "weights_path": weights_path,
        }
    }
    with open(os.path.join(logs_path, "best_metrics.json"), "w") as f:
        json.dump(best_json, f, indent=4)

    writer.close()
    
    
