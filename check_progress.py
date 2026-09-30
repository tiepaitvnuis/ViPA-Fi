from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = r'D:\GraphPose-Fi-main\GraphPose-Fi-main\logs\mmfi-csi\protocol1-s1\pose_scratch\vipafi'
ea = EventAccumulator(log_dir)
ea.Reload()
scalar_tags = ea.Tags().get('scalars', [])

print("=" * 70)
print("ViPA-Fi Training Progress (Protocol 1, Setting 1)")
print("=" * 70)

# Main metrics per epoch
main_tags = ['Loss/Train', 'Loss/Train_Pose', 'Loss/Train_Extremity', 
             'Loss/Train_NLL', 'Loss/Train_Visibility',
             'MPJPE/Val_Extremity', 'Visibility/Val_Mean', 'Uncertainty/Val_Mean',
             'Uncertainty/Val_Correlation', 'LearningRate']

for tag in main_tags:
    if tag in scalar_tags:
        events = ea.Scalars(tag)
        print(f"\n--- {tag} ---")
        for e in events:
            print(f"  Epoch {e.step}: {e.value:.5f}")

# Per-joint MPJPE
print("\n" + "=" * 70)
print("Per-Joint MPJPE (last epoch)")
print("=" * 70)
joint_tags = [t for t in scalar_tags if t.startswith('MPJPE_Joint/')]
for tag in sorted(joint_tags):
    if tag in scalar_tags:
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            # Convert to mm (multiply by 1000 if in meters)
            val = last.value
            name = tag.replace('MPJPE_Joint/', '')
            print(f"  {name:20s}: {val*1000:.1f} mm" if val < 1 else f"  {name:20s}: {val:.1f} mm")

# Best values
print("\n" + "=" * 70)
print("Best Values")
print("=" * 70)
for tag in ['Best/MPJPE', 'Best/PA-MPJPE']:
    if tag in scalar_tags:
        events = ea.Scalars(tag)
        if events:
            print(f"  {tag}: {events[-1].value:.5f}")
for tag in scalar_tags:
    if tag.startswith('Best/PCK'):
        events = ea.Scalars(tag)
        if events:
            print(f"  {tag}: {events[-1].value:.4f}")
