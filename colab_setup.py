"""
ViPA-Fi Colab Setup Script
===========================
Upload file này lên Colab, rồi chạy:
    !python colab_setup.py --dataset_source drive --drive_path "/content/drive/MyDrive/MMFi_Dataset"

Hoặc download trực tiếp:
    !python colab_setup.py --dataset_source download --output /content/MMFi_Dataset
"""

import os
import sys
import argparse
import shutil
import subprocess


def check_gpu():
    """Check GPU availability and specs."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"✅ GPU: {name} ({mem:.1f} GB)")
            return True
        else:
            print("❌ No GPU available!")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_disk():
    """Check disk space."""
    total, used, free = shutil.disk_usage("/content")
    print(f"📁 Colab Disk: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    
    if os.path.exists("/content/drive"):
        total_d, used_d, free_d = shutil.disk_usage("/content/drive")
        print(f"📁 Google Drive: {free_d/1e9:.1f} GB free / {total_d/1e9:.1f} GB total")
    
    return free


def install_deps():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "einops", "timm", "tensorboard", "scipy", "scikit-learn",
        "h5py", "pyyaml", "opencv-python-headless", "gdown"
    ], check=True)
    print("✅ Dependencies installed")


def setup_dataset_from_drive(drive_path):
    """Use dataset from Google Drive mount."""
    if not os.path.exists(drive_path):
        print(f"❌ Path not found: {drive_path}")
        print("   Make sure Google Drive is mounted and path is correct")
        print("   Try: from google.colab import drive; drive.mount('/content/drive')")
        return None
    
    # Check for E01-E04
    found = []
    for env in ['E01', 'E02', 'E03', 'E04']:
        env_path = os.path.join(drive_path, env)
        if os.path.isdir(env_path):
            subjects = [d for d in os.listdir(env_path) if d.startswith('S')]
            print(f"  ✅ {env}: {len(subjects)} subjects")
            found.append(env)
        else:
            print(f"  ❌ {env}: NOT FOUND")
    
    if len(found) == 4:
        print(f"\n✅ Dataset ready at: {drive_path}")
        return drive_path
    else:
        print(f"\n⚠️ Only found {len(found)}/4 environments")
        return drive_path if found else None


def download_dataset(output_dir):
    """Download dataset from Google Drive splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        "E01.zip": "1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B",
        "E02.zip": "1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o",
        "E03.zip": "1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq",
        "E04.zip": "1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1",
    }
    
    import gdown
    
    for name, fid in splits.items():
        zip_path = os.path.join("/content", name)
        env_name = name.replace('.zip', '')
        env_dir = os.path.join(output_dir, env_name)
        
        if os.path.isdir(env_dir) and os.listdir(env_dir):
            print(f"  ⏩ {name}: already extracted")
            continue
        
        if not os.path.exists(zip_path):
            print(f"  ⬇️ Downloading {name} (~20GB)...")
            try:
                gdown.download(id=fid, output=zip_path, quiet=False)
            except Exception as e:
                print(f"  ❌ Download failed: {e}")
                print(f"     Try manual download from Baidu: https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316")
                continue
        
        print(f"  📦 Extracting {name}...")
        subprocess.run(["unzip", "-q", zip_path, "-d", output_dir], check=True)
        # Clean up zip to save space
        os.remove(zip_path)
        print(f"  ✅ {name} extracted & zip removed")
    
    return output_dir


def update_config(code_dir, dataset_root):
    """Update YAML config with dataset path."""
    import yaml
    
    config_files = [
        os.path.join(code_dir, "config", "mmfi", f)
        for f in os.listdir(os.path.join(code_dir, "config", "mmfi"))
        if f.endswith('.yaml')
    ]
    
    for cf in config_files:
        with open(cf, 'r') as f:
            config = yaml.safe_load(f)
        config['dataset_root'] = dataset_root
        with open(cf, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"  ✅ Updated: {os.path.basename(cf)}")


def main():
    parser = argparse.ArgumentParser(description="ViPA-Fi Colab Setup")
    parser.add_argument("--dataset_source", choices=["drive", "download"], default="drive",
                        help="'drive' = use mounted Google Drive, 'download' = download from GDrive")
    parser.add_argument("--drive_path", type=str, default="/content/drive/MyDrive/MMFi_Dataset",
                        help="Path to dataset on Google Drive (for --dataset_source drive)")
    parser.add_argument("--output", type=str, default="/content/MMFi_Dataset",
                        help="Output path for downloaded dataset")
    parser.add_argument("--code_dir", type=str, default="/content/ViPAFi",
                        help="Path to ViPA-Fi code directory")
    parser.add_argument("--skip_deps", action="store_true", help="Skip installing dependencies")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 ViPA-Fi Colab Setup")
    print("=" * 60)
    
    # 1. Check GPU
    print("\n--- GPU Check ---")
    check_gpu()
    
    # 2. Check disk
    print("\n--- Disk Check ---")
    free_space = check_disk()
    
    # 3. Install dependencies
    if not args.skip_deps:
        install_deps()
    
    # 4. Setup dataset
    print("\n--- Dataset Setup ---")
    if args.dataset_source == "drive":
        dataset_root = setup_dataset_from_drive(args.drive_path)
    else:
        dataset_root = download_dataset(args.output)
    
    if dataset_root is None:
        print("\n❌ Dataset setup failed!")
        sys.exit(1)
    
    # 5. Update config
    print("\n--- Config Update ---")
    if os.path.isdir(args.code_dir):
        update_config(args.code_dir, dataset_root)
    else:
        print(f"⚠️ Code dir not found: {args.code_dir}")
        print(f"   Upload code first, then run: python colab_setup.py again")
    
    # 6. Print training command
    print("\n" + "=" * 60)
    print("✅ Setup complete! Run training with:")
    print("=" * 60)
    print(f"""
cd {args.code_dir}

# For T4 (16GB):
python train.py \\
    --config_file config/mmfi/pose_config_p1s1.yaml \\
    --experiment_name vipafi \\
    --batch_size 64 \\
    --max_device_batch_size 32 \\
    --num_workers 2 \\
    --total_epoch 50

# For A100 (40GB):
python train.py \\
    --config_file config/mmfi/pose_config_p1s1.yaml \\
    --experiment_name vipafi \\
    --batch_size 256 \\
    --max_device_batch_size 128 \\
    --num_workers 4 \\
    --total_epoch 50
""")


if __name__ == "__main__":
    main()
