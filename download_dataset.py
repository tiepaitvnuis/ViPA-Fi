"""
Script tải MM-Fi dataset và giải nén tự động.

Hướng dẫn sử dụng:
─────────────────
Cách 1 (Google Drive - nếu hết rate limit thì dùng Cách 2):
    python download_dataset.py --method gdrive --output E:\MMFi_dataset

Cách 2 (Tải thủ công từ Baidu Netdisk):
    1. Mở link: https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316
    2. Tải 4 file: E01.zip, E02.zip, E03.zip, E04.zip
    3. Đặt vào E:\MMFi_dataset\
    4. Chạy: python download_dataset.py --method local --output E:\MMFi_dataset

Cách 3 (Copy Google Drive vào account cá nhân rồi tải):
    1. Mở: https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz
    2. Right-click từng file E01-E04.zip -> "Make a copy"
    3. Tải các bản copy từ Google Drive cá nhân vào E:\MMFi_dataset\
    4. Chạy: python download_dataset.py --method local --output E:\MMFi_dataset
"""

import os
import sys
import argparse
import zipfile
import glob


GDRIVE_FILE_IDS = {
    "E01.zip": "1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B",
    "E02.zip": "1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o",
    "E03.zip": "1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq",
    "E04.zip": "1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1",
}

BAIDU_LINK = "https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316"
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz"


def download_gdrive(output_dir):
    """Download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown --quiet")
        import gdown

    os.makedirs(output_dir, exist_ok=True)

    for filename, file_id in GDRIVE_FILE_IDS.items():
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            print(f"[SKIP] {filename} already exists.")
            continue
        print(f"[DOWNLOAD] {filename} ...")
        try:
            gdown.download(id=file_id, output=output_path, quiet=False)
            print(f"[OK] {filename}")
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")
            print(f"\n  Google Drive rate limit detected.")
            print(f"  Please use --method local and download manually:")
            print(f"  Baidu Netdisk: {BAIDU_LINK}")
            print(f"  Google Drive:  {GDRIVE_FOLDER}")
            return False
    return True


def extract_all(output_dir):
    """Extract all zip files in the output directory."""
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    zip_files = sorted(glob.glob(os.path.join(output_dir, "E*.zip")))
    if not zip_files:
        print(f"[ERROR] No E*.zip files found in {output_dir}")
        print(f"  Please download E01.zip - E04.zip into {output_dir}")
        return False

    for zf in zip_files:
        name = os.path.basename(zf)
        env_name = name.replace(".zip", "")
        env_dir = os.path.join(dataset_dir, env_name)
        if os.path.exists(env_dir) and os.listdir(env_dir):
            print(f"[SKIP] {name} already extracted to {env_dir}")
            continue
        print(f"[EXTRACT] {name} -> {dataset_dir} ...")
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(dataset_dir)
            print(f"[OK] {name} extracted.")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            return False

    return True


def verify_dataset(output_dir):
    """Verify the dataset structure is correct."""
    dataset_dir = os.path.join(output_dir, "dataset")
    expected_envs = ["E01", "E02", "E03", "E04"]
    missing = []

    for env in expected_envs:
        env_path = os.path.join(dataset_dir, env)
        if not os.path.isdir(env_path):
            missing.append(env)
            continue
        # Check for at least one subject
        subjects = [d for d in os.listdir(env_path) if d.startswith("S") and os.path.isdir(os.path.join(env_path, d))]
        if not subjects:
            missing.append(f"{env} (no subjects)")

    if missing:
        print(f"[WARNING] Missing or incomplete: {missing}")
        return False

    print(f"[OK] Dataset verified at {dataset_dir}")
    print(f"  Environments: {expected_envs}")

    # Count total subject-action pairs
    total = 0
    for env in expected_envs:
        env_path = os.path.join(dataset_dir, env)
        for subj in sorted(os.listdir(env_path)):
            subj_path = os.path.join(env_path, subj)
            if os.path.isdir(subj_path):
                actions = [d for d in os.listdir(subj_path) if d.startswith("A")]
                total += len(actions)
    print(f"  Total subject-action pairs: {total}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and setup MM-Fi dataset")
    parser.add_argument("--method", choices=["gdrive", "local"], default="local",
                        help="'gdrive' to download via gdown, 'local' to just extract existing zips")
    parser.add_argument("--output", type=str, default=r"E:\MMFi_dataset",
                        help="Output directory for the dataset")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction step")
    args = parser.parse_args()

    print("=" * 60)
    print("MM-Fi Dataset Setup")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Method: {args.method}")
    print()

    if args.method == "gdrive":
        success = download_gdrive(args.output)
        if not success:
            print("\n[INFO] Download failed. You can:")
            print(f"  1. Open Baidu Netdisk: {BAIDU_LINK}")
            print(f"  2. Download E01-E04.zip manually to {args.output}")
            print(f"  3. Re-run: python download_dataset.py --method local --output {args.output}")
            sys.exit(1)

    if not args.skip_extract:
        print("\n--- Extracting ---")
        success = extract_all(args.output)
        if not success:
            sys.exit(1)

    print("\n--- Verifying ---")
    verify_dataset(args.output)

    dataset_path = os.path.join(args.output, "dataset")
    print(f"\n{'=' * 60}")
    print(f"Dataset ready at: {dataset_path}")
    print(f"Update your config YAML with:")
    print(f"  dataset_root: {dataset_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
