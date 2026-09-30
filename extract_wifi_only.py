"""
Script trích xuất chỉ wifi-csi + ground_truth từ MM-Fi dataset.
Giảm từ 77GB → ~17GB, đủ để upload lên Google Drive 15GB free hoặc Colab disk.

Cách dùng:
─────────
Bước 1: Chạy script này trên máy local (đã có dataset đầy đủ):
    python extract_wifi_only.py --input "E:\\MMFi_Dataset\\MMFi_Dataset" --output "E:\\MMFi_WiFi_Only"

Bước 2: Nén thành zip:
    python extract_wifi_only.py --input "E:\\MMFi_Dataset\\MMFi_Dataset" --output "E:\\MMFi_WiFi_Only" --zip

Bước 3: Upload file zip lên Google Drive (hoặc trực tiếp lên Colab)

Bước 4: Trên Colab, giải nén và chạy train bình thường
"""

import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path


def extract_wifi_only(input_root, output_root, verbose=True):
    """Copy only wifi-csi folders and ground_truth.npy from full dataset."""
    
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    if not input_root.exists():
        print(f"❌ Input path not found: {input_root}")
        sys.exit(1)
    
    total_files = 0
    total_bytes = 0
    skipped_bytes = 0
    
    environments = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.startswith('E')])
    
    if not environments:
        print(f"❌ No E01-E04 folders found in {input_root}")
        sys.exit(1)
    
    print(f"📁 Input: {input_root}")
    print(f"📁 Output: {output_root}")
    print(f"🔍 Found environments: {[e.name for e in environments]}")
    print()
    
    for env_dir in environments:
        try:
            subjects = sorted([d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
        except OSError as e:
            print(f"⚠️ Skipping environment {env_dir.name} due to disk error: {e}")
            continue
        
        for subj_dir in subjects:
            try:
                actions = sorted([d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith('A')])
            except OSError as e:
                print(f"⚠️ Skipping subject {subj_dir.name} due to disk error: {e}")
                continue
            
            for act_dir in actions:
                # === Copy wifi-csi folder ===
                wifi_src = act_dir / 'wifi-csi'
                wifi_dst = output_root / env_dir.name / subj_dir.name / act_dir.name / 'wifi-csi'
                
                try:
                    if wifi_src.is_dir():
                        if not wifi_dst.exists():
                            wifi_dst.mkdir(parents=True, exist_ok=True)
                            for f in sorted(wifi_src.iterdir()):
                                if f.is_file():
                                    try:
                                        shutil.copy2(f, wifi_dst / f.name)
                                        total_files += 1
                                        total_bytes += f.stat().st_size
                                    except OSError as e:
                                        print(f"⚠️ Error copying {f.name}: {e}")
                        else:
                            # Already copied
                            for f in wifi_dst.iterdir():
                                try:
                                    total_files += 1
                                    total_bytes += f.stat().st_size
                                except OSError:
                                    pass
                except OSError as e:
                    print(f"⚠️ Error accessing {wifi_src}: {e}")
                
                # === Copy ground_truth.npy ===
                gt_src = act_dir / 'ground_truth.npy'
                gt_dst = output_root / env_dir.name / subj_dir.name / act_dir.name / 'ground_truth.npy'
                
                try:
                    if gt_src.is_file():
                        gt_dst.parent.mkdir(parents=True, exist_ok=True)
                        if not gt_dst.exists():
                            try:
                                shutil.copy2(gt_src, gt_dst)
                            except OSError as e:
                                print(f"⚠️ Error copying GT {gt_src.name}: {e}")
                        total_files += 1
                        total_bytes += gt_src.stat().st_size
                except OSError as e:
                    print(f"⚠️ Error accessing GT {gt_src}: {e}")
                
                # === Count skipped data ===
                try:
                    for item in act_dir.iterdir():
                        if item.name not in ['wifi-csi', 'ground_truth.npy']:
                            try:
                                if item.is_dir():
                                    for f in item.rglob('*'):
                                        try:
                                            if f.is_file():
                                                skipped_bytes += f.stat().st_size
                                        except OSError:
                                            pass
                                elif item.is_file():
                                    skipped_bytes += item.stat().st_size
                            except OSError:
                                pass
                except OSError:
                    pass
            
            if verbose:
                print(f"  ✅ {env_dir.name}/{subj_dir.name}: {len(actions)} actions copied")
    
    print()
    print("=" * 60)
    print(f"📊 Extraction Summary")
    print("=" * 60)
    print(f"  Files copied:   {total_files:,}")
    print(f"  WiFi+GT size:   {total_bytes / 1e9:.2f} GB")
    print(f"  Skipped:        {skipped_bytes / 1e9:.2f} GB")
    print(f"  Reduction:      {(1 - total_bytes / (total_bytes + skipped_bytes)) * 100:.1f}%")
    print(f"  Output:         {output_root}")
    
    return total_bytes


def create_zip(output_root, zip_path=None):
    """Create a zip file from the extracted dataset."""
    output_root = Path(output_root)
    if zip_path is None:
        zip_path = output_root.parent / f"{output_root.name}.zip"
    
    print(f"\n📦 Creating zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        file_count = 0
        for f in sorted(output_root.rglob('*')):
            if f.is_file():
                arcname = f.relative_to(output_root.parent)
                zf.write(f, arcname)
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"  ... {file_count} files added")
    
    zip_size = os.path.getsize(zip_path)
    print(f"✅ Zip created: {zip_path}")
    print(f"   Size: {zip_size / 1e9:.2f} GB")
    print(f"   Upload this file to Google Drive or Colab")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract only wifi-csi + ground_truth from MM-Fi dataset (77GB → ~17GB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract wifi-csi only
  python extract_wifi_only.py --input "E:\\MMFi_Dataset\\MMFi_Dataset" --output "E:\\MMFi_WiFi_Only"

  # Extract + create zip for upload
  python extract_wifi_only.py --input "E:\\MMFi_Dataset\\MMFi_Dataset" --output "E:\\MMFi_WiFi_Only" --zip
  
  # Then upload MMFi_WiFi_Only.zip to Google Drive and use on Colab
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to full MM-Fi dataset (containing E01-E04)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for wifi-only dataset")
    parser.add_argument("--zip", action="store_true",
                        help="Also create a zip file for easy upload")
    parser.add_argument("--zip_path", type=str, default=None,
                        help="Custom zip output path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 MM-Fi WiFi-CSI Only Extractor")
    print("   Reduces dataset from ~77GB to ~17GB")
    print("=" * 60)
    
    extract_wifi_only(args.input, args.output)
    
    if args.zip:
        create_zip(args.output, args.zip_path)
    
    print(f"\n💡 On Colab, update your config:")
    print(f'   dataset_root: "/content/drive/MyDrive/MMFi_WiFi_Only"')
    print(f"   (or wherever you extract the zip)")


if __name__ == "__main__":
    main()
