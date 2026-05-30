import os
import csv
import argparse
import shutil
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Copy specific video files based on a CSV list.")
    parser.add_argument(
        "-i", "--input_csv", type=str, 
        default="openhumanvid_06hvtss.csv",
        help="Path to the input CSV file containing filtered video names."
    )
    parser.add_argument(
        "-s", "--source_dir", type=str, 
        default="../../../../../../../../datasets/openhumanvid/video_10s_25fps",
        help="Directory containing the source mp4 files."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, 
        default="../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss/video_720p",
        help="Directory to save the copied mp4 files."
    )
    
    args = parser.parse_args()

    # 1. 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 读取 CSV 文件，获取需要提取的视频文件名集合
    print(f"Reading target videos from CSV: {args.input_csv}")
    target_video_names = set()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: CSV file '{args.input_csv}' not found.")
        return

    with open(args.input_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'video_name' in row:
                # 提取纯文件名 (例如: 012bf495a4e9d61aa78f441afb4c5753.mp4)
                basename = os.path.basename(row['video_name'])
                target_video_names.add(basename)
                
    print(f"Found {len(target_video_names)} target videos in CSV.")

    # 3. 开始查找并复制文件
    copied_count = 0
    missing_files = []

    # 使用 tqdm 显示进度条
    for video_name in tqdm(target_video_names, desc="Copying videos"):
        source_path = os.path.join(args.source_dir, video_name)
        target_path = os.path.join(args.output_dir, video_name)

        # 检查源文件是否存在
        if os.path.exists(source_path):
            try:
                # 只有当目标文件不存在，或者大小不一致时才复制，避免重复复制浪费时间
                if not os.path.exists(target_path) or os.path.getsize(source_path) != os.path.getsize(target_path):
                    # shutil.copy2 会保留文件的元数据（如创建时间、修改时间等）
                    shutil.copy2(source_path, target_path)
                copied_count += 1
            except Exception as e:
                tqdm.write(f"Error copying {video_name}: {e}")
        else:
            missing_files.append(video_name)

    # 4. 打印最终结果
    print(f"\nDone! Successfully copied {copied_count} video files to {args.output_dir}.")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} videos were not found in the source directory.")
        # 如果缺失的文件不多，可以打印出来看看是哪些
        if len(missing_files) <= 10:
            for missing in missing_files:
                print(f"  - Missing: {missing}")
        else:
            print(f"  - (First 10 missing files): {missing_files[:10]}")

if __name__ == "__main__":
    main()