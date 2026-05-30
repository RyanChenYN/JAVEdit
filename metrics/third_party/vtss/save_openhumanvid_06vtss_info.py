import os
import json
import csv
import argparse
import glob
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract specific video info to individual JSON files based on CSV.")
    parser.add_argument(
        "-i", "--input_csv", type=str, 
        default="openhumanvid_06hvtss.csv",
        help="Path to the input CSV file containing filtered video names."
    )
    parser.add_argument(
        "-s", "--source_info_dir", type=str, 
        default="../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid_info",
        help="Directory containing the source JSON files with video info."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, 
        default="../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss/info/",
        help="Directory to save the individual JSON files."
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
                # 无论 CSV 里存的是绝对路径还是仅文件名，我们都提取出纯文件名 (例如: xxx.mp4)
                basename = os.path.basename(row['video_name'])
                target_video_names.add(basename)
                
    print(f"Found {len(target_video_names)} target videos in CSV.")

    # 3. 查找源目录下的所有 JSON 文件
    source_json_files = glob.glob(os.path.join(args.source_info_dir, "*.json"))
    print(f"Found {len(source_json_files)} JSON files in source directory.")

    # 4. 遍历源 JSON 文件，匹配并保存
    matched_count = 0
    
    for json_file in tqdm(source_json_files, desc="Processing source JSONs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                
            # 确保读取到的是一个列表
            if not isinstance(data_list, list):
                continue
                
            for item in data_list:
                if "path" in item:
                    # 提取 JSON 中 path 字段的纯文件名 (例如: xxx.mp4)
                    item_basename = os.path.basename(item["path"])
                    
                    # 如果这个视频在我们需要的集合里
                    if item_basename in target_video_names:
                        # 去掉 .mp4 后缀，作为新的 json 文件名
                        file_name_without_ext = os.path.splitext(item_basename)[0]
                        output_json_path = os.path.join(args.output_dir, f"{file_name_without_ext}.json")
                        
                        # 将这个字典单独保存为一个 JSON 文件
                        with open(output_json_path, 'w', encoding='utf-8') as out_f:
                            # ensure_ascii=False 保证中文字符正常显示，indent=4 保证格式美观
                            json.dump(item, out_f, ensure_ascii=False, indent=4)
                            
                        matched_count += 1
                        
                        # 可选优化：如果每个视频只出现一次，可以从集合中移除，加快后续匹配速度
                        target_video_names.remove(item_basename)
                        
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    print(f"Done! Successfully extracted and saved {matched_count} individual JSON files to {args.output_dir}.")
    
    # 检查是否有未找到的视频
    if matched_count < len(target_video_names):
        print(f"Warning: {len(target_video_names) - matched_count} videos from the CSV were not found in the source JSONs.")

if __name__ == "__main__":
    main()