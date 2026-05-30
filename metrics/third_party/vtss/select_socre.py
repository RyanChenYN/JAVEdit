import csv
import argparse
import os

def filter_csv(input_csv, output_csv, threshold):
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' does not exist.")
        return

    print(f"Reading from: {input_csv}")
    print(f"Filtering scores >= {threshold}")

    filtered_count = 0
    total_count = 0

    # 读取输入文件并写入输出文件
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        
        # 确保输入CSV包含我们需要的列
        if 'video_name' not in reader.fieldnames or 'score' not in reader.fieldnames:
            print("Error: Input CSV must contain 'video_name' and 'score' columns.")
            return

        # 设置输出CSV的表头
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        # 遍历每一行进行筛选
        for row in reader:
            total_count += 1
            try:
                score = float(row['score'])
                if score >= threshold:
                    writer.writerow(row)
                    filtered_count += 1
            except ValueError:
                print(f"Warning: Could not parse score for video {row.get('video_name', 'UNKNOWN')}. Skipping.")

    print(f"Done! Saved {filtered_count} out of {total_count} videos to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Filter videos by score from a CSV file.")
    parser.add_argument(
        "-i", "--input", type=str, default="openhumanvid.csv",
        help="Path to the input CSV file (e.g., scores.csv)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="openhumanvid_06hvtss.csv",
        help="Path to the output CSV file (default: openhumanvid_06hvtss.csv)"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.06,
        help="Score threshold for filtering (default: 0.06)"
    )

    args = parser.parse_args()

    filter_csv(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()