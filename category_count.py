import pandas as pd
import json
from collections import Counter
import os
import pyarrow.parquet as pq

def count_categories(df):
    # 解析purchase_history列
    def parse_json(ph_str):
        try:
            if pd.isna(ph_str):
                return None
            return json.loads(ph_str.replace("'", '"'))
        except:
            return None
    
    parsed = df['purchase_history'].apply(parse_json)
    
    # 统计所有类别
    categories = []
    for data in parsed:
        if data and 'categories' in data.keys():
            categories.append(data['categories'])
    
    # 统计各类别数量
    counter = Counter(categories)
    return counter

def process_all_parquets(folder_path):
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    total_counter = Counter()
    
    for file in parquet_files:
        file_path = os.path.join(folder_path, file)
        df = pq.ParquetFile(file_path).read().to_pandas()
        file_counter = count_categories(df)
        total_counter.update(file_counter)
    
    # 计算总数量用于计算占比
    total_items = sum(total_counter.values())
    
    # 打印美观的统计结果
    print("\n=== 所有文件汇总统计 ===")
    print(f"总共有 {len(total_counter)} 种不同的物品类别\n")
    
    # 按6行×7列格式打印
    items = total_counter.most_common(42)  # 最多显示42种
    for i in range(0, len(items), 7):
        row = items[i:i+7]
        for category, count in row:
            percentage = (count / total_items) * 100
            print(f"{category:<15}: {count:<6} ({percentage:.1f}%)".ljust(25), end="")
        print()
    
    return total_counter

if __name__ == "__main__":
    folder_path = "10G_data_new"
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
    else:
        total_counter = process_all_parquets(folder_path)