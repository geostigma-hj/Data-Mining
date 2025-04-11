import pandas as pd
import pyarrow.parquet as pq
import os
import time

def calculate_duplicate_ratio():
    try:
        # 读取并合并所有parquet文件
        parquet_files = [f for f in os.listdir('10G_data') if f.endswith('.parquet')]
        dfs = []
        print(f"开始处理 {len(parquet_files)} 个数据文件...")
        
        for file in parquet_files:
            df = pq.ParquetFile(os.path.join('10G_data', file)).read().to_pandas()
            dfs.append(df)
        # 合并所有DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if 'is_active' in combined_df.columns:
            active_count = df['is_active'].sum()
            print(f"\n[活跃用户统计] is_active=True 的记录数: {active_count} (占比: {active_count/len(df):.2%})")

        # 计算重复ID
        total_ids = len(combined_df)
        duplicate_count = combined_df['id'].duplicated().sum()
        duplicate_ratio = duplicate_count / total_ids
        return duplicate_count, total_ids, duplicate_ratio
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    result = calculate_duplicate_ratio()
    if result:
        dup_count, total, ratio = result
        print("\n=== 合并后重复ID统计 ===")
        print(f"总ID数量: {total}")
        print(f"重复ID数量: {dup_count}")
        print(f"重复比例: {ratio:.2%}")
    end_time = time.time()
    print(f"\n程序执行时间: {end_time - start_time:.2f}秒")