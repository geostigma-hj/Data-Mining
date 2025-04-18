import pandas as pd
import pyarrow.parquet as pq
import os
import time
import argparse

def calculate_duplicate_ratio(args):
    try:
        # 读取并合并所有parquet文件
        parquet_files = [f for f in os.listdir(f'{args.size}_data_new') if f.endswith('.parquet')]
        dfs = []
        print(f"开始处理 {len(parquet_files)} 个数据文件...")
        
        for file in parquet_files:
            df = pq.ParquetFile(os.path.join(f'{args.size}_data_new', file)).read().to_pandas()
            dfs.append(df)
        # 合并所有DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # 计算重复ID：貌似完全不重复，牛逼，下手真的黑
        total_ids = len(combined_df)
        duplicate_count = combined_df['id'].duplicated().sum()
        duplicate_ratio = duplicate_count / total_ids
        print(f"\nid 重复比例: {duplicate_ratio:.2%}")

        combined_df['user_id'] = combined_df['user_name'] + '_' + combined_df['phone_number'] # 看看 user_id 有没有重复的
        duplicate_count = combined_df['user_id'].duplicated().sum()
        duplicate_ratio = duplicate_count / len(combined_df)
        print(f"\nuser_id 重复比例: {duplicate_ratio:.2%}")

        combined_df = combined_df.drop(columns=['id', "last_login", 'user_id']) # 看看去除 id 和 last_login 之后是否有重复数据
        temp = combined_df.drop_duplicates()
        print(f"\n重复数据比例: {(len(combined_df)-len(temp))/len(combined_df):.2%}")
        
        if 'is_active' in combined_df.columns:
            active_count = combined_df['is_active'].sum()
            print(f"\n[活跃用户统计] is_active=True 的记录数: {active_count} (占比: {active_count/len(combined_df):.2%})")

    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True, default="10G",
                       help='指定要处理的数据集大小 (10G 或 30G)')
    args = parser.parse_args()
    start_time = time.time()
    result = calculate_duplicate_ratio(args)
    end_time = time.time()
    print(f"\n程序执行时间: {end_time - start_time:.2f}秒")