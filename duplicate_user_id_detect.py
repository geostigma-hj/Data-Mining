import pandas as pd
import argparse

def find_duplicate_userids(file1, file2, id_column='user_id'):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    ids1 = set(df1[id_column].unique())
    ids2 = set(df2[id_column].unique())
    
    duplicates = ids1 & ids2
    
    print(f"\n{file1} 中 user_id 数量: {len(ids1)}")
    print(f"{file2} 中 user_id 数量: {len(ids2)}")
    print(f"重复的 user_id 数量: {len(duplicates)}")
    
    if duplicates:
        print("重复的 user_id 示例(前10个):")
        for i, uid in enumerate(list(duplicates)[:10]):
            print(f"{i+1}. {uid}")
    
    return list(duplicates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True, default='10G',
                       help='指定要处理的数据集大小 (10G 或 30G)')
    args = parser.parse_args()
    if args.size == '10G':
        length = 8
        files = [f'processed_csv_10G/data_part_{i}.csv' for i in range(length)]
    else:
        length = 16
        files = [f'processed_csv_30G/data_part_{i}.csv' for i in range(length)]

    for i in range(length):
        for j in range(i+1, length):
            duplicates = find_duplicate_userids(files[i], files[j])
        
            if duplicates:
                print("发现重复的 user_id!")
            else:
                print("未发现重复的 user_id")