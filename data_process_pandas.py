import pyarrow.parquet as pq
import pandas as pd
import time
import json
import numpy as np
from datetime import datetime
import warnings
from collections import Counter
import os
import argparse

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 增强型JSON解析函数
def safe_json_parse(ph_str):
    try:
        # 处理可能的单引号问题
        if isinstance(ph_str, str):
            ph_str = ph_str.replace("'", '"')
            # 处理可能的 NaN 或空值
            if ph_str.strip() in ['', 'nan', 'None']:
                return None
            return json.loads(ph_str)
        return ph_str if not pd.isna(ph_str) else None
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"解析错误: {e}，原始数据: {ph_str}")
        return None

# 优化后的购买记录处理
def process_purchase_history(df):
    parse_errors = 0
    multi_category_count = 0  # 新增：多类别购买计数器
    parsed_data = df.apply(safe_json_parse)
    
    results = []
    for data in parsed_data:
        if not data:
            results.append({
                'total_spent': 0,
                'avg_transaction': 0,
                'main_category': None,
                'items_count': 0,
                'items_list': [],
                'category_distribution': json.dumps({}),
                'book_spend': 0,
                'food_spend': 0,
                'home_spend': 0,
                'cloth_spend': 0,
                'electric_spend': 0
            })
            parse_errors += 1
            continue
        
        items = data.get('items', [])
        # 新增：统计多类别购买
        categories = set(item.get('category') for item in items if item.get('category'))
        if len(categories) >= 2:
            multi_category_count += 1
            
        items_count = len(items)
        avg_price = data.get('average_price', 0)
        category = data.get('category', None)

        category_spend = {
            '书籍': 0,
            '食品': 0,
            '家居': 0,
            '服装': 0,
            '电子产品': 0
        }

        category_spend[category] += avg_price * items_count
            
        # 计算品类分布
        category_dist = {category: items_count}
        results.append({
            'total_spent': avg_price * items_count,
            'avg_spent_per_trans': avg_price,
            'main_category': category,
            'items_count': items_count,
            'items_list': [item['id'] for item in items],
            'category_distribution': json.dumps(category_dist),
            'book_spend': category_spend['书籍'],
            'food_spend': category_spend['食品'],
            'home_spend': category_spend['家居'],
            'cloth_spend': category_spend['服装'],
            'electric_spend': category_spend['电子产品']
        })
    
    result_df = pd.DataFrame(results)
    print(result_df.iloc[0])
    result_df._parse_errors = parse_errors
    result_df._multi_category_count = multi_category_count  # 新增：附加多类别统计
    result_df.drop(columns=['items_list'], inplace=True)  # 删除 items_list 列
    return result_df

# 替换原有的数据加载部分
def load_and_preprocess(df):
    df = df.drop(columns=["id", "email", "is_active", "chinese_address"]) # 大量重复且对分析过程没有帮助，直接删掉
    
    purchase_history = df['purchase_history'].copy() # 提前记录下来用于后续处理
    df.drop(columns=['purchase_history'], inplace=True)  # 删掉防止 cudf 超限
    mask = pd.Series(True, index=df.index)  # 初始全True掩码

    original_count = len(df)
    print(f"原始记录数: {original_count}")
    
    # 1. 缺失值统计 - 修复这部分代码
    print("缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_stats['缺失比例'] = (missing_stats / len(df)).apply(lambda x: f"{x:.2%}")
    print(missing_stats)  # 转换为 pandas 格式输出

    # 验证日期格式
    if 'registration_date' in df.columns:
        # 修改日期类型检查方式
        if not np.issubdtype(df['registration_date'].dtype, np.datetime64):
            df['registration_date'] = pd.to_datetime(df['registration_date'])

    # 验证时序逻辑
    if 'registration_date' in df.columns and 'timestamp' in df.columns:
        # 更精确的时间戳解析（处理ISO 8601格式）
        df['timestamp'] = pd.to_datetime(df['timestamp'].str.extract(r'([^+]+)')[0])
        time_logic_violation = len(df[df['registration_date'] > df['timestamp']])
        print(f"\n[时序逻辑过滤] 注册时间晚于首次消费的记录数: {time_logic_violation} (占比: {time_logic_violation/original_count:.2%})")
        # 确保注册时间早于第一次消费时间
        time_mask = (df['registration_date'] <= df['timestamp'])
        mask = mask & time_mask  # 更新掩码
        # df = df[time_mask]

    # 处理异常值
    pre_filter_count = len(df[time_mask])
    age_mask = df['age'].between(18, 100)
    income_mask = (df['income'] > 0)
    credit_mask = (df['credit_score'] > 0)
    mask = mask & age_mask & income_mask & credit_mask  # 更新掩码
    # df = df[mask]
    post_filter_count = len(df[mask])
    print(f"\n[异常值过滤] 删除记录数: {pre_filter_count - post_filter_count} (占比: {(pre_filter_count - post_filter_count)/pre_filter_count:.2%})")
    print(f"剩余有效记录数: {len(df[mask])} (保留比例: {len(df[mask])/original_count:.2%})")

    # 处理 purchase_history
    purchase_history = purchase_history[mask]
    df = df[mask]
    purchase_features = process_purchase_history(purchase_history)
    # 新增：输出多类别购买统计
    print(f"\n[多类别购买统计] 单次购买两类及以上记录数: {purchase_features._multi_category_count} (占比: {purchase_features._multi_category_count/len(df):.2%})")
    parse_errors = purchase_features._parse_errors  # 获取解析错误数
    print(f"\n[JSON解析统计] 解析失败记录数: {parse_errors} (占比: {parse_errors/len(df):.2%})")
    # df = cudf.concat([df, cudf.DataFrame.from_pandas(purchase_features)], axis=1)
    # 过滤掉没有购物记录或者记录为空的用户
    purchase_features.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    valid_mask = purchase_features['items_count'] > 0
    valid_indices = purchase_features[valid_mask].index
    df = df.loc[valid_indices]
    purchase_features = purchase_features.loc[valid_indices]
    print(f"\n[购物记录过滤] 有效购物记录数: {len(df)} (占比: {len(df)/original_count:.2%})")
    
    # 合并特征
    df = pd.concat([df, purchase_features], axis=1)
    
    # 定义唯一用户id
    df['user_id'] = df['chinese_name'] + '_' + df['phone_number']
    
    # 聚合函数
    grouped = df.groupby('user_id').agg({
        'timestamp': ['count', 'nunique', 'max'],
        'total_spent': 'sum',
        'avg_spent_per_trans': 'mean',
        'items_count': 'sum',
        'age': 'first',
        'income': 'first',
        'gender': 'first',
        'country': 'first',
        'registration_date': 'first',
        'credit_score': 'mean',
        'book_spend': 'sum',
        'food_spend': 'sum', 
        'home_spend': 'sum',
        'cloth_spend': 'sum',
        'electric_spend': 'sum'
    })
    
    # 单独处理 category_distribution (转换为 pandas 处理)
    temp_df = df[['user_id', 'category_distribution']]
    category_dist = temp_df.groupby('user_id')['category_distribution'].apply(
        lambda x: json.dumps(
            dict(sum(
                (Counter(json.loads(str(item))) if not pd.isna(item) else Counter() for item in x),
                Counter()
            ))
        )
    )
    # 合并结果
    grouped['category_distribution'] = category_dist

    # 列名处理和重命名
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.rename(columns={
        'timestamp_count': 'purchase_count',
        'total_spent_sum': 'total_spent',
        'avg_spent_per_trans_mean': 'avg_transaction',
        'items_count_sum': 'total_items',
        'timestamp_max': 'last_purchase', 
        'category_distribution_<lambda>': 'category_distribution',
        'book_spend_sum': 'book_spend',
        'food_spend_sum': 'food_spend',
        'home_spend_sum': 'home_spend',
        'cloth_spend_sum': 'cloth_spend',
        'electric_spend_sum': 'electric_spend'
    })
    
    # 计算衍生特征
    df_final = grouped.reset_index()
    df_final.rename(columns=lambda col: col[:-6] if col.endswith('_first') else col, inplace=True) # 去掉 first 后缀
    df_final = df_final[df_final['purchase_count'] > 0] # 二次过滤

    current_time = datetime.now()
    df_final['recency_days'] = (current_time - df_final['last_purchase']).dt.days
    df_final['membership_days'] = (current_time - df_final['registration_date']).dt.days
    df_final['purchase_freq'] = df_final['purchase_count'] / ((df_final['membership_days']/30) + 1)  # 月均购买次数

    # 删除冗余列
    return df_final.drop(columns=['user_name', 'phone_number'], errors='ignore')

# 主程序改造
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True, default='10G',
                       help='指定要处理的数据集大小 (10G 或 30G)')
    args = parser.parse_args()

    start_time = time.time()
    if args.size == '10G':
        parquet_files = [f for f in os.listdir('10G_data') if f.endswith('.parquet')]
    else:
        parquet_files = [f for f in os.listdir('30G_data') if f.endswith('.parquet')]

    for i, file in enumerate(parquet_files):
        print(f"\nProcessing part {i+1}")
        df = pq.ParquetFile(os.path.join(f"{args.size}_data", file)).read().to_pandas()
        df = load_and_preprocess(df)
        df.to_csv(f'pandas_test_time.csv', index=False)
        print(f"Part {i+1} Done! 处理行数: {len(df)}")
        single_timestamp_count = len(df[df['timestamp_nunique'] == 1])
        print(f"\n[时间戳分析] 单时间戳用户数: {single_timestamp_count} (占比: {single_timestamp_count/len(df):.2%})")
        break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n程序总执行时间: {elapsed_time:.2f}秒")