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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
size = '10G'

def draw_bar_chart(df, size):
    # 使用与post_visulize.py一致的配色方案
    COLORS = {
        'primary': ['#8ECFC9', '#FFBE7A', '#FA7F6F'],
        'secondary': ['#B395BD', '#7DAEE0', '#EA8379'],
        'accent': ['#299D8F', '#E9C46A', '#D87659'],
        'gender': ['#456990', '#E487C0'],
        'mono': ['#333333', '#666666', '#999999']
    }

    # print("\n支付方式分布:")
    # print(df['payment_method'].value_counts(normalize=True))
    
    # print("\n支付状态分布:")
    # print(df['payment_status'].value_counts(normalize=True))

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 支付方式分布图（保持柱状图）
    payment_method_percent = df['payment_method'].value_counts(normalize=True) * 100
    sns.barplot(
        x=payment_method_percent.index,
        y=payment_method_percent.values,
        palette=COLORS['primary'],
        edgecolor='black',
        linewidth=0.5,
        ax=ax1
    )
    ax1.set_title('支付方式分布', pad=15)
    ax1.set_xlabel('支付方式', labelpad=10)
    ax1.set_ylabel('比例(%)', labelpad=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加百分比标签（顶部）
    for p, percent in zip(ax1.patches, payment_method_percent):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2.,
                height + 0.3,
                f'{percent:.1f}%',
                ha='center', va='bottom',
                fontsize=10)
    
    # 支付状态分布图（改为饼图）
    payment_status_percent = df['payment_status'].value_counts(normalize=True) * 100
    payment_status_percent.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=COLORS['secondary'],
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 10},
        ax=ax2
    )
    ax2.set_title('支付状态分布', pad=15)
    ax2.set_ylabel('')  # 清除默认的ylabel
    
    plt.tight_layout()
    if size == '10G':
        plt.savefig('result_imgs/payment_distribution.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('result_imgs_30G/payment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 删除这两列
    df = df.drop(columns=['payment_method', 'payment_status'])

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

# 处理登录记录，提取第一次登录时间和最后一次登录（原始文本的 last_login 直接删除，用从 login_history 提取的 last_login 代替）
# def process_login_history(df):
#     parse_errors = 0
#     parsed = df.apply(safe_json_parse)
#     login_features = []
#     for data in parsed:
#         if not data:
#             login_features.append({
#                 "first_login_history": None,
#                 "last_login_history": None,
#                 "login_count": 0, 
#                 "avg_session": 0
#             })
#             parse_errors += 1
#             continue
            
#         timestamps = data.get("timestamps", [])
#         # 处理时间戳，只取前10个字符（YYYY-MM-DD）
#         processed_timestamps = [ts[:10] for ts in timestamps if ts and len(ts) >= 10]
        
#         first_login = pd.to_datetime(min(processed_timestamps)) if processed_timestamps else None
#         last_login = pd.to_datetime(max(processed_timestamps)) if processed_timestamps else None

#         login_features.append({
#             "first_login_history": first_login,
#             "last_login_history": last_login,
#             "login_count": data.get("login_count", 0),
#             "avg_session": data.get("avg_session_duration", 0)
#         })
    
#     result_df = pd.DataFrame(login_features)
#     result_df._parse_errors = parse_errors
#     return result_df

def process_purchase_history(df):
    # 向量化处理JSON解析
    parsed_data = df.apply(safe_json_parse)
    valid_mask = parsed_data.notnull()
    parse_errors = len(df) - valid_mask.sum()
    
    # 批量处理类别映射
    category_distribute = {
        "电子数码": ["智能手机", "平板电脑", "笔记本电脑", "智能手表", "耳机", "音响", "摄像机", "相机", "游戏机"],
        "汽车用品": ["车载电子", "汽车装饰"],
        "服装服饰": ["内衣", "上衣", "裤子", "外套", "裙子", "鞋子", "帽子", "围巾", "手套"],
        "家居办公": ["厨具", "卫浴用品", "床上用品", "家具", "文具", "办公用品"],
        "食品生鲜": ["米面", "蛋奶", "蔬菜", "水果", "零食", "饮料", "水产", "肉类", "调味品"],
        "母婴儿童": ["婴儿用品", "儿童课外读物", "益智玩具", "玩具", "模型"],
        "运动户外": ["健身器材", "户外装备"]
    }
    reverse_category = {v:k for k, values in category_distribute.items() for v in values}
    
    # 向量化处理核心逻辑
    def process_row(data):
        if not data:
            return {
                'total_spent': 0, 'avg_spent_per_trans': 0, 'items_count': 0,
                'category_distribution': json.dumps({}), **{f'{k}_spend':0 for k in category_distribute},
                'payment_method': None, 'payment_status': None, 'purchase_date': None, 'credit_score':0
            }
            
        items = data.get('items', [])
        main_category = reverse_category.get(data.get('categories'), None)
        
        # 支付状态处理
        payment_status = data.get('payment_status', '支付成功')
        multiplier = 1 if payment_status == '支付成功' else 0.5 if payment_status == '部分退款' else 0
        
        # 计算核心指标
        items_count = len(items)
        avg_price = data.get('avg_price', 0)
        adjusted_total = avg_price * items_count * multiplier
        # credit_contribution = items_count * (multiplier if multiplier != 0.5 else 0.5)
        
        # 类别消费分配
        category_spend = {f'{k}_spend': adjusted_total if k == main_category else 0 for k in category_distribute}
        
        return {
            'total_spent': adjusted_total,
            'avg_spent_per_trans': avg_price,
            'items_count': items_count * multiplier,
            'category_distribution': json.dumps({main_category: items_count} if main_category else {}),
            **category_spend,
            'payment_method': data.get('payment_method'),
            'payment_status': payment_status,
            'purchase_date': data.get('purchase_date'),
            # 'credit_score': credit_contribution
        }
    
    # 批量处理所有记录
    results = [process_row(data) for data in parsed_data]
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(results)
    result_df._parse_errors = parse_errors
    result_df._multi_category_count = sum(len(set(item.get('category') for item in data.get('items', []))) >= 2 
                                       for data in parsed_data if data)
    return result_df

def process_login_history(df):
    # 使用pandas字符串方法替代apply
    parsed_str = df.astype(str).str.replace("'", '"', regex=False)
    parsed = pd.json_normalize(parsed_str.apply(json.loads))
    
    # 向量化处理时间戳
    timestamps = parsed['timestamps'].explode().str[:10]
    login_dates = pd.to_datetime(timestamps, errors='coerce').dropna()
    
    # 使用groupby聚合代替循环
    aggregated = login_dates.groupby(level=0).agg(
        first_login_history='min',
        last_login_history='max',
    )
    
    # 合并平均会话时长
    result_df = parsed[['avg_session_duration']].join(aggregated)
    result_df['avg_session'] = result_df['avg_session_duration'].fillna(0)
    
    # 合并所有需要的字段
    result_df = parsed[['login_count', 'avg_session_duration']].join(aggregated)
    result_df['avg_session'] = result_df['avg_session_duration'].fillna(0)
    
    # 处理空记录
    parse_errors = len(df) - len(result_df.dropna())
    result_df = result_df.fillna({
        'first_login_history': pd.NaT,
        'last_login_history': pd.NaT,
        'login_count': 0,
        'avg_session': 0
    })
    
    result_df._parse_errors = parse_errors
    return result_df

# 优化后的购买记录处理
def process_purchase_history(df):
    parse_errors = 0
    multi_category_count = 0  # 新增：多类别购买计数器
    parsed_data = df.apply(safe_json_parse)

    category_distribute = {
        "电子数码": ["智能手机", "平板电脑", "笔记本电脑", "智能手表", "耳机", "音响", "摄像机", "相机", "游戏机"],
        "汽车用品": ["车载电子", "汽车装饰"],
        "服装服饰": ["内衣", "上衣", "裤子", "外套", "裙子", "鞋子", "帽子", "围巾", "手套"],
        "家居办公":["厨具", "卫浴用品", "床上用品", "家具", "文具", "办公用品"],
        "食品生鲜": ["米面", "蛋奶", "蔬菜", "水果", "零食", "饮料", "水产", "肉类", "调味品"],
        "母婴儿童": ["婴儿用品", "儿童课外读物", "益智玩具", "玩具", "模型"],
        "运动户外": ["健身器材", "户外装备"]
    }
    
    results = []
    for data in parsed_data:
        if not data:
            results.append({
                'total_spent': 0,
                'avg_transaction': 0,
                'category': None,
                'items_count': 0,
                'items_list': [],
                'category_distribution': json.dumps({}),
                'electronics_spend': 0,      # 电子数码产品
                'automotive_spend': 0,       # 汽车相关
                'cloth_spend': 0,         # 服装服饰
                'home_office_spend': 0,      # 家居与办公
                'groceries_spend': 0,        # 食品生鲜
                'mother_baby_spend': 0,      # 母婴与儿童
                'sports_outdoor_spend': 0,    # 运动户外
                'payment_method': None, # 支付方式
                'payment_status': None, # 支付状态
                'purchase_date': None, # 购买日期，用其最大值代替 referency_date
                'login_count': None,
                'avg_session': None,
                'last_login': None  # 新增最后登录时间
            })
            parse_errors += 1
            continue
        
        items = data.get('items', [])
        # 新增：统计多类别购买
        categories = set(item.get('category') for item in items if item.get('category'))
        if len(categories) >= 2:
            multi_category_count += 1
            
        items_count = len(items)
        avg_price = data.get('avg_price', 0)
        category = data.get('categories', None)
        payment_method = data.get('payment_method', None)
        payment_status  = data.get('payment_status', None)
        purchase_date = data.get('purchase_date', None)

        original_total = avg_price * items_count
        original_items_count = items_count
        refund_count, refund_partial = 0, 0 # 默认退款数为 0

        # 根据支付状态调整金额和数量
        if payment_status == '已退款':
            refund_count = items_count
            adjusted_total = 0
            adjusted_items_count = 0
            credit_contribution = 0
        elif payment_status == '部分退款':
            refund_partial = items_count * 0.5
            adjusted_total = original_total * 0.5
            adjusted_items_count = original_items_count * 0.5
            credit_contribution = original_items_count * 0.5  # 每个物品加0.5分
        else:  # 默认成功支付
            adjusted_total = original_total
            adjusted_items_count = original_items_count
            credit_contribution = original_items_count * 1  # 每个物品加1分

        # 类别替换
        for key, value in category_distribute.items():
            if category in value:
                category = key
                break

        # 42个小类分为7个小类
        category_spend = {
            '电子数码': max(0, adjusted_total) if category == '电子数码' else 0,
            '汽车用品': max(0, adjusted_total) if category == '汽车用品' else 0,
            '服装服饰': max(0, adjusted_total) if category == '服装服饰' else 0,
            '家居办公': max(0, adjusted_total) if category == '家居办公' else 0,
            '食品生鲜': max(0, adjusted_total) if category == '食品生鲜' else 0,
            '母婴儿童': max(0, adjusted_total) if category == '母婴儿童' else 0,
            '运动户外': max(0, adjusted_total) if category == '运动户外' else 0
        }

        # 计算品类分布
        category_dist = {category: items_count}
        # refund_dist = {category: refund_count}
        # refund_partial_dist = {category: refund_partial}
        results.append({
            'total_spent': adjusted_total,
            'avg_spent_per_trans': avg_price,
            # 'refund_dist': json.dumps(refund_dist),
            # 'refund_partial_dist':json.dumps(refund_partial_dist),
            'items_count': adjusted_items_count,
            # 'items_list': [item['id'] for item in items],
            'category_distribution': json.dumps(category_dist),
            'electronics_spend': category_spend['电子数码'],      # 电子数码产品
            'automotive_spend': category_spend['汽车用品'],       # 汽车相关
            'cloth_spend': category_spend['服装服饰'],         # 服装服饰
            'home_office_spend': category_spend['家居办公'],      # 家居与办公
            'groceries_spend': category_spend['食品生鲜'],        # 食品生鲜
            'mother_baby_spend': category_spend['母婴儿童'],      # 母婴与儿童
            'sports_outdoor_spend': category_spend['运动户外'],    # 运动户外
            'payment_method': payment_method, # 支付方式
            'payment_status': payment_status, # 支付状态
            'purchase_date': purchase_date, # 购买日期，用其最大值代替 referency_date
            'credit_score': credit_contribution
        })
    
    result_df = pd.DataFrame(results)
    # print(result_df.iloc[0])
    result_df._parse_errors = parse_errors
    result_df._multi_category_count = multi_category_count  # 新增：附加多类别统计
    # result_df.drop(columns=['items_list'], inplace=True)  # 删除 items_list 列
    return result_df


def load_and_preprocess(df):
    # last_login 的内容是随机生成的，完全没有任何意义
    df = df.drop(columns=["id", "email", "address", "last_login"]) # 大量重复且对分析过程没有帮助，直接删掉
    # missing_stats = df.isnull().sum()
    # missing_stats['缺失比例'] = (missing_stats / len(df)).apply(lambda x: f"{x:.2%}")
    # print(missing_stats)

    missing_stats = df.isnull().sum().to_frame('缺失数量')
    missing_stats['缺失比例'] = (missing_stats['缺失数量'] / len(df)).apply(lambda x: f"{x:.2%}")
    print("\n缺失值统计:")
    print(missing_stats.to_string())

    # original_count = len(df)
    # print(f"\n原始记录数: {original_count}")
    # df = df.drop_duplicates()
    # duplicate_count = original_count - len(df)
    # print(f"\n[重复记录过滤] 删除完全重复记录数: {duplicate_count} (占比: {duplicate_count/original_count:.2%})")
    # print(f"剩余有效记录数: {len(df)} (保留比例: {len(df)/original_count:.2%})")

    purchase_features = process_purchase_history(df["purchase_history"])
    # 新增：输出多类别购买统计
    print(f"\n[多类别购买统计] 单次购买两类及以上记录数: {purchase_features._multi_category_count} (占比: {purchase_features._multi_category_count/len(df):.2%})")
    parse_errors = purchase_features._parse_errors  # 获取解析错误数
    print(f"\n[purchase_history JSON解析统计] 解析失败记录数: {parse_errors} (占比: {parse_errors/len(df):.2%})")
    print("剩余有效记录数: ", len(df) - parse_errors)
    
    # original_count = len(df)
    # purchase_features.reset_index(drop=True, inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # valid_mask = purchase_features['items_count'] > 0
    # valid_indices = purchase_features[valid_mask].index
    # # df = df.loc[valid_indices]
    # purchase_features = purchase_features.loc[valid_indices]
    # print(f"\n[购物记录过滤] 删除记录数: {original_count - len(df)} (占比: {len(df)/original_count:.2%})")
    # print(f"剩余有效记录数: {len(df)} (保留比例: {len(df)/original_count:.2%})")
    # 合并特征
    # df = pd.concat([df, purchase_features], axis=1)

    ###############################################
    ############ 新增登录记录解析 ##################
    login_features = process_login_history(df["login_history"])
    print(f"\n[login_history JSON解析统计] 解析失败记录数: {parse_errors} (占比: {parse_errors/len(df):.2%})")
    print("剩余有效记录数: ", len(df) - parse_errors)
    
    # original_count = len(df)
    # login_features.reset_index(drop=True, inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # df = df.loc[valid_indices]
    # valid_indices_login = login_features[valid_mask].index
    # login_features = login_features.loc[valid_indices_login]

    # 上面删筛过了，这里不用管，后面的时序逻辑筛选会做的
    # print(f"\n[购物记录过滤] 删除记录数: {original_count - len(df)} (占比: {len(df)/original_count:.2%})")
    # print(f"剩余有效记录数: {len(df)} (保留比例: {len(df)/original_count:.2%})")
    
    # 合并特征
    df = pd.concat([df, purchase_features, login_features], axis=1)
    df = df.drop(columns=["purchase_history", "login_history"])

    # df = df.drop(columns=["purchase_history", "login_history"]) # 这个怕是真会超

    # missing_stats = df.isnull().sum().to_pandas()
    # missing_stats['缺失比例'] = (missing_stats / len(df)).apply(lambda x: f"{x:.2%}")
    # print(missing_stats)  # 转换为 pandas 格式输出

    # 验证日期格式
    if 'registration_date' in df.columns:
        # 修改日期类型检查方式
        if not np.issubdtype(df['registration_date'].dtype, np.datetime64):
            df['registration_date'] = pd.to_datetime(df['registration_date'])

    # 验证时序逻辑
    # 由于多了购物时间，所以需要确保购物时间在注册时间之后且不能超过最后一次登录时间
    if 'registration_date' in df.columns and 'purchase_date' in df.columns and 'last_login_history' in df.columns:
        # 更精确的时间戳解析（处理ISO 8601格式）
        original_count = len(df)

        # df['timestamp'] = cudf.to_datetime(df['timestamp'].str.extract(r'([^+]+)')[0])
        # time_logic_violation = len(df[df['registration_date'] > df['timestamp']])
        # print(f"\n[时序逻辑过滤] 注册时间晚于首次消费的记录数: {time_logic_violation} (占比: {time_logic_violation/original_count:.2%})")
        # # 确保注册时间早于第一次消费时间
        # df = df[df['registration_date'] <= df['timestamp']]

        # 时间格式转化
        if not np.issubdtype(df['purchase_date'].dtype, np.datetime64):
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])

        if not np.issubdtype(df['last_login_history'].dtype, np.datetime64):
            df['last_login_history'] = pd.to_datetime(df['last_login_history'])

        # 由于登录时间是从数组里面排序提取的，所以首次登录时间肯定早于最后一次登录时间，这个不用检查

        # 1. 检查注册时间是否早于首次消费时间
        reg_after_purchase = df['registration_date'] > df['purchase_date']
        time_logic_violation = reg_after_purchase.sum()
        print(f"\n[时序逻辑过滤] 注册时间晚于首次消费的记录数: {time_logic_violation} (占比: {time_logic_violation/original_count:.2%})")
        
        # 2. 检查购物时间是否超过最后一次登录时间
        purchase_after_login = df['purchase_date'] > df['last_login_history']
        login_logic_violation = purchase_after_login.sum()
        print(f"[时序逻辑过滤] 购物时间超过最后一次登录时间的记录数: {login_logic_violation} (占比: {login_logic_violation/original_count:.2%})")
        
        # 应用过滤条件
        valid_mask = (~reg_after_purchase) & (~purchase_after_login)
        df = df[valid_mask]

    # 处理异常值
    pre_filter_count = len(df)
    df = df[(df['age'].between(18, 100)) & (df['income'] > 0)]
    post_filter_count = len(df)
    print(f"\n[异常值过滤] 删除记录数: {pre_filter_count - post_filter_count} (占比: {(pre_filter_count - post_filter_count)/pre_filter_count:.2%})")
    print(f"剩余有效记录数: {post_filter_count} (保留比例: {post_filter_count/pre_filter_count:.2%})")

    # 定义唯一用户id
    df['user_id'] = df['user_name'] + '_' + df['phone_number']
    
    # 支付状态和支付方式不好聚合，所以这里直接提前处理把图画了
    draw_bar_chart(df[['payment_method', 'payment_status']], size)

    # 聚合函数（按理来说 user_id 完全没有重复应该不需要聚合，但是购买字段每次只包含一次购买记录，所以我觉得聚合还是有必要的）
    grouped = df.groupby('user_id').agg({
        'purchase_date': ['count', 'nunique', 'max'],
        'total_spent': 'sum',
        'items_count': 'sum',
        'age': 'first',
        'income': 'first',
        'is_active': 'first',
        'gender': 'first',
        'country': 'first',
        'registration_date': 'first',
        'first_login_history': 'first',
        # 'credit_score': 'sum', 
        'electronics_spend': 'sum',      
        'automotive_spend': 'sum',       
        'cloth_spend': 'sum',        
        'home_office_spend': 'sum',      
        'groceries_spend': 'sum',      
        'mother_baby_spend': 'sum',    
        'sports_outdoor_spend': 'sum',   
        'login_count': 'sum',
        'avg_session': 'mean',
        'last_login_history': 'max'  # 新增最后登录时间
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

    # ------------------- 退款统计 -----------------------------
    # 这里我直接把退款件数跟类别放到一起成一个json，后处理的时候再进行解析

    # 列名处理和重命名
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.rename(columns={
        'purchase_date_count': 'purchase_count',
        'total_spent_sum': 'total_spent',
        'avg_spent_per_trans_mean': 'avg_transaction',
        'items_count_sum': 'total_items',
        'purchase_date_max': 'last_purchase', 
        'category_distribution_<lambda>': 'category_distribution',
        'electronics_spend_sum': 'electronics_spend',     
        'automotive_spend_sum': 'automotive_spend',      
        'cloth_spend_sum': 'cloth_spend',        
        'home_office_spend_sum': 'home_office_spend',     
        'groceries_spend_sum': 'groceries_spend',      
        'mother_baby_spend_sum': 'mother_baby_spend',    
        'sports_outdoor_spend_sum': 'sports_outdoor_spend',   
        'login_count_sum': 'login_count',
        'avg_session_mean': 'avg_session',
        'last_login_history_max': 'last_login', 
        'credit_score_sum': 'credit_score'
    })
    
    # 计算衍生特征
    df_final = grouped.reset_index()
    df_final.rename(columns=lambda col: col[:-6] if col.endswith('_first') else col, inplace=True) # 去掉 first 后缀
    
    # original_count = len(df_final)
    # 暂时不过滤这部分内容（毕竟完全退款也是值得分析的，在分析金额和购买数量等后处理时再进行过滤就行）
    # df_final = df_final[df_final['total_items'] != 0] # 二次过滤，但是留下已退款的表项用于统计退货率跟其他属性之间的关系
    # filtered_count = len(df_final)

    # print(f"\n[购物记录过滤（聚合后] 删除记录数: {original_count-filtered_count} (占比: {(original_count-filtered_count)/original_count:.2%})")
    # print(f"剩余有效记录数: {len(df_final)} (保留比例: {len(df_final)/original_count:.2%})")

    current_time = datetime.now()
    df_final['recency_days'] = (current_time - pd.to_datetime(df_final['last_purchase'])).dt.days
    df_final['membership_days'] = (current_time - df_final['registration_date']).dt.days
    
    # 综合活跃度计算 = 登录次数 × 平均会话时长
    df_final['active_score'] = df_final['login_count'] * df_final['avg_session']
    
    df_final.drop(columns=['login_count', 'avg_session'], inplace=True)

    # 电商零售的话根据购买物品数计算频率会比较合理
    df_final['purchase_freq'] = df_final['total_items'] / ((df_final['membership_days']/30) + 1)  # 月均购买次数

    # 删除冗余列
    return df_final.drop(columns=['user_name', 'phone_number'], errors='ignore')

# 主程序改造
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True,
                       help='指定要处理的数据集大小 (10G 或 30G)')
    args = parser.parse_args()
    size = args.size

    # 根据参数选择数据目录和输出目录
    data_dir = f'{args.size}_data_new'
    output_dir = f'processed_csv_{args.size}'
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    start_time = time.time()
    
    dfs = []
    for i, file in enumerate(parquet_files):
        print(f"Loading part {i+1}")
        df = pq.ParquetFile(os.path.join(data_dir, file)).read().to_pandas()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    start_time = time.time()
    df = load_and_preprocess(df)
    df.to_csv(f'{output_dir}/pandas_{args.size}.csv', index=False)
    single_timestamp_count = len(df[df['purchase_date_nunique'] == 1])
    print(f"\n[时间戳分析] 单时间戳用户数: {single_timestamp_count} (占比: {single_timestamp_count/len(df):.2%})")

    print(f"\n数据加载用时: {elapsed_time:.2f}秒")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n程序总执行时间: {elapsed_time:.2f}秒")