import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import json
import numpy as np
from scipy import stats
import warnings
from cuml.preprocessing import StandardScaler
import argparse
from multiprocessing import Pool
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框
save_type = "10G"

def save_single_plot(fig, ax, filename):
    """单独保存子图的工具函数"""
    fig.tight_layout()
    if save_type == "10G":
        fig.savefig(f'result_imgs/{filename}', dpi=300, bbox_inches='tight')
    else:
        fig.savefig(f'result_imgs_30G/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭当前figure释放内存

# 1. 年龄分布直方图；2. 不同类别物品消费结构饼状图；3. 国家分布柱状图；
# 4. 收入-信用评分关系热力图；5. 购买类别分析热力图（哪些物品通常一起被购买）；
# 6. 比较不同性别和活跃状态用户的消费分布（活跃用户）；7. 消费金额分布（非活跃用户）；8. 用户消费金额分布
def vis1(df):
    df_without_refund = df[df['total_items'] > 0] # 过滤掉退款的记录，这些记录不计入商家销售额

    COLORS = {
        'primary': ['#8ECFC9', '#FFBE7A', '#FA7F6F'],
        'secondary': ['#B395BD', '#7DAEE0', '#EA8379', '#C8A2C8'],
        'accent': ['#299D8F', '#E9C46A', '#D87659'],
        'gender': ['#456990', '#E487C0'],
        'mono': ['#333333', '#666666', '#999999']
    }

    # 1. 年龄分布
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['age'], bins=20, kde=True, color=COLORS['primary'][2], linewidth=0.5, ax=ax1)
    ax1.set_title('年龄分布', pad=20)
    ax1.set_xlabel('年龄', labelpad=10)
    ax1.set_ylabel('样本量', labelpad=10)
    sns.despine()
    save_single_plot(fig1, ax1, '01_age_distribution.png')

    # 2. 消费结构分析（需要筛掉未付款的）
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    category_data = pd.json_normalize(df_without_refund['category_distribution'].apply(json.loads)).sum()
    category_data.plot.pie(
        colors=COLORS['primary'] + COLORS['secondary'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'w'},
        ax=ax2
    )
    ax2.set_title('消费结构分析', pad=20)
    save_single_plot(fig2, ax2, '02_category_distribution.png')

    # 3. 销售额占比（需要筛掉未付款的）
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    category_sales = df_without_refund[['electronics_spend', 'automotive_spend', 'cloth_spend', 'home_office_spend', 'groceries_spend', 'mother_baby_spend', 'sports_outdoor_spend']].sum()
    # 定义标签
    categories = ['电子数码', '汽车相关', '服装服饰', '家居办公', '食品生鲜', '母婴儿童', '运动户外']
    category_sales.plot.pie(
        labels=categories,
        colors=COLORS['primary'] + COLORS['secondary'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'w'},
        ax=ax3
    )
    ax3.set_title('物品类别销售总额占比', pad=20)
    save_single_plot(fig3, ax3, '03_category_sales_distribution.png')

    # 4. 购买类别相关性（需要筛掉未付款的，不过感觉筛不筛都可以）
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    try:
        # 初始化共现矩阵
        categories = ['电子数码', '汽车用品', '服装服饰', '家居办公', '食品生鲜', '母婴儿童', '运动户外']
        co_occurrence = pd.DataFrame(0, index=categories, columns=categories, dtype=float)
        
        # 从数据中随机采样10000条记录
        sample_df = df_without_refund.sample(n=min(10000, len(df_without_refund)), random_state=42)
        
        # 遍历采样用户的购买记录
        for dist in sample_df['category_distribution']:
            cat_dict = json.loads(dist)
            # 获取当前用户购买的品类（这里抽样时把退款记录自动删了）
            present_cats = [cat for cat, count in cat_dict.items() if count > 0]
            # 更新共现矩阵（考虑多品类组合）
            for i in range(len(present_cats)):
                for j in range(i, len(present_cats)):
                    co_occurrence.loc[present_cats[i], present_cats[j]] += 1
                    if i != j:  # 避免重复计数对角线
                        co_occurrence.loc[present_cats[j], present_cats[i]] += 1
        
        # 计算概率矩阵（考虑品类出现次数）
        row_sums = co_occurrence.sum(axis=1)
        co_occurrence_prob = co_occurrence.div(row_sums, axis=0)

        # 生成热力图
        sns.heatmap(
            co_occurrence_prob,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax4,
        )
        ax4.set_title('品类共现概率分析', pad=20, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
    except Exception as e:
        print(f"共现热力图生成错误: {str(e)}")
        ax4.text(0.5, 0.5, '数据不足或格式错误\n'+str(e), 
                ha='center', va='center',
                fontsize=12, color='red')
    save_single_plot(fig4, ax4, '04_category_correlation.png')

    # 5. 国家分布
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, y='country', palette=COLORS['primary'] + COLORS['secondary'], edgecolor='black', ax=ax5)
    ax5.set_title('国家分布', pad=20)
    ax5.set_xlabel('人数', labelpad=10)
    save_single_plot(fig5, ax5, '05_country_distribution.png')

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    df['active_status'] = df['is_active'].map({False: '不活跃', True: '活跃'})  # 修改图例文字
    
    sns.boxplot(
        data=df,
        x='gender',
        y='total_spent',
        hue='active_status',
        palette=COLORS['primary'][:2],  # 使用两种颜色区分
        width=0.7,
        fliersize=3,
        linewidth=1.2,
        ax=ax6
    )
    ax6.set_title('用户活跃度消费分布对比', pad=20)
    ax6.set_xlabel('性别', labelpad=10)
    ax6.set_ylabel('消费金额', labelpad=10)
    ax6.legend(title='用户状态', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    save_single_plot(fig6, ax6, '06_user_activity_comparison.png')

    # 7. 消费金额分布
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.histplot(
        df_without_refund['total_spent'],
        bins=30,
        kde=True,
        color=COLORS['accent'][0],
        edgecolor='white',
        linewidth=0.5,
        ax=ax7
    )
    ax7.set_title('消费金额分布', pad=20)
    ax7.set_xlabel('金额', labelpad=10)
    ax7.set_ylabel('样本量', labelpad=10)
    save_single_plot(fig7, ax7, '07_total_spent_distribution.png')

    # -------------------------- 活跃度相关分析 -----------------------------
    # 动态划分活跃等级（使用四分位数）
    activity_quantiles = df['active_score'].quantile([0.25, 0.5, 0.75]).values
    df['active_level'] = pd.cut(
        df['active_score'],
        bins=[-1, activity_quantiles[0], activity_quantiles[1], activity_quantiles[2], np.inf],
        labels=['低活跃', '中活跃', '高活跃', '极高活跃']
    )

    # 8. 性别-活跃用户关系
    fig8, ax8 = plt.subplots(figsize=(8, 6))
    activity_gender = pd.crosstab(df['gender'], df['active_level'], normalize='index') * 100
    sns.heatmap(
        activity_gender.T,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': '占比 (%)'},
        ax=ax8
    )
    ax8.set_title('性别-活跃度分布热力图', pad=20)
    ax8.set_xlabel('性别', labelpad=10)
    ax8.set_ylabel('活跃等级', labelpad=10)
    ax8.tick_params(axis='y', rotation=0)
    save_single_plot(fig8, ax8, '08_gender_activity_heatmap.png')

    # 9 年龄与活跃度关系（更新为散点密度图）
    fig9 = plt.figure(figsize=(8, 6))
    joint_grid = sns.jointplot(
        data=df,
        x='age',
        y='active_score',
        kind='hex',
        cmap='Greens',
        height=8,
        marginal_kws=dict(color=COLORS['accent'][0])  # 移除了bins参数
    )
    joint_grid.fig.suptitle('年龄-活跃度分布密度图', y=1.02)
    joint_grid.ax_joint.set_xlabel('年龄', labelpad=10)
    joint_grid.ax_joint.set_ylabel('综合活跃度', labelpad=10)
    
    # 保存图表
    if save_type == "10G":
        joint_grid.savefig('result_imgs/09_age_activity_density.png', dpi=300, bbox_inches='tight')
    else:
        joint_grid.savefig('result_imgs_30G/09_age_activity_density.png', dpi=300, bbox_inches='tight')
    plt.close(joint_grid.fig)

# 年龄-产品类别；产品类别-性别；年龄-收入 堆叠图 + 每个国家的销售总额以及人均消费（销售总额用饼状图，人均消费用柱状图）
# 这里收入以 10w 为一档进行划分，具体年龄划分减内部代码
def vis2(df):
    df_without_refund = df[df['total_items'] > 0] # 过滤掉退款的记录，这些记录不计入商家销售额

    COLORS = {
        'primary': ['#8ECFC9', '#C8A2C8', '#FA7F6F'],
        'secondary': ['#B395BD', '#7DAEE0', '#EA8379', '#FFBE7A'],
        'accent': ['#299D8F', '#E9C46A', '#D87659'],
        'gender': ['#456990', '#E487C0']
    }

    start_time = time.time()
    # 预处理公共数据
    category_df = df['category_distribution'].apply(lambda x: json.loads(x)).apply(pd.Series).fillna(0)
    # 年龄分箱
    age_bins = [18, 35, 50, 65, 79, 100]
    age_labels = ['18-35', '36-50', '51-65', '66-79', '80-100']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df_without_refund['age_group'] = df['age_group']
    # 收入分箱
    income_bins = list(range(0, 1000001, 100000))
    income_labels = [f'{i//10000}w-{(i+100000)//10000}w' for i in income_bins[:-1]]
    df['income_group'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)
    df_without_refund['income_group'] = df['income_group']
    end_time = time.time()
    print(f"vis2 预处理耗时: {end_time - start_time:.2f} 秒")

    # 1. 年龄-产品类别堆叠图（涉及到购买产品类别，也需要筛掉未付款的记录）
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    age_cat = pd.concat([df_without_refund['age_group'], category_df], axis=1).groupby('age_group').sum()
    age_cat_pct = age_cat.div(age_cat.sum(axis=1), axis=0)
    colors = plt.cm.tab10.colors[:7]
    colors = [(r, g, b, 0.5) for r, g, b in colors]
    age_cat_pct.plot.bar(
        stacked=True,
        color=colors,
        ax=ax1
    )
    ax1.set_title('年龄-产品类别分布', pad=20)
    ax1.set_ylabel('占比 (%)')
    ax1.set_xlabel('年龄组')
    ax1.tick_params(axis='x', rotation=0) 
    ax1.legend(
        loc='upper right',  # 设置图例位置为右上角
        bbox_to_anchor=(1.15, 1),  
        frameon=False,  
        fontsize=10  
    )
    for patch in ax1.patches:
        height = patch.get_height()
        if height > 0.01:
            ax1.text(patch.get_x() + patch.get_width()/2, 
                   patch.get_y() + height/2, 
                   f'{height:.1%}', 
                   ha='center', va='center',
                   fontsize=10, color='white')
    save_single_plot(fig1, ax1, '10_age_category_distribution.png')

    # 2. 性别-产品类别堆叠图（涉及到购买产品类别，也需要筛掉未付款的记录）
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    gender_cat = pd.concat([df['gender'], category_df], axis=1).groupby('gender').sum()
    gender_cat_pct = gender_cat.div(gender_cat.sum(axis=1), axis=0)
    colors = plt.cm.tab10.colors[:7]
    colors = [(r, g, b, 0.5) for r, g, b in colors]
    gender_cat_pct.plot.bar(
        stacked=True,
        color=colors,
        ax=ax2
    )
    ax2.set_title('性别-产品类别分布', pad=20)
    ax2.set_ylabel('占比 (%)')
    ax2.tick_params(axis='x', rotation=0) 
    ax2.legend(
        loc='upper right', 
        bbox_to_anchor=(1.15, 1),  
        frameon=False, 
        fontsize=10  
    )
    for patch in ax2.patches:
        height = patch.get_height()
        if height > 0.01:
            ax2.text(patch.get_x() + patch.get_width()/2,
                   patch.get_y() + height/2,
                   f'{height:.1%}', 
                   ha='center', va='center',
                   fontsize=10, color='white')
    save_single_plot(fig2, ax2, '11_gender_category_distribution.png')

    # 3. 年龄-收入分布堆叠图
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    colors = [(r, g, b, 0.5) for r, g, b in colors]
    age_income = df.groupby(['age_group', 'income_group']).size().unstack(fill_value=0)
    age_income_pct = age_income.div(age_income.sum(axis=1), axis=0)
    age_income_pct.plot.bar(
        stacked=True,
        color=colors,
        alpha=0.5,
        ax=ax3
    )
    ax3.set_title('年龄-收入分布', pad=20)
    ax3.set_ylabel('占比 (%)')
    ax3.tick_params(axis='x', rotation=0) 
    ax3.legend(
        loc='upper right',  
        bbox_to_anchor=(1.15, 1),  
        frameon=False, 
        fontsize=10 
    )
    for patch in ax3.patches:
        height = patch.get_height()
        if height > 0.05:
            ax3.text(patch.get_x() + patch.get_width()/2,
                   patch.get_y() + height/2,
                   f'{height:.0%}', 
                   ha='center', va='center',
                   fontsize=10, color='white')
    save_single_plot(fig3, ax3, '12_age_income_distribution.png')

    # 4. 国家销售总额分析（total_spent 未付款的会被赋0，虽然可以不筛，但是筛掉处理会快一点）
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    country_sales = df_without_refund.groupby('country')['total_spent'].sum().sort_values(ascending=False)
    country_sales.plot.pie(
        colors=COLORS['primary']+ COLORS['secondary'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 3, 'edgecolor': 'w'},
        ax=ax4
    )
    ax4.set_title('国家销售总额分布', pad=20)
    save_single_plot(fig4, ax4, '13_country_sales_distribution.png')

    # 5. 国家人均消费分析（没花钱，但是算人头，筛不筛？）
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    country_order = df_without_refund.groupby('country')['total_spent'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df_without_refund,
        x='country',
        y='total_spent',
        palette=COLORS['primary'] + COLORS['accent'],
        width=0.6,
        showfliers=False,
        linewidth=1.2,
        order=country_order,
        ax=ax5
    )
    ax5.set_title('国家人均消费分布', pad=20)
    ax5.set_xlabel('国家', labelpad=10)
    ax5.set_ylabel('消费金额', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    save_single_plot(fig5, ax5, '14_country_spending_distribution.png')

    # 6. 性别分布统计
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    plot = sns.countplot(
        data=df,
        x='gender',
        palette=COLORS['gender'],
        edgecolor='black',
        ax=ax6
    )
    total = len(df) 
    # 在每个柱子上添加数量和百分比标签
    for p in plot.patches:
        height = p.get_height()
        percentage = 100 * height / total
        ax6.text(p.get_x() + p.get_width()/2.,
                height + 1,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10)
    ax6.set_title('性别分布统计', pad=20)
    ax6.set_ylabel('人数')
    save_single_plot(fig6, ax6, '15_gender_distribution.png')

# 收入-消费金额关系分析（使用 Spearman 相关系数）
def analyze_income_spending(df):
    # 1. 数据分箱处理（每 10w 一档）
    df['income_bin'] = pd.cut(df['income'],
                             bins=list(range(0, int(df['income'].max())+100000, 100000)),
                             right=False)
    
    # 2. 聚合统计
    agg_df = df.groupby('income_bin').agg(
        median_spending=('total_spent', 'median'),
        avg_spending=('total_spent', 'mean'),
        count=('user_id', 'count')
    ).reset_index()

    # 转换分箱标签
    agg_df['income_label'] = agg_df['income_bin'].apply(
        lambda x: f"{x.left//10000}-{(x.right)//10000}"
    )
    # 3. 过滤样本量不足区间
    agg_df = agg_df[agg_df['count'] >= 10]
    # 4. 创建组合图表
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 主图：消费中位数趋势
    sns.lineplot(
        data=agg_df,
        x='income_label',
        y='median_spending',
        marker='o',
        color='skyblue',  
        linewidth=2,
        label='消费金额中位数',
        ax=ax1
    )
    
    # 添加平均值曲线
    sns.lineplot(
        data=agg_df,
        x='income_label',
        y='avg_spending',
        marker='^',
        color='#FA7F6F',  # 红色系强调
        linewidth=1.5,
        linestyle='--',
        label='消费金额平均值',
        ax=ax1
    )
    ax1.set_ylabel('消费金额（元）', color='#456990')
    ax1.tick_params(axis='y', labelcolor='#456990')
    
    # 次坐标轴：样本量分布
    ax2 = ax1.twinx()
    sns.barplot(
        data=agg_df,
        x='income_label',
        y='count',
        alpha=0.3,
        color='#8ECFC9',
        ax=ax2
    )
    ax2.set_ylabel('样本量', color='#456990')
    ax2.tick_params(axis='y', labelcolor='#456990')
    
    # 图表优化
    plt.title('收入-消费金额关系分析（单位：10万元）', pad=20)
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('收入区间（10万元）', labelpad=10)
    sns.despine()
    
    # 添加统计指标
    corr, p = stats.spearmanr(df['income'], df['total_spent'])
    plt.text(
        0.7, 0.85, 
        f"Spearman's r = {corr:.2f}\np = {p:.2g}",
        transform=ax1.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    if save_type == "10G":
        plt.savefig('result_imgs/income_spending_analysis.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('result_imgs_30G/income_spending_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 高价值用户识别函数改造
def identify_high_value_users(df):

    print(len(df))
    df = df[df['age'] <= 70] # 年龄较大的用户不予以考虑
    df = df.dropna() # 防止 NA 值影响后续分析
    # print(len(df))
    df['is_active'] = df['is_active'].astype(int)

    zero_spent_mask = df['total_spent'] == 0
    df.loc[zero_spent_mask, 'recency_days'] = 1000
    df.loc[zero_spent_mask, 'purchase_freq'] = 0

    scaler = StandardScaler()
    # 标准化指标
    metrics = ['recency_days', 'total_spent', 'income', 'active_score', 'purchase_freq']
    features = scaler.fit_transform(df[metrics])
    features.columns = metrics

    # 计算综合评分
    weights = {
        'recency_days': 0.25,   
        'purchase_freq': 0.15,  
        'total_spent': 0.25,  
        'income': 0.1,   
        'active_score': 0.25
    }
    df['rfm_score'] = (
        (1 - features['recency_days']) * weights['recency_days'] + 
        features['purchase_freq'] * weights['purchase_freq'] +
        features['total_spent'] * weights['total_spent'] +
        features['income'] * weights['income'] +
        features['active_score'] * weights['active_score']
    )
    
    # 识别高价值用户(前15%)
    threshold = df['rfm_score'].quantile(0.85)
    high_value_users = df[df['rfm_score'] > threshold].copy()
    # 用户分层(3个等级)
    labels=['白银用户', '黄金用户', '钻石用户']
    high_value_users['hvc_segment'] = pd.qcut(
        high_value_users['rfm_score'],
        3,
        labels=labels,
        duplicates='drop'
    )
    if save_type == "10G":
        high_value_users.to_csv('high_value_users/high_value_users.csv', index=False)
    else:
        high_value_users.to_csv('high_value_users_30G/high_value_users.csv', index=False)

    # 创建平衡训练集
    non_hvc_users = df[df['rfm_score'] <= threshold].sample(n=len(high_value_users))
    balanced_train = pd.concat([
        high_value_users.assign(label=1),  # 高价值用户标记为1
        non_hvc_users.assign(label=0)  # 非高价值用户标记为0
    ])
    if save_type == "10G":
        balanced_train.to_csv('predict_model/balanced_train_set.csv', index=False)
    else:
        balanced_train.to_csv('predict_model/balanced_train_set_30G.csv', index=False)

    # 7. 可视化分析
    plt.figure(figsize=(14, 8))
    palette = {
        '白银用户': '#C0C0C0',
        '黄金用户': '#FFD700',
        '钻石用户': '#B9F2FF'
    }
    sns.scatterplot(
        data=high_value_users[:min(len(high_value_users), 500)],
        x='total_spent',
        y='active_score',
        hue='hvc_segment',
        size='recency_days',
        sizes=(40, 200),
        alpha=0.8,
        palette=palette,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title('高价值客户分层分析', fontsize=16, pad=20)
    plt.xlabel('总消费金额 (元)', fontsize=12, labelpad=10)
    plt.ylabel('用户活跃程度', fontsize=12, labelpad=10)
    plt.legend(
        title='客户等级',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    if save_type == "10G":
        plt.savefig('high_value_users/hvc_segmentation.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('high_value_users_30G/hvc_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return high_value_users

def iqr_outlier_filter(df, exclude_cols=None, include_cols=None, multiplier=1.5, min_bound=0, 
                                inplace=False, min_valid_percent=0.7):
    if not inplace:
        df = df.copy()
    # 记录原始数据量
    original_count = len(df)
    
    # 设置默认排除列
    if exclude_cols is None:
        exclude_cols = ['purchase_freq', 'electronics_spend', 'automotive_spend', 'cloth_spend', 'home_office_spend', 'groceries_spend', 'mother_baby_spend', 'sports_outdoor_spend']
    # 确定要处理的列
    if include_cols is not None:
        numeric_cols = [col for col in include_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    
    outlier_stats = {}
    conditions = []
    dropped_conditions = []
    for col in numeric_cols:
        col_multiplier = multiplier[col] if isinstance(multiplier, dict) and col in multiplier else multiplier
        col_min_bound = min_bound[col] if isinstance(min_bound, dict) and col in min_bound else min_bound
        
        # 处理前的非缺失值数量
        non_null_count = df[col].count()
        
        # 计算四分位数和IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(q1 - col_multiplier * iqr, col_min_bound)
        upper_bound = q3 + col_multiplier * iqr
        
        # 构建过滤条件
        cond = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        
        # 检查该条件应用后剩余的数据比例
        valid_percent = cond.mean()
        
        # 如果筛选后数据量太少，放弃该条件
        if valid_percent < min_valid_percent:
            dropped_conditions.append(col)
            continue
            
        conditions.append(cond)
        
        # 统计异常值
        outliers = (~cond).sum()
        if outliers > 0:
            outlier_stats[col] = {
                '异常值数量': int(outliers),
                '异常比例': outliers/non_null_count,
                '下界': lower_bound,
                '上界': upper_bound,
                'Q1': q1,
                'Q3': q3,
                'IQR': iqr
            }
    
    # 应用过滤条件
    if conditions:
        combined_cond = pd.Series(True, index=df.index)
        for c in conditions:
            combined_cond &= c
        df = df[combined_cond]
    
    # 构建统计信息
    stats = {
        '原始数据量': original_count,
        '处理后数据量': len(df),
        '删除记录数': original_count - len(df),
        '删除占比': (original_count - len(df))/original_count if original_count > 0 else 0,
        '异常值明细': outlier_stats,
        '已跳过列': dropped_conditions
    }
    
    # 打印统计信息
    print(f"删除前数据量: {stats['原始数据量']}")
    print(f"\n[全局异常值过滤] 删除记录数: {stats['删除记录数']} (占比: {stats['删除占比']:.2%})")
    
    if outlier_stats:
        stats_df = pd.DataFrame.from_dict(
            {k: {
                '异常值数量': v['异常值数量'], 
                '异常比例': f"{v['异常比例']:.2%}",
                '正常范围': f"[{v['下界']:.2f}, {v['上界']:.2f}]"
            } for k, v in outlier_stats.items()}, 
            orient='index'
        )
        print("\n异常值统计明细:")
        print(stats_df.to_string())
    else:
        print("\n未检测到显著异常值")
        
    if dropped_conditions:
        print(f"\n以下列因筛选后数据量低于{min_valid_percent:.0%}而被跳过:")
        print(", ".join(dropped_conditions))
        
    print(f"\n处理后的数据量: {len(df)}")
    return df

def read_csv_parallel(file_path):
        return pd.read_csv(file_path)

# 主程序改造
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True, default="10G",
                       help='指定要处理的数据集大小 (10G 或 30G)')
    parser.add_argument('--type', type=str, choices=['partial', 'pandas'], required=True, default="pandas",
                       help='指定要处理的数据集')
    args = parser.parse_args()
    save_type = args.size
    start_time = time.time()  # 记录开始时间

    if args.type == 'pandas':
        file_path = f'processed_csv_{args.size}/{args.type}_{args.size}.csv'     
        df = pd.read_csv(file_path)

    else:
        # 使用多进程并行读取文件
        if save_type == "10G":
            file_paths = [f'processed_csv/data_part_{i}.csv' for i in range(8)]
        else:
            file_paths = [f'processed_csv_30G/data_part_{i}.csv' for i in range(16)]

        # 根据CPU核心数设置进程池大小
        cpu_count = os.cpu_count()
        pool_size = min(cpu_count, len(file_paths)) if cpu_count else 8
        
        with Pool(pool_size) as pool:
            dfs = pool.map(read_csv_parallel, file_paths)

    df = iqr_outlier_filter(df)

    # # 示例分析
    print("\n处理后的数据样例:")
    print(df.head(3)[['user_id', 'purchase_count', 'total_spent']])

    vis1(df)
    analyze_income_spending(df)
    vis2(df)
    
    # 高价值用户识别
    hvc_df = identify_high_value_users(df)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"\n程序总执行时间: {elapsed_time:.2f}秒")