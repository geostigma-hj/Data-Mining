import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 配置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_model_comparison(final_metrics):
    """绘制更美观的模型对比柱状图"""
    # 转换数据结构并提取数值
    metrics_df = pd.DataFrame(final_metrics).T
    # 提取均值部分转换为浮点数
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].apply(lambda x: float(x.split('±')[0].strip()))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(16, 8))
    # 为每个指标创建子图
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        # 绘制柱状图并添加数值标签
        ax = sns.barplot(x=metrics_df.index, y=metric, data=metrics_df, 
                        hue=metrics_df.index,
                        palette='viridis', alpha=0.8, legend=False)
        # 添加数值标签
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.title(metric.upper(), fontsize=12, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=15)
        plt.grid(True, linestyle='--', alpha=0.6)
    # 添加整体标题
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_metrics_table(final_metrics):
    """保存指标对比表格"""
    df = pd.DataFrame(final_metrics).T
    # 保存CSV格式
    df.to_csv('results/metrics_comparison.csv')

# 数据加载和预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("原始数据NaN统计:")
    print(df.isna().sum())
    # 删除无关特征
    drop_columns = ['user_id', 'purchase_count', 'timestamp_nunique', 
                'rfm_score', 'hvc_segment', 'registration_date', 'last_purchase']
    df = df.drop(columns=drop_columns)
    original_count = len(df)
    df = df.dropna()
    removed_count = original_count - len(df)
    print(f"\n移除了 {removed_count} 条包含NaN的记录")
    
    # 日期信息之前已经处理过了（注册时间和最后购买时间）

    # current_date = pd.to_datetime('today')
    # df['last_purchase'] = (current_date - pd.to_datetime(df['last_purchase'])).dt.days
    # df['registration_date'] = (current_date - pd.to_datetime(df['registration_date'])).dt.days
    
    return df

# 定义特征工程
def create_preprocessor(num_features, cat_features):
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])
    
    return preprocessor

# 定义评估函数
def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }
    return metrics

# 定义PyTorch神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def save_metrics_table(final_metrics, path):
    df = pd.DataFrame(final_metrics).T
    df.to_csv(path)
