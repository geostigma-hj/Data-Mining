import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from cuml.ensemble import RandomForestClassifier
from utils import load_data, create_preprocessor, evaluate_model, save_metrics_table
import argparse

def main(size):
    # 加载数据
    if size == '10G':
        df = load_data('balanced_train_set.csv')
    else:
        df = load_data('balanced_train_set_30G.csv')
    
    # 定义特征和标签
    y = df['label'].values
    X = df.drop(columns=['label'])
    # 定义特征类型
    num_features = ['total_spent', 'avg_transaction',
                    'total_items', 'age', 'income', 'credit_score_mean',
                    'recency_days', 'membership_days', 'purchase_freq']
    cat_features = ['gender', 'country']
    
    # 创建预处理管道
    preprocessor = create_preprocessor(num_features, cat_features)
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_history = {'RF': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 数据预处理
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        # 转换为适合GPU处理的格式
        X_train_gpu = np.ascontiguousarray(X_train_processed.astype('float32'))
        X_test_gpu = np.ascontiguousarray(X_test_processed.astype('float32'))
        y_train_gpu = np.ascontiguousarray(y_train.astype('float32'))

        # 1. 随机森林（使用cuML GPU加速）
        rf = RandomForestClassifier(n_estimators=100, max_depth=16)
        rf.fit(X_train_gpu, y_train_gpu)
        rf_pred = rf.predict(X_test_gpu)
        rf_proba = rf.predict_proba(X_test_gpu)[:,1]
        metrics_rf = evaluate_model(y_test, rf_pred, rf_proba)
        metrics_history['RF'].append(metrics_rf)

    # 计算平均指标
    final_metrics = {}
    for model in metrics_history:
        df_metrics = pd.DataFrame(metrics_history[model])
        final_metrics[model] = {
            'accuracy': f"{df_metrics['accuracy'].mean():.4f} ± {df_metrics['accuracy'].std():.4f}",
            'precision': f"{df_metrics['precision'].mean():.4f} ± {df_metrics['precision'].std():.4f}",
            'recall': f"{df_metrics['recall'].mean():.4f} ± {df_metrics['recall'].std():.4f}",
            'f1': f"{df_metrics['f1'].mean():.4f} ± {df_metrics['f1'].std():.4f}",
            'roc_auc': f"{df_metrics['roc_auc'].mean():.4f} ± {df_metrics['roc_auc'].std():.4f}"
        }
    
    save_metrics_table(final_metrics, 'results/metrics_RF.csv')
    
    # 打印格式化后的结果
    print("\nFinal Metrics with Standard Deviation:")
    print(pd.DataFrame(final_metrics).T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True, default='10G',
                       help='指定要处理的数据集大小 (10G 或 30G)')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    main(args.size)