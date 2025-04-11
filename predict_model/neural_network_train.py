import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from utils import (device, load_data, create_preprocessor, 
                   evaluate_model, NeuralNetwork, save_metrics_table)
import argparse

def plot_training_history(histories, model_name):
    """绘制神经网络训练过程图表"""
    plt.figure(figsize=(12, 6))
    
    # 合并所有fold的历史数据
    all_loss = []
    all_acc = []
    for h in histories:
        all_loss.append(h['loss'])      # 修改这里
        all_acc.append(h['accuracy'])   # 修改这里
    
    # 计算均值和标准差
    epochs = range(1, len(all_loss[0])+1)
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    mean_acc = np.mean(all_acc, axis=0)
    std_acc = np.std(all_acc, axis=0)

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_loss, 'b-', label='Training loss')
    plt.fill_between(epochs, 
                    mean_loss - std_loss,
                    mean_loss + std_loss,
                    color='blue', alpha=0.2)
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_acc, 'r-', label='Training accuracy')
    plt.fill_between(epochs,
                    mean_acc - std_acc,
                    mean_acc + std_acc,
                    color='red', alpha=0.2)
    plt.title(f'{model_name} - Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'figures/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    metrics_history = {'NN': []}
    nn_histories = []  # 改为列表存储每个fold的训练历史
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 数据预处理
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # 神经网络
        X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
        X_test_tensor = torch.FloatTensor(X_test_processed).to(device)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)

        # 添加数据验证
        assert torch.all(y_train_tensor >= 0) and torch.all(y_train_tensor <= 1), "标签值必须在0和1之间"
        assert not torch.any(torch.isnan(X_train_tensor)), "训练数据包含NaN值"
        assert not torch.any(torch.isinf(X_train_tensor)), "训练数据包含无限大值"
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        # 初始化模型
        model = NeuralNetwork(X_train_processed.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # 添加学习率调度
        
        # 修改训练循环部分
        fold_history = {'loss': [], 'accuracy': []}  # 记录每个fold的历史
        
        for epoch in range(50):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            fold_history['loss'].append(epoch_loss)
            fold_history['accuracy'].append(epoch_acc)
            
             # 在每个epoch后调整学习率
            scheduler.step(epoch_loss)

            # 评估模型
            model.eval()
            with torch.no_grad():
                nn_proba = model(X_test_tensor).cpu().numpy().flatten()
                nn_pred = (nn_proba > 0.5).astype(int)
                metrics_nn = evaluate_model(y_test, nn_pred, nn_proba)
                metrics_history['NN'].append(metrics_nn)
            
        nn_histories.append(fold_history)  # 将整个fold的历史添加到列表中
        print(f'Fold {fold+1} completed')

    plot_training_history(nn_histories, 'Neural_Network')
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
    
    save_metrics_table(final_metrics, "results/metrics_NN.csv")
    
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