from utils import plot_model_comparison
import pandas as pd

def save_metrics_table():
    """保存指标对比表格"""
    # 读取已有的两个CSV文件
    df_nn = pd.read_csv('results/metrics_NN.csv', index_col=0)
    df_rf = pd.read_csv('results/metrics_RF.csv', index_col=0)
    
    # 合并两个DataFrame
    df = pd.concat([df_nn, df_rf])
    # 保存合并后的CSV
    df.to_csv('results/metrics_comparison.csv')

def main():
    save_metrics_table()
    merged_metrics = pd.read_csv('results/metrics_comparison.csv', index_col=0).to_dict(orient='index')
    plot_model_comparison(merged_metrics)

if __name__ == "__main__":
    main()