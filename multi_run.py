import os
import argparse

def run_in_terminal(gpu_id: int, idx: int, file_chunk: list, size: str):  # 修改参数列表
    # 将文件列表转换为字符串
    file_str = ' '.join(file_chunk)
    # print(gpu_id, file_str)
    command = f'''
    tmux new-session -d -s {size}_gpu{gpu_id} "export CUDA_VISIBLE_DEVICES={gpu_id}; python data_process_cudf_partial.py --idx {idx} --file {file_str} --size {size}; exec bash"
    '''
    os.system(command)

def main(args: argparse.Namespace) -> None:
    num_of_gpu = args.gpus
    data_dir = f"{args.size}_data_new"
    
    # 获取所有parquet文件并排序
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    
    # 创建有效的GPU ID列表（跳过ID为4的GPU），并按降序排列
    valid_gpu_ids = sorted([i for i in range(num_of_gpu) if i != 4], reverse=True)
    actual_gpus = len(valid_gpu_ids)
    
    # 分割文件列表，使前面的GPU（ID更大的）获得更多文件
    chunk_size = len(file_list) // actual_gpus
    remainder = len(file_list) % actual_gpus
    
    # 分配文件块，前面的GPU多分一个文件
    file_chunks = []
    start = 0
    for i in range(actual_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        file_chunks.append(file_list[start:end])
        start = end
    
    # 启动终端会话（按GPU ID降序）
    for idx, gpu_id in enumerate(valid_gpu_ids):
        run_in_terminal(gpu_id=gpu_id, idx=idx, file_chunk=file_chunks[idx], size=args.size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理不同规模的数据集')
    parser.add_argument('--size', type=str, choices=['10G', '30G'], required=True,
                       help='指定要处理的数据集大小 (10G 或 30G)')
    parser.add_argument('--gpus', type=int, required=True, default=8,
                       help='并行处理的 GPU 数量')
    args = parser.parse_args()
    main(args)