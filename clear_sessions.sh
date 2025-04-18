#!/bin/bash

# 检查是否传入了size参数
if [ $# -eq 0 ]; then
    echo "Usage: $0 [10G|30G]"
    exit 1
fi

size=$1

# 验证参数有效性
if [[ "$size" != "10G" && "$size" != "30G" ]]; then
    echo "Error: Invalid size parameter. Must be 10G or 30G"
    exit 1
fi

# 循环从0到8
for i in {0..7}
do
    # 构建会话名称
    session_name="${size}_gpu${i}"
    
    # 检查会话是否存在
    tmux has-session -t ${session_name} 2>/dev/null
    
    if [ $? -eq 0 ]; then
        # 如果会话存在，则杀掉该会话
        tmux kill-session -t ${session_name}
        echo "Killed session: ${session_name}"
    else
        echo "Session does not exist: ${session_name}"
    fi
done