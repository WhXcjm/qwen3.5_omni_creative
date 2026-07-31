
#!/bin/bash

# 配置参数
FILE_NAME="ip.list"      # 输出文件名
PREFIX="10.42.26."      # IP 前缀
START_NUM=194            # 起始数字
COUNT=16                # 循环生成的次数

# 清空旧文件（如果存在）
> $FILE_NAME

echo "开始生成 IP 列表到 $FILE_NAME..."

for ((i=0; i<COUNT; i++))
do
    # 计算当前末尾数字
    CURRENT_LAST=$((START_NUM + i))
    # 拼接并写入文件
    echo "${PREFIX}${CURRENT_LAST}" >> $FILE_NAME
done
