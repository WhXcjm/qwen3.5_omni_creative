#!/bin/bash
# ==========================================================================
# 长测 LongTest 05-new 启动脚本：muon lr4e-4 + dyn + hg0 (all-head)
# config: scripts_qwen3_5/longTest/longTest_05-new_dyn_lr4e-4_hg0.yaml
# 日志: /data/w50062357/weights/qwen3_5_10B-A2B_exp4_gas32_muon_longTest_260810/longTest_05-new_dyn_lr4e-4_hg0_5k-its.log
# 用途: Muon AdamW 参数路由修复后的 task5 配置
# 须从仓库根目录启动: cd /data/w50062357/projects/qwen3.5_omni_creative_bx0522
# ==========================================================================

# source /share/b00585163/miniconda3/bin/activate ms-mm-zj
# source /share/b00585163/cann-8.5.1/ascend-toolkit/set_env.sh
# source /share/b00585163/cann-8.5.1/nnal/atb/set_env.sh

# source /share/x00886543/miniconda3/bin/activate msmm
# source /share/x00886543/Ascend/8.5.0/cann/set_env.sh
# source /share/x00886543/Ascend/8.5.0/nnal/atb/set_env.sh

source /opt/miniconda3/bin/activate muon
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

echo "INPUT: master addr" $1, "nnodes: " $2,  "node_rank: " $3


export HCCL_CONNECT_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0

export CPU_AFFINITY_CONF=1,lazy_bind:0
export TASK_QUEUE_ENABLE=2
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2

export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0

DEFAULT_MASTER_ADDR=80.5.5.133
DEFAULT_NNODES=1
DEFAULT_NODE_RANK=0
# echo "我的 RankID 是: $DEFAULT_NODE_RANK"


NPUS_PER_NODE=8
MASTER_ADDR=${1:-$DEFAULT_MASTER_ADDR}
MASTER_PORT=6667
NNODES=${2:-$DEFAULT_NNODES}
NODE_RANK=${3:-$DEFAULT_NODE_RANK}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    scripts_qwen3_5/longTest/longTest_05-new_dyn_lr4e-4_hg0.yaml
