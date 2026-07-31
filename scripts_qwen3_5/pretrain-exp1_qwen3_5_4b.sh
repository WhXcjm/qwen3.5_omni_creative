 #!/bin/bash
root_dir=/mnt/yWXKUIzKaqvtk0rLm/model_train/

OLD_ARGS=("$@")
set --
source ${root_dir}/Ascend/CANN8.5.0.v2/ascend-toolkit/set_env.sh
source ${root_dir}/Ascend/CANN8.5.0.v2/nnal/atb/set_env.sh
source ${root_dir}/miniconda3/bin/activate
set -- "${OLD_ARGS[@]}"

echo "INPUT: master addr" $1, "nnodes: " $2,  "node_rank: " $3

conda activate ms-mm

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export GLOO_SOCKET_IFNAME=enp66s0f0
export HCCL_SOCKET_IFNAME=enp66s0f0

export CPU_AFFINITY_CONF=1,lazy_bind:0
export TASK_QUEUE_ENABLE=2
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2

DEFAULT_MASTER_ADDR=localhost
DEFAULT_NNODES=16
DEFAULT_NODE_RANK=0
echo "我的 RankID 是: $DEFAULT_NODE_RANK"

datename=$(date +%Y%m%d-%H%M%S)

model_task=pretrain_exp1_qwen3_5-4b

NPUS_PER_NODE=8
MASTER_ADDR=${1:-$DEFAULT_MASTER_ADDR}
MASTER_PORT=6000
NNODES=${2:-$DEFAULT_NNODES}
NODE_RANK=${3:-$DEFAULT_NODE_RANK}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

LOG_DIR=${root_dir}/exp/MindSpeed-LLM/logs_qwen3_5/${model_task}/log_${NODE_RANK}
mkdir -p $LOG_DIR

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    scripts_qwen3_5/configs/qwen3_5_4B_config.yaml \
    2>&1 | tee ${LOG_DIR}/${model_task}_${WORLD_SIZE}P_${datename}.log