#!/bin/bash
# ==========================================================================
# model_eval 示例脚本（通用，可上传）
#
# 从 MindSpeed-MM FSDP2 DCP 检查点做单文本贪心补全（CPU-only，只读）。
# 具体到某台机器/某个实验的个性化脚本请放 model_eval/scripts/（已被 .gitignore 忽略）。
#
# 用法：
#   bash model_eval/run_example.sh [CKPT_DIR] [PROMPT] [OUT_JSON]
# 不传参则用下面的占位默认值。
# ==========================================================================
set -u

# 训练环境（含 torch / mindspeed_mm 依赖），按需替换
source /opt/miniconda3/bin/activate muon

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CKPT="${1:-/path/to/your/ckpt/iter_0005000}"
PROMPT="${2:-Machine learning is a branch of artificial intelligence that}"
OUT="${3:-model_eval/results/example_completion.json}"

mkdir -p "$(dirname "$OUT")"

python model_eval/eval_completion.py \
  --ckpt "$CKPT" \
  --prompt "$PROMPT" \
  --max-new-tokens 128 \
  --out "$OUT"
