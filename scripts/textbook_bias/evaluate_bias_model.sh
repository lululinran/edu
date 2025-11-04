#!/usr/bin/env bash

set -e

# 进入项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 进入 align-anything 目录
if [ -d "align-anything" ]; then
  cd align-anything
  ALIGN_ANYTHING_DIR="$(pwd)"
else
  ALIGN_ANYTHING_DIR="$(pwd)"
fi

# 获取checkpoint路径（可选参数）
CHECKPOINT_PATH="${1:-./outputs/textbook_bias_dpo}"
EVAL_DATASET="${EVAL_DATASET:-data/edu_bias_pairs.jsonl}"
EVAL_TEMPLATE="${EVAL_TEMPLATE:-TextbookBiasDetection}"

echo "=========================================="
echo "Evaluating Bias Detection Model"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Eval Dataset: ${EVAL_DATASET}"
echo "Template: ${EVAL_TEMPLATE}"
echo "=========================================="

# 检查checkpoint是否存在
if [ ! -d "$ALIGN_ANYTHING_DIR/$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint not found at $ALIGN_ANYTHING_DIR/$CHECKPOINT_PATH"
  echo "Usage: $0 [checkpoint_path]"
  exit 1
fi

# 检查评估数据集是否存在
if [ ! -f "$ALIGN_ANYTHING_DIR/$EVAL_DATASET" ]; then
  echo "Warning: Eval dataset not found at $ALIGN_ANYTHING_DIR/$EVAL_DATASET"
  echo "Using training dataset for evaluation..."
  EVAL_DATASET="data/edu_bias_pairs.jsonl"
fi

cd "$ALIGN_ANYTHING_DIR"

# 运行评估
python -m align_anything.trainers.text_to_text.dpo \
  --model_name_or_path "${CHECKPOINT_PATH}" \
  --eval_datasets "${EVAL_DATASET}" \
  --eval_template "${EVAL_TEMPLATE}" \
  --config_file align_anything/configs/train/text_to_text/edu_bias_dpo.yaml \
  --do_eval_only

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
