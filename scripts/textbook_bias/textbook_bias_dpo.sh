#!/usr/bin/env bash

set -e

# 进入项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# 进入 align-anything 目录（项目实际位置）
if [ -d "align-anything" ]; then
  cd align-anything
  ALIGN_ANYTHING_DIR="$(pwd)"
else
  ALIGN_ANYTHING_DIR="$(pwd)"
fi

echo "Project root: $PROJECT_ROOT"
echo "Align-Anything dir: $ALIGN_ANYTHING_DIR"

# 训练配置
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-7B-Instruct}"
TRAIN_DATASETS="${TRAIN_DATASETS:-data/edu_bias_pairs.jsonl}"
TRAIN_TEMPLATE="${TRAIN_TEMPLATE:-TextbookBiasDetection}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/textbook_bias_dpo}"
CONFIG_FILE="${CONFIG_FILE:-align_anything/configs/train/text_to_text/edu_bias_dpo.yaml}"

# 训练模式：deepspeed (默认) | accelerate | single
TRAIN_MODE="${TRAIN_MODE:-deepspeed}"

# 环境变量（可选）
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_key}"  # 如果没有设置则使用占位符
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # 指定使用的 GPU
export MASTER_PORT="${MASTER_PORT:-29500}"

# 激活 conda 环境（如果存在）或虚拟环境
if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
  echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -f "$PROJECT_ROOT/align-anything/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/align-anything/.venv/bin/activate"
  echo "Activated venv: $PROJECT_ROOT/align-anything/.venv"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
  echo "Activated venv: $PROJECT_ROOT/.venv"
fi

# 确保在正确的目录下安装/检查模块
cd "$ALIGN_ANYTHING_DIR"

# 检查 align_anything 模块是否可导入
if ! python -c "import align_anything" 2>/dev/null; then
  echo "Warning: align_anything module not found. Installing..."
  pip install -e .
fi

# 检查 CUDA 是否可用
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null; then
  echo "Warning: CUDA not available. Training may be slow or fail."
  if [ "$TRAIN_MODE" = "deepspeed" ]; then
    echo "Switching to single GPU mode..."
    TRAIN_MODE="single"
  fi
fi

echo "=========================================="
echo "Starting DPO Training for Bias Detection"
echo "=========================================="
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Dataset: ${TRAIN_DATASETS}"
echo "Template: ${TRAIN_TEMPLATE}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Train Mode: ${TRAIN_MODE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "$ALIGN_ANYTHING_DIR/$TRAIN_DATASETS" ]; then
  echo "Error: Training dataset not found at $ALIGN_ANYTHING_DIR/$TRAIN_DATASETS"
  exit 1
fi

# 确保在 align-anything 目录下运行
cd "$ALIGN_ANYTHING_DIR"

# 根据训练模式选择不同的启动方式
if [ "$TRAIN_MODE" = "deepspeed" ]; then
  echo "Using DeepSpeed training mode..."
  if ! command -v deepspeed &> /dev/null; then
    echo "Error: deepspeed not found. Install it or use TRAIN_MODE=single"
    exit 1
  fi
  
  deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_to_text.dpo \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template ${TRAIN_TEMPLATE} \
    --output_dir ${OUTPUT_DIR} \
    --config_file ${CONFIG_FILE}

elif [ "$TRAIN_MODE" = "single" ]; then
  echo "Using single GPU training mode (without DeepSpeed)..."
  export DISABLE_DEEPSPEED_DIST=1
  
  python -m align_anything.trainers.text_to_text.dpo \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template ${TRAIN_TEMPLATE} \
    --output_dir ${OUTPUT_DIR} \
    --config_file ${CONFIG_FILE}

elif [ "$TRAIN_MODE" = "accelerate" ]; then
  echo "Using Accelerate training mode..."
  if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate not found. Install it: pip install accelerate"
    exit 1
  fi
  
  export DISABLE_DEEPSPEED_DIST=1
  accelerate launch \
    --main_process_port ${MASTER_PORT} \
    align_anything/trainers/text_to_text/dpo.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template ${TRAIN_TEMPLATE} \
    --output_dir ${OUTPUT_DIR} \
    --config_file ${CONFIG_FILE}

else
  echo "Error: Unknown TRAIN_MODE: $TRAIN_MODE"
  echo "Available modes: deepspeed, single, accelerate"
  exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="


