#!/bin/bash

MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct"
TRAIN_DATASETS="data/edu_safe/train.jsonl"
EVAL_DATASETS="data/edu_safe/eval.jsonl"
TRAIN_TEMPLATE="EduSafe"
OUTPUT_DIR="./outputs/edu_safe_rm"

# 设置环境变量
export WANDB_API_KEY="f61f2a6633aa96f9ed836f4abbf84904ba008485"
export MASTER_PORT=29500

# 启动训练
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_to_text.rm \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
  --eval_datasets ${EVAL_DATASETS} \
  --train_template ${TRAIN_TEMPLATE} \
  --output_dir ${OUTPUT_DIR} \
  --config_file align_anything/configs/train/text_to_text/edu_rm.yaml


