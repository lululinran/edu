# 教育偏见检测训练脚本

本目录包含用于训练和评估教育偏见检测模型的脚本。

## 📁 文件说明

- `textbook_bias_dpo.sh`: DPO训练脚本（主脚本）
- `evaluate_bias_model.sh`: 模型评估脚本
- `test_bias_detection.py`: 模型测试脚本（交互式和批量测试）
- `README.md`: 本文件

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用默认配置（DeepSpeed模式，推荐用于Linux多GPU）
bash scripts/textbook_bias/textbook_bias_dpo.sh

# 使用单GPU模式（适合macOS或没有DeepSpeed的环境）
TRAIN_MODE=single bash scripts/textbook_bias/textbook_bias_dpo.sh

# 使用自定义配置
MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct" \
TRAIN_DATASETS="data/edu_bias_pairs.jsonl" \
OUTPUT_DIR="./outputs/my_bias_model" \
CONFIG_FILE="align_anything/configs/train/text_to_text/edu_bias_dpo.yaml" \
bash scripts/textbook_bias/textbook_bias_dpo.sh
```

### 2. 环境变量配置

```bash
# 必需（如果使用wandb）
export WANDB_API_KEY="your_wandb_key"

# 可选
export CUDA_VISIBLE_DEVICES="0"        # 指定GPU
export TRAIN_MODE="deepspeed"          # 训练模式: deepspeed/single/accelerate
export MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct"
export TRAIN_DATASETS="data/edu_bias_pairs.jsonl"
export OUTPUT_DIR="./outputs/textbook_bias_dpo"
export CONFIG_FILE="align_anything/configs/train/text_to_text/edu_bias_dpo.yaml"
```

### 3. 评估模型

```bash
# 使用默认checkpoint
bash scripts/textbook_bias/evaluate_bias_model.sh

# 指定checkpoint路径
bash scripts/textbook_bias/evaluate_bias_model.sh ./outputs/textbook_bias_dpo/checkpoint-1000
```

### 4. 测试模型

```bash
# 交互式测试
python scripts/textbook_bias/test_bias_detection.py \
  ./outputs/textbook_bias_dpo/checkpoint-1000 \
  --interactive

# 批量测试（使用默认测试用例）
python scripts/textbook_bias/test_bias_detection.py \
  ./outputs/textbook_bias_dpo/checkpoint-1000

# 自定义测试用例
python scripts/textbook_bias/test_bias_detection.py \
  ./outputs/textbook_bias_dpo/checkpoint-1000 \
  --test_cases "测试表述1" "测试表述2"
```

## 📊 训练模式说明

### DeepSpeed模式（默认，推荐用于Linux）

- ✅ 适合多GPU训练
- ✅ 支持ZeRO优化，节省内存
- ✅ 训练速度快
- ❌ 需要安装DeepSpeed
- ❌ macOS上可能不支持（需要mpi4py）

```bash
TRAIN_MODE=deepspeed bash scripts/textbook_bias/textbook_bias_dpo.sh
```

### Single模式（推荐用于macOS或测试）

- ✅ 单GPU训练，不使用DeepSpeed
- ✅ 适合macOS或测试环境
- ✅ 不需要额外依赖
- ❌ 训练速度较慢
- ❌ 不支持多GPU

```bash
TRAIN_MODE=single bash scripts/textbook_bias/textbook_bias_dpo.sh
```

### Accelerate模式

- ✅ 使用HuggingFace Accelerate
- ✅ 适合多GPU训练但不需要DeepSpeed的场景
- ❌ 需要安装accelerate

```bash
TRAIN_MODE=accelerate bash scripts/textbook_bias/textbook_bias_dpo.sh
```

## 📝 数据格式

训练数据应为JSONL格式，每行包含：

```json
{
  "chosen": "无偏见的表述（模型应该学习的方向）",
  "rejected": "有偏见的表述（模型应该避免的方向）",
  "meta": {
    "bias_type": "gender_sexuality"  // 可选：偏见类型
  }
}
```

示例：
```json
{"chosen": "无论性别，逻辑与语言能力都能通过训练提升。", "rejected": "男生天生逻辑更好；女生适合语言类专业。", "meta": {"bias_type": "gender_sexuality"}}
```

## 📦 输出说明

训练完成后，模型会保存在 `OUTPUT_DIR` 目录下：

```
outputs/textbook_bias_dpo/
├── checkpoint-1000/          # 检查点目录
│   ├── config.json          # 模型配置
│   ├── pytorch_model.bin    # 模型权重（如果save_checkpoint=True）
│   ├── tokenizer_config.json
│   └── ...
├── trainer_state.json       # 训练状态
├── training_args.bin        # 训练参数
└── train.log               # 训练日志（如果重定向）
```

## 📈 训练指标

训练过程中会记录以下指标（在wandb/tensorboard中）：

- `train/loss`: DPO损失值（越小越好，应该稳定下降）
- `train/reward_accuracy`: 奖励准确率（应该>0.5，越高越好）
- `train/reward_margin`: 奖励差值（越大越好，表示更好/更差样本区分度）
- `train/better_sample_reward`: 更好样本的奖励
- `train/worse_sample_reward`: 更差样本的奖励
- `train/lr`: 学习率

### 指标解读

- **reward_accuracy > 0.5**: 模型能正确区分更好和更差的样本
- **reward_margin > 0**: 更好样本的奖励高于更差样本
- **loss下降**: 模型在学习偏好

## 🔧 配置说明

### 默认配置 vs 优化配置

| 参数 | 默认配置 (`dpo.yaml`) | 优化配置 (`edu_bias_dpo.yaml`) | 说明 |
|------|---------------------|------------------------------|------|
| `scale_coeff` | 0.1 | 0.5 | DPO缩放系数，控制偏好强度 |
| `learning_rate` | 1e-6 | 5e-6 | 学习率，优化配置略高 |
| `per_device_train_batch_size` | 1 | 2 | 批大小，优化配置更大 |
| `gradient_accumulation_steps` | 1 | 4 | 梯度累积步数 |
| `eval_strategy` | epoch | steps | 评估策略 |
| `eval_interval` | 10 | 100 | 评估间隔（steps） |

### 内存优化选项

如果GPU内存不足，可以在配置文件中启用：

1. **LoRA**（推荐）：
   ```yaml
   lora_cfgs:
     use_lora: True
     r: 16
     lora_alpha: 16
   ```

2. **QLoRA**（更节省内存）：
   ```yaml
   bnb_cfgs:
     use_bnb: True
     load_in_4bit: True
   ```

3. **减小batch size**：
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 8  # 保持有效batch size
   ```

## ❓ 常见问题

### Q: 在macOS上训练失败，提示缺少mpi4py？

**A**: 使用单GPU模式：
```bash
TRAIN_MODE=single bash scripts/textbook_bias/textbook_bias_dpo.sh
```

### Q: GPU内存不足？

**A**: 尝试以下方法：

1. 减小batch size：
   ```bash
   # 在配置文件中或通过环境变量
   export PER_DEVICE_TRAIN_BATCH_SIZE=1
   export GRADIENT_ACCUMULATION_STEPS=8
   ```

2. 启用LoRA（推荐）：
   ```yaml
   # 在配置文件中设置
   lora_cfgs:
     use_lora: True
   ```

3. 使用QLoRA：
   ```yaml
   # 在配置文件中设置
   bnb_cfgs:
     use_bnb: True
     load_in_4bit: True
   ```

### Q: 训练过程中loss不下降？

**A**: 检查以下方面：

1. **学习率是否合适**：尝试调整学习率（1e-6 到 1e-5）
2. **数据质量**：确保chosen/rejected标签正确
3. **scale_coeff**：尝试增加scale_coeff（0.1 到 1.0）
4. **训练步数**：确保训练足够的时间

### Q: reward_accuracy一直很低？

**A**: 可能的原因：

1. **数据问题**：检查chosen/rejected是否标注正确
2. **模型初始化**：确保使用合适的预训练模型
3. **训练不足**：增加训练epochs或steps

### Q: 如何监控训练？

**A**: 

1. **使用wandb**（推荐）：
   ```bash
   export WANDB_API_KEY="your_key"
   # 训练脚本会自动记录到wandb
   ```

2. **查看日志**：
   ```bash
   tail -f outputs/textbook_bias_dpo/train.log
   ```

3. **使用tensorboard**：
   ```yaml
   # 在配置文件中设置
   logger_cfgs:
     log_type: tensorboard
   ```

## 📚 相关资源

- [Align-Anything文档](https://align-anything.readthedocs.io/)
- [DPO论文](https://arxiv.org/abs/2305.18290)
- [训练教程](../align-anything/cookbooks/zh/text_to_text_dpo.ipynb)

## 📄 许可证

本项目遵循Apache License 2.0。
