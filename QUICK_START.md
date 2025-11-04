# 教育偏见检测 - 快速开始指南

本指南将帮助您快速开始使用 Align-Anything 框架训练教育偏见检测模型。

## 📋 前置要求

- Python 3.10+
- CUDA支持的GPU（推荐16GB+显存）
- 已安装 align-anything 框架

## 🚀 快速开始

### 1. 准备数据

确保您的训练数据位于 `align-anything/data/edu_bias_pairs.jsonl`，格式如下：

```json
{"chosen": "无论性别，逻辑与语言能力都能通过训练提升。", "rejected": "男生天生逻辑更好；女生适合语言类专业。", "meta": {"bias_type": "gender_sexuality"}}
```

### 2. 开始训练

#### 方式一：使用默认配置（推荐）

```bash
cd align-anything
bash ../scripts/textbook_bias/textbook_bias_dpo.sh
```

#### 方式二：macOS环境（单GPU模式）

```bash
cd align-anything
TRAIN_MODE=single bash ../scripts/textbook_bias/textbook_bias_dpo.sh
```

#### 方式三：自定义参数

```bash
MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct" \
TRAIN_DATASETS="data/edu_bias_pairs.jsonl" \
OUTPUT_DIR="./outputs/my_bias_model" \
TRAIN_MODE="single" \
WANDB_API_KEY="your_key" \
bash ../scripts/textbook_bias/textbook_bias_dpo.sh
```

### 3. 监控训练

训练过程中可以通过以下方式监控：

```bash
# 查看训练日志
tail -f align-anything/outputs/textbook_bias_dpo/train.log

# 或使用wandb（如果已配置）
# 访问 https://wandb.ai 查看实时指标
```

**关键指标**：
- `train/reward_accuracy` > 0.5（越高越好）
- `train/reward_margin` > 0（越大越好）
- `train/loss` 稳定下降

### 4. 评估模型

```bash
# 评估训练后的模型
bash scripts/textbook_bias/evaluate_bias_model.sh

# 或指定checkpoint
bash scripts/textbook_bias/evaluate_bias_model.sh \
  ./outputs/textbook_bias_dpo/checkpoint-1000
```

### 5. 测试模型

```bash
# 交互式测试
python scripts/textbook_bias/test_bias_detection.py \
  ./outputs/textbook_bias_dpo/checkpoint-1000 \
  --interactive

# 批量测试（使用默认测试用例）
python scripts/textbook_bias/test_bias_detection.py \
  ./outputs/textbook_bias_dpo/checkpoint-1000
```

## 📊 训练模式说明

| 模式 | 适用场景 | 设置方法 |
|------|---------|---------|
| `deepspeed` | 多GPU训练（默认） | `TRAIN_MODE=deepspeed` |
| `single` | 单GPU/测试环境 | `TRAIN_MODE=single` |
| `accelerate` | 多GPU但不用DeepSpeed | `TRAIN_MODE=accelerate` |

## 🔧 配置说明

### 默认配置 vs 优化配置

| 参数 | 默认配置 | 优化配置 | 说明 |
|------|---------|---------|------|
| `scale_coeff` | 0.1 | 0.5 | DPO缩放系数，控制偏好强度 |
| `learning_rate` | 1e-6 | 5e-6 | 学习率，优化配置略高 |
| `per_device_train_batch_size` | 1 | 2 | 批大小，优化配置更大 |
| `gradient_accumulation_steps` | 1 | 4 | 梯度累积步数 |
| `eval_strategy` | epoch | steps | 评估策略 |

脚本默认使用优化配置（`edu_bias_dpo.yaml`），您也可以使用默认配置：

```bash
CONFIG_FILE="align_anything/configs/train/text_to_text/dpo.yaml" \
bash scripts/textbook_bias/textbook_bias_dpo.sh
```

## 🎯 预期输出

训练完成后，您将得到：

1. **模型检查点**：`outputs/textbook_bias_dpo/checkpoint-*/`
   - 包含模型权重、配置等

2. **训练指标**：
   - `reward_accuracy`：奖励准确率（应>0.5）
   - `reward_margin`：奖励差值（应>0）
   - `loss`：DPO损失（应稳定下降）

3. **模型能力**：
   - 更偏好生成无偏见表述
   - 能够识别并避免偏见内容

## ⚠️ 常见问题

### macOS环境问题

如果在macOS上遇到DeepSpeed相关错误，使用单GPU模式：

```bash
TRAIN_MODE=single bash scripts/textbook_bias/textbook_bias_dpo.sh
```

### GPU内存不足

1. 减小batch size
2. 启用LoRA（在配置文件中设置 `lora_cfgs.use_lora: True`）
3. 使用QLoRA（在配置文件中设置 `bnb_cfgs.use_bnb: True`）

### 训练不收敛

1. 检查数据质量
2. 调整学习率（1e-6 到 1e-5）
3. 增加训练epochs
4. 调整scale_coeff（0.1 到 1.0）

## 📚 更多资源

- [详细设计评估文档](DESIGN_EVALUATION.md)
- [脚本使用说明](scripts/textbook_bias/README.md)
- [Align-Anything文档](https://align-anything.readthedocs.io/)
- [DPO论文](https://arxiv.org/abs/2305.18290)

## 🆘 获取帮助

如果遇到问题：

1. 查看 [常见问题](scripts/textbook_bias/README.md#常见问题)
2. 查看训练日志：`outputs/textbook_bias_dpo/train.log`
3. 检查 [GitHub Issues](https://github.com/PKU-Alignment/align-anything/issues)
