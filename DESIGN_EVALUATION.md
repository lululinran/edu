# 教育偏见检测 DPO 训练设计评估与优化建议

## 📊 当前设计评估

### ✅ 优点

1. **数据格式清晰**：JSONL格式，包含`chosen`（无偏见）和`rejected`（有偏见）配对
2. **模板设计合理**：使用对话格式，符合Qwen2模型的训练方式
3. **配置完整**：覆盖了DPO训练的主要超参数
4. **脚本结构良好**：包含环境检查、路径处理等

### ⚠️ 发现的问题

1. **🔴 严重Bug**：`TextbookBiasTemplate`类中`format_preference_sample`方法被重复定义
   - **已修复**：删除重复定义，保留正确实现

2. **⚠️ 环境兼容性**：在macOS上使用DeepSpeed需要特殊处理
   - DeepSpeed需要mpi4py，macOS上可能不支持
   - 建议：添加单GPU训练模式作为备选

3. **⚠️ 配置优化空间**：
   - `scale_coeff: 0.1` 可能偏小（标准DPO通常0.1-1.0）
   - `learning_rate: 1.e-6` 对于DPO可能偏小
   - 缺少评估数据集配置

4. **⚠️ 缺少的功能**：
   - 没有验证集/测试集
   - 缺少训练后的评估脚本
   - 缺少模型推理示例

## 🎯 预计输出

### 训练输出

训练完成后，会在`./outputs/textbook_bias_dpo/`目录下生成：

```
outputs/textbook_bias_dpo/
├── checkpoint-*/          # 模型检查点
│   ├── config.json       # 模型配置
│   ├── pytorch_model.bin # 模型权重（如果save_checkpoint=True）
│   └── ...
├── trainer_state.json    # 训练状态
├── training_args.bin     # 训练参数
└── train.log            # 训练日志
```

### 训练指标

DPO训练会输出以下指标（在wandb/tensorboard中）：

- `train/loss`: DPO损失值（越小越好）
- `train/reward`: 奖励信号
- `train/better_sample_reward`: 更好样本的奖励
- `train/worse_sample_reward`: 更差样本的奖励
- `train/reward_accuracy`: 奖励准确率（应该>0.5，越高越好）
- `train/reward_margin`: 奖励差值（越大越好，表示更好/更差样本区分度）

### 模型行为预期

训练后的模型应该：
1. **偏好无偏见表述**：在给定相同提示时，模型更倾向于生成无偏见的回答
2. **识别偏见**：能够识别并避免生成有性别、种族等偏见的表述
3. **保持能力**：在偏见检测任务上表现更好，同时保持原有的语言理解能力

## 🚀 优化建议

### 1. 创建专门的训练配置文件

建议创建`edu_bias_dpo.yaml`，针对教育偏见任务优化：

```yaml
train_cfgs:
  epochs: 3
  per_device_train_batch_size: 2  # 增加batch size（如果GPU内存允许）
  gradient_accumulation_steps: 4
  learning_rate: 5.e-6  # 略微提高学习率
  scale_coeff: 0.5  # 增加DPO缩放系数
  eval_strategy: steps
  eval_interval: 100  # 每100步评估一次

data_cfgs:
  train_datasets: "data/edu_bias_pairs.jsonl"
  train_template: "TextbookBiasDetection"
  eval_datasets: "data/edu_bias_pairs_eval.jsonl"  # 添加评估集
  eval_template: "TextbookBiasDetection"

logger_cfgs:
  log_project: "EduAlign-BiasDetection"
  log_run_name: "textbook_bias_dpo"
```

### 2. 改进训练脚本

添加单GPU训练模式作为DeepSpeed的备选：

```bash
# 添加单GPU模式检测
if [ "$USE_DEEPSPEED" != "true" ]; then
  # 使用accelerate或单GPU训练
  python -m align_anything.trainers.text_to_text.dpo \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    ...
else
  # 使用deepspeed
  deepspeed ...
fi
```

### 3. 添加评估脚本

创建`scripts/textbook_bias/evaluate_bias_detection.sh`用于评估模型：

```bash
#!/bin/bash
# 评估训练后的模型在偏见检测任务上的表现
python -m align_anything.serve.text_modal_cli \
  --model_name_or_path ./outputs/textbook_bias_dpo/checkpoint-* \
  --eval_file data/edu_bias_pairs_test.jsonl
```

### 4. 数据增强建议

- 添加更多偏见类型（除性别外，还有种族、年龄、地域等）
- 平衡正负样本比例
- 添加难样本（边界情况）

### 5. 超参数调优建议

| 参数 | 当前值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `scale_coeff` | 0.1 | 0.1-1.0 | DPO缩放系数，控制偏好强度 |
| `learning_rate` | 1e-6 | 5e-6 ~ 1e-5 | 对于7B模型，可以适当提高 |
| `epochs` | 3 | 2-5 | 根据数据量调整 |
| `per_device_train_batch_size` | 1 | 1-4 | 根据GPU内存调整 |

### 6. 评估指标设计

建议添加以下评估指标：
- **偏见识别准确率**：模型能否正确识别偏见表述
- **偏见生成率**：模型生成偏见内容的比例（应该降低）
- **任务保持率**：在非偏见任务上的能力保持

## 📝 使用建议

### 训练前检查清单

- [ ] 确认数据格式正确（JSONL，包含chosen/rejected字段）
- [ ] 检查GPU内存是否足够（7B模型至少需要16GB）
- [ ] 设置WANDB_API_KEY（如果使用wandb）
- [ ] 确认CUDA可用

### 训练后验证

1. **检查训练指标**：
   - `reward_accuracy`应该>0.5并逐渐上升
   - `reward_margin`应该为正且增大
   - `loss`应该稳定下降

2. **人工评估**：
   - 测试模型在偏见检测任务上的表现
   - 检查是否有能力退化

3. **对比测试**：
   - 与原始模型对比偏见识别能力
   - 测试在不同偏见类型上的表现

## 🔗 相关资源

- DPO论文：https://arxiv.org/abs/2305.18290
- Align-Anything文档：https://align-anything.readthedocs.io/
- 训练教程：[text_to_text_dpo.ipynb](../align-anything/cookbooks/zh/text_to_text_dpo.ipynb)

