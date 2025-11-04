#!/bin/bash
# 检查 DPO 训练所需的核心文件

echo "=== 检查 DPO 训练必需文件 ==="
echo ""

BASE_DIR="align-anything"

# 定义必需文件列表
declare -a REQUIRED_FILES=(
    # 1. 项目根目录和 __init__
    "$BASE_DIR/align_anything/__init__.py"
    "$BASE_DIR/align_anything/configs/__init__.py"
    "$BASE_DIR/align_anything/configs/template.py"
    "$BASE_DIR/align_anything/configs/format_dataset.py"
    "$BASE_DIR/align_anything/configs/format_model.py"
    
    # 2. Utils 模块
    "$BASE_DIR/align_anything/utils/__init__.py"
    "$BASE_DIR/align_anything/utils/template_registry.py"
    "$BASE_DIR/align_anything/utils/tools.py"
    "$BASE_DIR/align_anything/utils/device_utils.py"
    "$BASE_DIR/align_anything/utils/multi_process.py"
    "$BASE_DIR/align_anything/utils/logger.py"
    
    # 3. 模型相关
    "$BASE_DIR/align_anything/models/__init__.py"
    "$BASE_DIR/align_anything/models/pretrained_model.py"
    "$BASE_DIR/align_anything/models/model_registry.py"
    
    # 4. 数据集相关
    "$BASE_DIR/align_anything/datasets/__init__.py"
    "$BASE_DIR/align_anything/datasets/text_to_text/__init__.py"
    "$BASE_DIR/align_anything/datasets/text_to_text/preference.py"
    
    # 5. 训练器相关
    "$BASE_DIR/align_anything/trainers/__init__.py"
    "$BASE_DIR/align_anything/trainers/base/__init__.py"
    "$BASE_DIR/align_anything/trainers/base/supervised_trainer.py"
    "$BASE_DIR/align_anything/trainers/text_to_text/__init__.py"
    "$BASE_DIR/align_anything/trainers/text_to_text/dpo.py"
    
    # 6. 配置文件
    "$BASE_DIR/align_anything/configs/train/text_to_text/dpo.yaml"
    "$BASE_DIR/align_anything/configs/deepspeed/ds_z0_config.json"
    
    # 7. 数据文件
    "$BASE_DIR/data/edu_bias_pairs.jsonl"
    
    # 8. 训练脚本
    "scripts/textbook_bias/textbook_bias_dpo.sh"
)

MISSING_FILES=()
EXISTING_FILES=()

echo "检查文件..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        EXISTING_FILES+=("$file")
        echo "✓ $file"
    else
        MISSING_FILES+=("$file")
        echo "✗ MISSING: $file"
    fi
done

echo ""
echo "=== 统计 ==="
echo "总计: ${#REQUIRED_FILES[@]} 个文件"
echo "存在: ${#EXISTING_FILES[@]} 个文件"
echo "缺失: ${#MISSING_FILES[@]} 个文件"

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "=== 缺失的文件列表 ==="
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "请确保这些文件已同步到 Linux 机器！"
else
    echo ""
    echo "✓ 所有必需文件都存在！"
fi

