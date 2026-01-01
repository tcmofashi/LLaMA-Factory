#!/bin/bash
# 完整数据集生成脚本

echo "################################################################################"
echo "# 完整QA数据集生成"
echo "# 共72个动画需要处理"
echo "################################################################################"
echo ""

# 显示当前配置
echo "📋 当前配置:"
echo "   - Agent模型: GLM-4.7 (智谱AI原生API)"
echo "   - 裁判模型: DS V3 (SiliconFlow)"
echo "   - 每个动画目标: 5个问题"
echo "   - 审批阈值: 60分"
echo ""

# 确认开始
read -p "是否开始处理? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

echo ""
echo "🚀 开始批量处理..."
echo ""

# 运行自动化脚本
python react_agent/auto_qa_workflow.py

echo ""
echo "################################################################################"
echo "# 批量处理完成！"
echo "################################################################################"
echo ""

# 显示统计信息
echo "📊 最终统计:"
if [ -f agent_data/questions.json ]; then
    q_count=$(python -c "import json; print(len(json.load(open('agent_data/questions.json'))))")
    echo "   ✅ questions.json: $q_count 个问题"
fi

if [ -f agent_data/train_fake.json ]; then
    t_count=$(python -c "import json; print(len(json.load(open('agent_data/train_fake.json'))))")
    echo "   ✅ train_fake.json: $t_count 条训练记录"
fi

echo ""
echo "📁 数据文件位置:"
echo "   - agent_data/questions.json (所有问题)"
echo "   - agent_data/train_fake.json (训练数据)"
echo "   - agent_data/answer_record/ (完整对话记录)"
echo ""
