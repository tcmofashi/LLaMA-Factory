#!/bin/bash
# 全流程批量处理所有动画的便捷脚本

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        全流程批量处理动画 - QA生成 + 训练数据生成        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📺 动画总数: $(wc -l < /home/tcmofashi/LLaMA-Factory/agent_data/anime.txt)"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 运行批量处理
cd /home/tcmofashi/LLaMA-Factory
python3 react_agent/batch_process_all_anime.py

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   批量处理完成                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
