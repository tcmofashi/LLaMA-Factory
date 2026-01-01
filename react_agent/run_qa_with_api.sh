#!/bin/bash
# QA Pipeline自动管理API服务版本
# 用法: ./run_qa_with_api.sh "动画名称" [其他参数]

cd /home/tcmofashi/LLaMA-Factory

if [ -z "$1" ]; then
    echo "用法: $0 <动画名称> [其他参数]"
    echo ""
    echo "示例:"
    echo "  $0 \"BanG Dream! It's MyGO!!!!!\""
    echo "  $0 \"白箱 SHIROBAKO\" --max-rounds 5"
    echo "  $0 \"莉兹与青鸟\" --no-api  # 禁用API服务"
    echo "  $0 \"MyGO!!!!!\" --keep-api  # 保持API服务运行"
    echo ""
    echo "参数:"
    echo "  --no-api       禁用API服务，使用本地模式"
    echo "  --keep-api     执行完后不停止API服务"
    echo "  --max-rounds N 最大审批轮数（默认3）"
    echo "  --output-dir DIR 输出目录"
    exit 1
fi

ANIME_NAME="$1"
shift  # 移除第一个参数，剩下的传给Python脚本

# 显示信息
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        QA Pipeline - 自动管理API服务                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📺 动画名称: $ANIME_NAME"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 执行Python脚本
python3 react_agent/qa_pipeline_v2.py \
  --anime "$ANIME_NAME" \
  "$@"

EXIT_CODE=$?

echo ""
echo "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 执行成功！"
else
    echo "❌ 执行失败，退出码: $EXIT_CODE"
fi

exit $EXIT_CODE
