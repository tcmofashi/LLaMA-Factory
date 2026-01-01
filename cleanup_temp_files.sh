#!/bin/bash
# 清理临时文件和测试文件，保留核心功能

echo "🧹 开始清理临时文件..."

# 删除测试文件
echo "删除测试文件..."
rm -f test_*.py
rm -f test_first_3*.py
rm -f test_first_3*.log

# 删除临时分析文件
echo "删除临时分析文件..."
rm -f *_ANALYSIS*.md
rm -f GLM_TIMEOUT_ANALYSIS.md
rm -f TIMEOUT_ANALYSIS_FINAL.md
rm -f full_param_analysis.md
rm -f dataset_analysis_report.md

# 删除临时脚本
echo "删除临时脚本..."
rm -f compare_models.py
rm -f search_moegirl.py
rm -f evaluate_cpt.py
rm -f analyze_token_ratio.py
rm -f auto_qa_workflow.py

# 删除临时环境文件
echo "删除临时环境文件..."
rm -f *_env.sh
rm -f claude_code_env.sh
rm -f data_env.sh

# 删除重复的旧文件（保留react_agent目录中的）
echo "删除根目录中的重复文件..."
rm -f agent.py
rm -f tools.py
rm -f rate_limiter.py
rm -f load_balancer.py
rm -f qa_pipeline.py
rm -f qa_pipeline_v2.py
rm -f generate_training_data.py
rm -f batch_process_all_anime.py
rm -f moegirl_search_server.py
rm -f moegirl_api_client.py
rm -f moegirl_api_manager.py
rm -f example_tools.py
rm -f config.py
rm -f prompt_templates.py
rm -f quality_checker.py

# 删除旧的README和文档
echo "删除旧的文档..."
rm -f AUTO_API_README.md
rm -f BATCH_PROCESSING_README.md
rm -f INTEGRATION_COMPLETE.md
rm -f MOEGIRL_API_README.md
rm -f MOEGIRL_TOOLS_GUIDE.md
rm -f QA_USAGE.md
rm -f SEARCH_SERVICE_SUMMARY.md
rm -f QUICKREF.md
rm -f QUICK_START.md
rm -f COMIC_GIRLS_ANALYSIS.md

# 删除临时数据文件
echo "删除临时数据文件..."
rm -f agent_data/test_3*.json
rm -f agent_data/test_cases_*.json

echo "✅ 清理完成！"
echo ""
echo "📁 保留的核心文件："
echo "   - full_pipeline.py (主流程脚本)"
echo "   - react_agent/ (核心功能模块)"
echo "   - config.toml (配置文件)"
echo "   - agent_data/ (数据目录)"
echo "   - run_*.sh (运行脚本)"
echo "   - README.md (主文档)"
