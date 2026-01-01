#!/bin/bash
# 使用API模式运行QA Pipeline

cd /home/tcmofashi/LLaMA-Factory

echo "=========================================="
echo "启动萌娘百科API服务..."
echo "=========================================="

# 在后台启动API服务
cd react_agent
nohup python3 moegirl_search_server.py > moegirl_api.log 2>&1 &
API_PID=$!
cd ..

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查服务是否启动成功
if curl -s http://localhost:8765/health > /dev/null; then
    echo "✅ API服务启动成功 (PID: $API_PID)"
else
    echo "❌ API服务启动失败，查看日志:"
    cat react_agent/moegirl_api.log
    exit 1
fi

echo ""
echo "=========================================="
echo "使用API模式运行QA Pipeline..."
echo "=========================================="

# 设置环境变量启用API模式
export USE_MOEGIRL_API=true
export MOEGIRL_API_URL=http://localhost:8765

# 运行单个动画测试
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/tcmofashi/LLaMA-Factory/react_agent')

from agent import ReActAgent

agent = ReActAgent(
    max_iterations=20,
    verbose=True
)
agent.initialize_model()

# 测试查询
query = "请为动画《BanG Dream! It's MyGO!!!!!》生成5个高质量的问题"
response = agent.run(query)

print("\n" + "=" * 80)
print("最终回答:")
print("=" * 80)
print(response)
EOF

echo ""
echo "=========================================="
echo "停止API服务..."
echo "=========================================="
kill $API_PID
echo "✅ 服务已停止"

echo ""
echo "提示: 可以使用以下命令持续运行API服务:"
echo "  cd react_agent"
echo "  ./start_api_server.sh"
