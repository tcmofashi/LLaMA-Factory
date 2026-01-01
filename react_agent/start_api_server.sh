#!/bin/bash
# 启动萌娘百科搜索API服务

cd /home/tcmofashi/LLaMA-Factory/react_agent

echo "正在启动萌娘百科搜索API服务..."

# 检查依赖
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "❌ 缺少依赖，正在安装..."
    pip install fastapi uvicorn pydantic
}

# 启动服务
python3 moegirl_search_server.py
