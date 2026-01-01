#!/bin/bash
# 安装萌娘百科API服务依赖

echo "检查并安装API服务依赖..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 安装依赖
echo "安装依赖包..."
pip install fastapi uvicorn pydantic requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
echo ""
echo "验证安装..."
python3 << 'EOF'
try:
    import fastapi
    print(f"✅ fastapi: {fastapi.__version__}")
except ImportError:
    print("❌ fastapi 未安装")

try:
    import uvicorn
    print(f"✅ uvicorn: {uvicorn.__version__}")
except ImportError:
    print("❌ uvicorn 未安装")

try:
    import pydantic
    print(f"✅ pydantic: {pydantic.__version__}")
except ImportError:
    print("❌ pydantic 未安装")

try:
    import requests
    print(f"✅ requests: {requests.__version__}")
except ImportError:
    print("❌ requests 未安装")

print("\n✅ 所有依赖已安装完成！")
EOF

echo ""
echo "现在可以启动API服务:"
echo "  cd react_agent"
echo "  ./start_api_server.sh"
