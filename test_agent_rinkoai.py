#!/usr/bin/env python3
"""
测试ReAct Agent使用rinkoai provider
"""

import sys
import os

# 设置使用API模式
os.environ['USE_MOEGIRL_API'] = 'true'

sys.path.insert(0, 'react_agent')

from agent import ReActAgent

print("=" * 100)
print("🧪 测试 ReAct Agent + rinkoai Provider")
print("=" * 100)
print()

try:
    # 初始化Agent
    print("📦 初始化 ReAct Agent...")
    agent = ReActAgent(
        max_iterations=3,
        max_new_tokens=500,
        verbose=True
    )

    print()
    print(f"✅ Agent 初始化成功")
    print(f"   加载的 Providers: {len(agent.providers)}")

    for key, provider in agent.providers.items():
        print(f"   - {key}: {provider['name']} ({provider['model']})")

    print()
    print("📝 运行测试查询...")
    print("-" * 100)
    print()

    # 简单测试查询（不需要工具）
    test_query = "你好，请简单介绍一下你自己。"

    response = agent.run(test_query)

    print()
    print("-" * 100)
    print()
    print("✅ 测试成功！")
    print()
    print(f"📝 响应内容:")
    print(f"   {response[:200]}...")
    print()

    print("=" * 100)
    print("✅ rinkoai Provider 集成测试通过")
    print("=" * 100)

    sys.exit(0)

except Exception as e:
    print()
    print("-" * 100)
    print()
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("=" * 100)
    print("❌ rinkoai Provider 集成测试失败")
    print("=" * 100)

    sys.exit(1)
