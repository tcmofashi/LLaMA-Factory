#!/usr/bin/env python3
"""
验证rinkoai provider配置和代码修改
"""

import toml
import sys
sys.path.insert(0, 'react_agent')

print("=" * 100)
print("🔍 验证 rinkoai Provider 配置")
print("=" * 100)
print()

# 1. 验证配置文件
print("1️⃣  检查配置文件...")
with open('config.toml', 'r') as f:
    config = toml.load(f)

providers = config.get('providers', {})

# 检查GLM相关的provider
glm_providers = {}
for key, provider_config in providers.items():
    if 'GLM' in provider_config.get('model', ''):
        glm_providers[key] = provider_config

print(f"✅ 找到 {len(glm_providers)} 个GLM provider:")

for key, provider in glm_providers.items():
    is_rinkoai = provider['name'] == 'rinkoai'
    status = "✅" if is_rinkoai else "⚠️ "
    print(f"   {status} [{key}]")
    print(f"      name: {provider['name']}")
    print(f"      model: {provider['model']}")
    print(f"      base_url: {provider['base_url']}")

if all(p['name'] == 'rinkoai' for p in glm_providers.values()):
    print("\n✅ 所有GLM provider都已配置为rinkoai")
else:
    print("\n⚠️  部分GLM provider未使用rinkoai")

print()

# 2. 验证代码修改
print("2️⃣  检查代码修改...")
with open('react_agent/agent.py', 'r') as f:
    agent_code = f.read()

checks = {
    'extract_response_content函数': 'def extract_response_content(message)' in agent_code,
    'reasoning字段支持': 'message.reasoning' in agent_code,
    '第一次调用使用extract': 'extract_response_content(response.choices[0].message)' in agent_code,
}

all_passed = True
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")
    if not passed:
        all_passed = False

print()

# 3. 测试连接
print("3️⃣  测试rinkoai连接...")
from openai import OpenAI

rinkoai_config = providers['primary']
client = OpenAI(
    api_key=rinkoai_config['api_key'],
    base_url=rinkoai_config['base_url']
)

try:
    response = client.chat.completions.create(
        model=rinkoai_config['model'],
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=100
    )

    # 使用extract_response_content提取内容
    from agent import extract_response_content
    content = extract_response_content(response.choices[0].message)

    if content:
        print(f"✅ 连接成功！响应: {content[:100]}...")
    else:
        print("⚠️  连接成功但响应为空")

except Exception as e:
    print(f"❌ 连接失败: {e}")
    all_passed = False

print()

# 4. 总结
print("=" * 100)
if all_passed:
    print("✅ 所有验证通过！rinkoai provider已正确配置")
else:
    print("⚠️  部分验证未通过，请检查配置")
print("=" * 100)
