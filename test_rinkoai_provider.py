#!/usr/bin/env python3
"""
测试rinkoai provider连接
"""

import toml
from openai import OpenAI


def test_rinkoai_provider():
    """测试rinkoai provider"""

    # 加载配置
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    # 获取rinkoai配置
    rinkoai_config = config['providers']['rinkoai']

    print("=" * 100)
    print("🧪 测试 rinkoai Provider")
    print("=" * 100)
    print()

    print(f"Provider配置:")
    print(f"  name: {rinkoai_config['name']}")
    print(f"  model: {rinkoai_config['model']}")
    print(f"  base_url: {rinkoai_config['base_url']}")
    print(f"  api_key: {rinkoai_config['api_key'][:20]}...")
    print()

    try:
        # 创建客户端
        print("🔌 创建 OpenAI 客户端...")
        client = OpenAI(
            api_key=rinkoai_config['api_key'],
            base_url=rinkoai_config['base_url']
        )

        # 发送测试请求
        print("📤 发送测试请求...")
        response = client.chat.completions.create(
            model=rinkoai_config['model'],
            messages=[
                {"role": "user", "content": "你好，请用一句话介绍自己。"}
            ],
            max_tokens=100,
            timeout=30
        )

        # 获取响应
        answer = response.choices[0].message.content

        print()
        print("✅ 连接成功！")
        print()
        print(f"📝 模型响应:")
        print(f"  {answer}")
        print()

        # 检查token使用
        if hasattr(response.usage, 'total_tokens'):
            print(f"📊 Token使用:")
            print(f"  总Token: {response.usage.total_tokens}")

        print()
        print("=" * 100)
        print("✅ rinkoai Provider 测试通过")
        print("=" * 100)

        return True

    except Exception as e:
        print()
        print("❌ 测试失败")
        print(f"   错误: {e}")
        print()
        print("=" * 100)
        print("❌ rinkoai Provider 测试失败")
        print("=" * 100)

        return False


if __name__ == "__main__":
    import sys
    success = test_rinkoai_provider()
    sys.exit(0 if success else 1)
