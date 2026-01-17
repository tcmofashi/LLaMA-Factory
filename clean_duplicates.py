#!/usr/bin/env python3
"""
清理 train_merge.json 中的重复数据
"""

import json
from pathlib import Path


def get_data_signature(item: dict) -> str:
    """生成训练数据的唯一签名"""
    try:
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            return None

        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")

        if not user_content or not assistant_content:
            return None

        # 使用问题+答案的前100字符作为签名
        signature = user_content[:100] + "|||" + assistant_content[:100]
        return signature
    except Exception:
        return None


def clean_duplicates(input_file: str, output_file: str = None):
    """
    清理训练数据中的重复项

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（默认覆盖原文件）
    """
    print(f"📂 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"❌ 文件格式不是列表")
        return

    original_count = len(data)
    print(f"📊 原始数据: {original_count} 条")

    # 去重
    seen_signatures = set()
    unique_data = []
    duplicates = []

    for i, item in enumerate(data):
        sig = get_data_signature(item)

        if not sig:
            # 无法生成签名的数据保留
            unique_data.append(item)
            continue

        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_data.append(item)
        else:
            duplicates.append({
                "index": i,
                "question": item.get("messages", [{}])[1].get("content", "")[:50] if len(item.get("messages", [])) > 1 else "N/A"
            })

    unique_count = len(unique_data)
    duplicate_count = len(duplicates)

    print(f"\n📊 去重统计:")
    print(f"  原始数据: {original_count} 条")
    print(f"  唯一数据: {unique_count} 条")
    print(f"  重复数据: {duplicate_count} 条")

    if duplicate_count > 0:
        print(f"\n❌ 发现 {duplicate_count} 条重复数据:")
        for dup in duplicates[:10]:  # 只显示前10条
            print(f"  - 索引 {dup['index']}: {dup['question']}...")
        if duplicate_count > 10:
            print(f"  ... 还有 {duplicate_count - 10} 条")

    # 保存去重后的数据
    output_path = output_file or input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 去重完成，保存到: {output_path}")

    # 显示文件大小变化
    input_size = Path(input_file).stat().st_size / 1024
    output_size = Path(output_path).stat().st_size / 1024
    print(f"📁 文件大小: {input_size:.1f} KB → {output_size:.1f} KB")

    return unique_data


if __name__ == "__main__":
    import sys

    input_file = "/home/tcmofashi/LLaMA-Factory/agent_data/train_merge.json"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    # 先备份原文件
    backup_file = input_file + ".backup"
    import shutil
    shutil.copy(input_file, backup_file)
    print(f"💾 已备份原文件到: {backup_file}\n")

    # 清理重复数据
    clean_duplicates(input_file)

    print(f"\n💡 提示: 如需恢复，使用备份文件: {backup_file}")
