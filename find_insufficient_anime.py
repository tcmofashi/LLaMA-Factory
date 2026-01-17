#!/usr/bin/env python3
"""
找出数据不足的作品，删除其文件以便重新生成
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict


def load_anime_list(file_path):
    """加载动漫列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def count_anime_in_train(train_file, anime_list):
    """统计训练数据中每个动漫的数量"""
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    anime_counts = defaultdict(int)

    for item in train_data:
        messages = item.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                matches = re.findall(r'《([^》]+)》', content)
                if matches:
                    anime = max(matches, key=len)
                    # 匹配到动漫列表
                    for full_name in anime_list:
                        if anime in full_name or full_name.split()[0] in anime:
                            anime_counts[full_name] += 1
                            break
                break

    return anime_counts


def main():
    agent_data_dir = Path("/home/tcmofashi/LLaMA-Factory/agent_data")
    train_file = agent_data_dir / "train_merge.json"
    anime_list_file = agent_data_dir / "anime.txt"

    print("🔍 分析数据完整性...\n")

    # 加载动漫列表
    anime_list = load_anime_list(anime_list_file)
    print(f"动漫列表: {len(anime_list)} 个作品\n")

    # 统计训练数据
    anime_counts = count_anime_in_train(train_file, anime_list)

    # 分类
    complete = []  # 5条
    partial = []  # 1-4条
    missing = []  # 0条

    for anime in anime_list:
        count = anime_counts.get(anime, 0)
        if count == 5:
            complete.append(anime)
        elif count > 0:
            partial.append((anime, count))
        else:
            missing.append(anime)

    print("=" * 100)
    print("📊 数据统计")
    print("=" * 100)
    print(f"完整作品 (5条): {len(complete)} 个")
    print(f"部分作品 (1-4条): {len(partial)} 个")
    print(f"缺失作品 (0条): {len(missing)} 个")
    print(f"总对话数: {sum(anime_counts.values())}")
    print(f"预期总数: {len(anime_list) * 5}")
    print(f"缺失总数: {len(anime_list) * 5 - sum(anime_counts.values())}\n")

    # 需要重新处理的作品
    insufficient_anime = [a[0] for a in partial] + missing

    print("=" * 100)
    print(f"📝 需要重新处理的作品: {len(insufficient_anime)} 个")
    print("=" * 100)

    if partial:
        print("\n【部分作品】")
        for anime, count in partial:
            print(f"  - {anime}: {count}/5 条 (缺少 {5-count} 条)")

    if missing:
        print(f"\n【缺失作品】({len(missing)} 个)")
        for anime in missing[:10]:
            print(f"  - {anime}")
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个")

    # 生成需要重新处理的作品列表
    output_file = agent_data_dir / "anime_to_regenerate.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for anime in insufficient_anime:
            f.write(anime + '\n')

    print(f"\n✅ 已保存需要重新处理的作品列表到: {output_file}")

    # 询问是否删除这些作品的旧文件
    print("\n" + "=" * 100)
    print("⚠️  下一步操作建议")
    print("=" * 100)
    print("\n为了重新生成这些作品的数据，建议删除它们的旧文件：")
    print("1. *_questions.jsonl (问题文件)")
    print("2. *_train.json (训练数据文件)")
    print("\n运行以下命令删除旧文件：")
    print(f"  python3 clean_insufficient_anime_files.py")

    return insufficient_anime


if __name__ == "__main__":
    main()
