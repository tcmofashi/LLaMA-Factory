#!/usr/bin/env python3
"""
检查 anime.txt 中作品与 train_fake.json 中数据的对应关系
找出缺失或数据不足的作品
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def parse_anime_txt(filepath):
    """解析 anime.txt 文件，提取作品列表"""
    animes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 移除序号前缀（如 "1→"）
            line = re.sub(r'^\d+→', '', line)

            # 分割中文名和日文名
            parts = line.split()
            if len(parts) >= 2:
                # 提取中文名（在第一个日文字符之前）
                chinese_name = parts[0]
                for i, part in enumerate(parts[1:], 1):
                    # 检查是否包含日文
                    if any('\u3040' <= c <= '\u309F' or  # 平假名
                           '\u30A0' <= c <= '\u30FF' or  # 片假名
                           '\u4E00' <= c <= '\u9FFF'    # 汉字
                           for c in part):
                        chinese_name = ' '.join(parts[:i])
                        break

                animes.append({
                    'full_line': line,
                    'chinese_name': chinese_name.strip(),
                    'keywords': extract_keywords(chinese_name)
                })

    return animes


def extract_keywords(name):
    """从作品名中提取关键词用于匹配"""
    keywords = []

    # 移除常见的后缀
    name = re.sub(r'\s*(?:第二季|SEASON\s*\d+|后半部分|前半部分|特别篇|特別編|剧场版|劇場版|映画|映像|[:：].+?$)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*第.+季', '', name)
    name = re.sub(r'\s*\[.+\]$', '', name)
    name = re.sub(r'\s*\(.+\)$', '', name)

    # 添加完整名称
    keywords.append(name.strip())

    # 添加短名称（去除副标题）
    short_name = re.split(r'\s+[:：〜～]\s+', name)[0]
    if short_name != name:
        keywords.append(short_name.strip())

    # 添加英文/拼音部分
    english_match = re.search(r'[a-zA-Z]{3,}', name)
    if english_match:
        keywords.append(english_match.group())

    return list(set(keywords))


def count_anime_in_train(train_data, animes):
    """统计每个作品在 train_fake.json 中的出现次数"""
    anime_counts = defaultdict(lambda: {'total': 0, 'samples': []})

    for idx, item in enumerate(train_data):
        content = ''
        for msg in item['messages']:
            content += msg['content'] + ' '

        # 为每个作品匹配
        for anime_info in animes:
            matched = False
            for keyword in anime_info['keywords']:
                if keyword in content:
                    anime_counts[anime_info['chinese_name']]['total'] += 1
                    anime_counts[anime_info['chinese_name']]['samples'].append(idx)
                    matched = True
                    break

    return anime_counts


def main():
    # 文件路径
    anime_txt = Path('/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt')
    train_json = Path('/home/tcmofashi/LLaMA-Factory/agent_data/train_fake.json')

    # 加载数据
    print("正在加载数据...")
    animes = parse_anime_txt(anime_txt)

    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print(f"✓ anime.txt 中的作品数: {len(animes)}")
    print(f"✓ train_fake.json 中的对话数: {len(train_data)}")
    print(f"✓ 预期对话数: {len(animes) * 5} (每个作品5个问题)")
    print()

    # 统计每个作品的对话数
    anime_counts = count_anime_in_train(train_data, animes)

    # 分析结果
    print("=" * 80)
    print("数据统计")
    print("=" * 80)

    missing_animes = []
    insufficient_animes = []

    for i, anime_info in enumerate(animes, 1):
        name = anime_info['chinese_name']
        count = anime_counts[name]['total']

        if count == 0:
            missing_animes.append((i, name, anime_info['full_line']))
        elif count < 5:
            insufficient_animes.append((i, name, count, anime_info['full_line']))

    # 按作品索引排序
    missing_animes.sort(key=lambda x: x[0])
    insufficient_animes.sort(key=lambda x: x[0])

    # 输出结果
    print(f"\n【缺失数据的作品】(0条对话，共{len(missing_animes)}个)")
    print("-" * 80)
    if missing_animes:
        for idx, name, full_line in missing_animes:
            print(f"{idx:3d}. {name}")
            print(f"     完整信息: {full_line}")
    else:
        print("无")

    print(f"\n【数据不足的作品】(1-4条对话，共{len(insufficient_animes)}个)")
    print("-" * 80)
    if insufficient_animes:
        for idx, name, count, full_line in insufficient_animes:
            print(f"{idx:3d}. {name}: {count}条 (缺少{5-count}条)")
            print(f"     完整信息: {full_line}")
    else:
        print("无")

    print(f"\n【数据完整的作品】(5条对话)")
    print("-" * 80)
    complete_count = len(animes) - len(missing_animes) - len(insufficient_animes)
    print(f"共{complete_count}个作品")

    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"总作品数: {len(animes)}")
    print(f"缺失作品数 (0条): {len(missing_animes)}")
    print(f"不足作品数 (1-4条): {len(insufficient_animes)}")
    print(f"完整作品数 (5条): {complete_count}")
    print(f"实际对话数: {len(train_data)}")
    print(f"预期对话数: {len(animes) * 5}")
    print(f"缺失对话总数: {len(animes) * 5 - len(train_data)}")

    # 输出详细数据到文件
    output_file = Path('/home/tcmofashi/LLaMA-Factory/missing_anime_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("缺失数据作品详细报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"【缺失数据的作品】(0条对话，共{len(missing_animes)}个)\n")
        f.write("-" * 80 + "\n")
        for idx, name, full_line in missing_animes:
            f.write(f"{idx:3d}. {name}\n")
            f.write(f"     {full_line}\n\n")

        f.write(f"\n【数据不足的作品】(1-4条对话，共{len(insufficient_animes)}个)\n")
        f.write("-" * 80 + "\n")
        for idx, name, count, full_line in insufficient_animes:
            f.write(f"{idx:3d}. {name}: {count}条 (缺少{5-count}条)\n")
            f.write(f"     {full_line}\n\n")

    print(f"\n详细报告已保存到: {output_file}")


if __name__ == '__main__':
    main()
