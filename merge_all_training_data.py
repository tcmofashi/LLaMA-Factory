#!/usr/bin/env python3
"""
重新整合所有训练数据
1. 收集所有 *_train.json 文件
2. 合并成完整的训练数据集
3. 生成缺失报告
"""

import json
from pathlib import Path
from collections import defaultdict


def load_anime_list(filepath):
    """加载 anime.txt 中的作品列表"""
    animes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 提取中文名（去掉序号）
            parts = line.split(None, 1)
            if len(parts) == 2:
                animes.append({
                    'index': len(animes) + 1,
                    'full_line': parts[1],
                    'simple_name': parts[1].split()[0] if parts[1].split() else parts[1]
                })
    return animes


def main():
    print("=" * 100)
    print("重新整合训练数据")
    print("=" * 100)
    print()

    # 加载作品列表
    anime_list = load_anime_list('/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt')
    print(f"✓ 加载了 {len(anime_list)} 个作品")
    print()

    # 扫描所有 *_train.json 文件
    agent_data_dir = Path('/home/tcmofashi/LLaMA-Factory/agent_data')
    train_files = sorted(agent_data_dir.glob('*_train.json'))

    print(f"✓ 找到 {len(train_files)} 个 *_train.json 文件")
    print()

    # 收集所有训练数据
    all_data = []
    file_stats = []

    for train_file in train_files:
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                count = len(data)
                all_data.extend(data)
                file_stats.append({
                    'filename': train_file.name,
                    'anime_name': train_file.stem.replace('_train', ''),
                    'count': count
                })
        except Exception as e:
            print(f"⚠️  无法读取 {train_file.name}: {e}")

    print(f"✓ 总共收集到 {len(all_data)} 条对话")
    print()

    # 统计每个作品的数据量
    anime_data_count = defaultdict(int)
    for item in all_data:
        # 尝试从消息中提取作品名
        for msg in item['messages']:
            if msg['role'] == 'user':
                import re
                matches = re.findall(r'《([^》]+)》', msg['content'])
                if matches:
                    # 使用最长的匹配
                    anime_name = max(matches, key=len)
                    anime_data_count[anime_name] += 1
                    break

    # 分析数据分布
    print("=" * 100)
    print("数据分布统计")
    print("=" * 100)
    print()

    complete_animes = []
    partial_animes = []
    missing_animes = []

    for anime_info in anime_list:
        anime_name = anime_info['simple_name']
        count = anime_data_count.get(anime_name, 0)

        # 也检查完整名称
        for full_name in anime_data_count:
            if anime_name in full_name or full_name in anime_name:
                count = max(count, anime_data_count[full_name])

        if count == 0:
            missing_animes.append((anime_info['index'], anime_info['full_line']))
        elif count < 5:
            partial_animes.append((anime_info['index'], anime_info['full_line'], count))
        else:
            complete_animes.append((anime_info['index'], anime_info['full_line'], count))

    print(f"✅ 完整作品 (5条): {len(complete_animes)}个")
    print(f"⚠️  部分作品 (1-4条): {len(partial_animes)}个")
    print(f"❌ 缺失作品 (0条): {len(missing_animes)}个")
    print()

    # 保存整合后的数据
    output_file = agent_data_dir / 'train_merged.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 整合后的数据已保存到: {output_file}")
    print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # 生成详细报告
    report_file = agent_data_dir / 'merged_data_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("训练数据整合报告\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"总对话数: {len(all_data)}\n")
        f.write(f"预期对话数: {len(anime_list) * 5}\n")
        f.write(f"完成度: {len(all_data) / (len(anime_list) * 5) * 100:.1f}%\n\n")

        f.write("=" * 100 + "\n")
        f.write("【完整作品】(5条)\n")
        f.write("=" * 100 + "\n")
        for idx, name, count in sorted(complete_animes):
            f.write(f"{idx:3d}. {name}: {count}条\n")

        f.write(f"\n总计: {len(complete_animes)}个作品\n\n")

        f.write("=" * 100 + "\n")
        f.write("【部分作品】(1-4条)\n")
        f.write("=" * 100 + "\n")
        for idx, name, count in sorted(partial_animes):
            f.write(f"{idx:3d}. {name}: {count}条 (缺少{5-count}条)\n")

        f.write(f"\n总计: {len(partial_animes)}个作品\n\n")

        f.write("=" * 100 + "\n")
        f.write("【缺失作品】(0条)\n")
        f.write("=" * 100 + "\n")
        for idx, name in sorted(missing_animes):
            f.write(f"{idx:3d}. {name}\n")

        f.write(f"\n总计: {len(missing_animes)}个作品\n\n")

        f.write("=" * 100 + "\n")
        f.write("【文件来源统计】\n")
        f.write("=" * 100 + "\n")
        for stat in sorted(file_stats, key=lambda x: x['count'], reverse=True):
            f.write(f"{stat['filename']}: {stat['count']}条\n")

    print(f"✓ 详细报告已保存到: {report_file}")
    print()

    # 对比原来的 train_fake.json
    print("=" * 100)
    print("与原 train_fake.json 对比")
    print("=" * 100)
    print()

    try:
        with open(agent_data_dir / 'train_fake.json', 'r', encoding='utf-8') as f:
            old_data = json.load(f)

        print(f"原 train_fake.json: {len(old_data)} 条对话")
        print(f"新 train_merged.json: {len(all_data)} 条对话")
        print(f"增加: {len(all_data) - len(old_data)} 条对话")
        print()

        if len(all_data) > len(old_data):
            print("✅ 整合成功！数据已补充")
        else:
            print("⚠️  数据量未增加，可能需要重新生成缺失数据")
    except:
        print("⚠️  无法找到或读取 train_fake.json")


if __name__ == '__main__':
    main()
