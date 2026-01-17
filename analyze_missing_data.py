#!/usr/bin/env python3
"""
详细分析 train_fake.json 数据缺失问题
对比 batch_progress.json 和 anime.txt
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
                    'name': parts[1].split()[0] if parts[1].split() else parts[1]
                })
    return animes


def load_batch_progress(filepath):
    """加载批处理进度"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_train_fake(filepath):
    """加载训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_anime_name_from_content(content):
    """从对话内容中提取作品名"""
    import re
    # 提取《》中的内容
    matches = re.findall(r'《([^》]+)》', content)
    if matches:
        # 返回最长的匹配（通常是完整作品名）
        return max(matches, key=len)
    return None


def main():
    print("=" * 100)
    print("训练数据缺失分析报告")
    print("=" * 100)
    print()

    # 加载数据
    anime_list = load_anime_list('/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt')
    batch_progress = load_batch_progress('/home/tcmofashi/LLaMA-Factory/agent_data/batch_progress.json')
    train_data = load_train_fake('/home/tcmofashi/LLaMA-Factory/agent_data/train_fake.json')

    print(f"📊 基础统计:")
    print(f"  - anime.txt 中的作品数: {len(anime_list)}")
    print(f"  - batch_progress.json 记录的作品数: {batch_progress['total']}")
    print(f"  - QA生成成功的作品数: {batch_progress['qa_success']}")
    print(f"  - 训练数据生成成功的作品数: {batch_progress['train_success']}")
    print(f"  - train_fake.json 中的对话数: {len(train_data)}")
    print(f"  - 预期对话数 (73作品 × 5问题): {73 * 5}")
    print()

    # 分析 batch_progress 中的结果
    print("=" * 100)
    print("批处理进度分析")
    print("=" * 100)
    print()

    success_list = []
    failed_list = []
    partial_list = []

    for result in batch_progress['results']:
        anime = result['anime']
        train_success = result.get('train_success', False)
        qa_success = result.get('qa_success', False)
        train_questions = result.get('train_questions', 0)

        if train_success and train_questions == 5:
            success_list.append(anime)
        elif train_success and train_questions < 5:
            partial_list.append((anime, train_questions))
        else:
            failed_list.append(anime)

    print(f"✅ 训练数据完整生成 (5条): {len(success_list)}个作品")
    print(f"⚠️  训练数据部分生成 (1-4条): {len(partial_list)}个作品")
    print(f"❌ 训练数据生成失败: {len(failed_list)}个作品")
    print()

    # 找出缺失的作品
    print("=" * 100)
    print("缺失/失败作品详情")
    print("=" * 100)
    print()

    # 从 batch_progress 中提取所有已处理的动漫名
    processed_animes = set()
    for result in batch_progress['results']:
        processed_animes.add(result['anime'])

    # 对比 anime.txt 找出未处理的
    unprocessed = []
    for anime_info in anime_list:
        if anime_info['full_line'] not in processed_animes:
            unprocessed.append(anime_info)

    if unprocessed:
        print(f"【完全未处理的作品】({len(unprocessed)}个):")
        print("-" * 100)
        for anime_info in unprocessed[:10]:  # 只显示前10个
            print(f"  {anime_info['index']:3d}. {anime_info['full_line']}")
        if len(unprocessed) > 10:
            print(f"  ... 还有 {len(unprocessed) - 10} 个")
        print()

    if failed_list:
        print(f"【处理失败的作品】({len(failed_list)}个):")
        print("-" * 100)
        for anime in failed_list[:10]:
            print(f"  - {anime}")
        if len(failed_list) > 10:
            print(f"  ... 还有 {len(failed_list) - 10} 个")
        print()

    if partial_list:
        print(f"【部分处理的作品】({len(partial_list)}个):")
        print("-" * 100)
        for anime, count in sorted(partial_list, key=lambda x: x[1])[:10]:
            print(f"  - {anime}: {count}条 (缺少{5-count}条)")
        if len(partial_list) > 10:
            print(f"  ... 还有 {len(partial_list) - 10} 个")
        print()

    # 分析 train_fake.json 中的数据分布
    print("=" * 100)
    print("train_fake.json 数据分布")
    print("=" * 100)
    print()

    # 统计每个动漫在 train_fake 中的数量
    anime_counts = defaultdict(int)
    for item in train_data:
        # 从第一个 user 消息中提取作品名
        for msg in item['messages']:
            if msg['role'] == 'user':
                name = extract_anime_name_from_content(msg['content'])
                if name:
                    anime_counts[name] += 1
                break

    print(f"涉及的作品数: {len(anime_counts)}")
    print()

    # 找出在 batch_progress 中成功但未在 train_fake 中出现的
    print("=" * 100)
    print("数据不一致分析")
    print("=" * 100)
    print()

    # 收集所有应该有数据的动漫
    expected_animes = set()
    for result in batch_progress['results']:
        if result.get('train_success') and result.get('train_questions', 0) > 0:
            expected_animes.add(result['anime'])

    # 收集实际有数据的动漫（使用简化名称）
    actual_animes = set(anime_counts.keys())

    print(f"应该包含的作品数: {len(expected_animes)}")
    print(f"实际包含的作品数: {len(actual_animes)}")
    print()

    # 诊断问题原因
    print("=" * 100)
    print("问题诊断")
    print("=" * 100)
    print()

    missing_count = 73 * 5 - len(train_data)
    print(f"❗ 缺失对话总数: {missing_count}")
    print()

    print("可能的原因:")
    print("1. 批处理脚本在中途中断，部分作品未完成处理")
    print("2. 某些作品在QA生成阶段失败，但没有训练数据生成")
    print("3. 某些作品的训练数据生成失败，但未记录到 batch_progress")
    print("4. train_fake.json 可能是从不同时间点的数据合并而成")
    print()

    # 检查是否有单独的作品文件
    print("=" * 100)
    print("单独的作品文件")
    print("=" * 100)
    print()

    agent_data_dir = Path('/home/tcmofashi/LLaMA-Factory/agent_data')
    train_files = list(agent_data_dir.glob('*_train.json'))

    print(f"找到 {len(train_files)} 个单独的 *_train.json 文件")
    print()

    if len(train_files) > 0:
        print("示例文件（前10个）:")
        for f in sorted(train_files)[:10]:
            print(f"  - {f.name}")
        print()

        # 计算这些文件的总对话数
        total_in_separate = 0
        for f in train_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    total_in_separate += len(data)
            except:
                pass

        print(f"这些文件中的总对话数: {total_in_separate}")
        print(f"与 train_fake.json 的关系: {'可能是源文件' if total_in_separate >= len(train_data) else '可能是其他数据'}")


if __name__ == '__main__':
    main()
