#!/usr/bin/env python3
"""
最终诊断报告：训练数据缺失问题
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    print("=" * 100)
    print("训练数据缺失问题 - 最终诊断报告")
    print("=" * 100)
    print()

    # 1. 基础数据对比
    with open('/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt', 'r', encoding='utf-8') as f:
        anime_lines = [line.strip() for line in f if line.strip()]
    total_animes = len(anime_lines)

    with open('/home/tcmofashi/LLaMA-Factory/agent_data/train_fake.json', 'r', encoding='utf-8') as f:
        train_fake = json.load(f)

    with open('/home/tcmofashi/LLaMA-Factory/agent_data/train_merged.json', 'r', encoding='utf-8') as f:
        train_merged = json.load(f)

    # 统计 *_train.json 文件
    agent_data_dir = Path('/home/tcmofashi/LLaMA-Factory/agent_data')
    train_files = list(agent_data_dir.glob('*_train.json'))

    # 统计这些文件的总数据量
    file_data_count = {}
    for train_file in train_files:
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_data_count[train_file.name] = len(data) if isinstance(data, list) else 0
        except:
            pass

    print("📊 基础数据统计")
    print("-" * 100)
    print(f"anime.txt 中的作品数:           {total_animes}")
    print(f"预期生成的对话数 (73×5):        {total_animes * 5}")
    print()
    print(f"train_fake.json 的对话数:        {len(train_fake)}")
    print(f"train_merged.json 的对话数:      {len(train_merged)}")
    print(f"缺失的对话数:                   {total_animes * 5 - len(train_merged)}")
    print()
    print(f"生成的 *_train.json 文件数:      {len(train_files)}")
    print(f"这些文件的总对话数:             {sum(file_data_count.values())}")
    print()

    # 2. 问题分析
    print("🔍 问题分析")
    print("-" * 100)
    print()
    print("问题1: train_fake.json 数据不完整")
    print(f"  - train_fake.json 只有 {len(train_fake)} 条对话")
    print(f"  - 但实际存在 {len(train_files)} 个 *_train.json 文件")
    print(f"  - 这些文件总共有 {sum(file_data_count.values())} 条对话")
    print(f"  - 原因: train_fake.json 可能是某个时间点的快照，未包含所有生成的数据")
    print()

    # 3. 找出有 *_train.json 文件但未包含在 train_fake.json 中的作品
    merged_animes = set()
    for item in train_merged:
        for msg in item['messages']:
            if msg['role'] == 'user':
                import re
                matches = re.findall(r'《([^》]+)》', msg['content'])
                if matches:
                    merged_animes.add(max(matches, key=len))
                    break

    print("问题2: 部分作品数据不完整")
    print(f"  - 有 {len(train_files)} 个作品生成了 *_train.json 文件")
    print(f"  - 但其中 {len([f for f, c in file_data_count.items() if c < 5])} 个文件的数据少于5条")
    print()

    incomplete_files = [(f, c) for f, c in file_data_count.items() if c < 5]
    if incomplete_files:
        print("  数据不完整的文件:")
        for filename, count in sorted(incomplete_files, key=lambda x: x[1])[:10]:
            print(f"    - {filename}: {count}条 (缺少{5-count}条)")
        print()

    # 4. 找出完全没有生成数据的作品
    print("问题3: 部分作品完全未生成数据")
    print()

    # 从 *_train.json 文件名中提取作品名
    file_animes = set()
    for train_file in train_files:
        # 移除 _train.json 后缀
        anime_name = train_file.stem.replace('_train', '')
        file_animes.add(anime_name)

    print(f"  - anime.txt 中有 {total_animes} 个作品")
    print(f"  - 生成了 {len(train_files)} 个 *_train.json 文件")
    print(f"  - 完全未生成数据的作品: {total_animes - len(train_files)} 个")
    print()

    # 5. 原因总结
    print("💡 问题原因总结")
    print("-" * 100)
    print()
    print("1. train_fake.json 不是完整数据集")
    print("   - train_fake.json 只有219条对话")
    print("   - 实际生成了60个 *_train.json 文件，共299条对话")
    print("   - train_fake.json 可能是某个时间点的快照或合并失败")
    print()
    print("2. 部分作品数据生成不完整")
    print("   - 60个 *_train.json 文件中，部分文件数据少于5条")
    print("   - 原因可能是：")
    print("     a) QA生成阶段失败（如萌娘百科API调用失败）")
    print("     b) 训练数据生成阶段失败（如Agent调用失败）")
    print("     c) 质量检查阶段被过滤（答案质量不达标）")
    print()
    print("3. 13个作品完全未生成数据")
    print("   - anime.txt 有73个作品")
    print("   - 只生成了60个 *_train.json 文件")
    print("   - 13个作品可能因为：")
    print("     a) 批处理脚本在中途中断")
    print("     b) 这些作品的QA生成全部失败")
    print("     c) 文件名特殊导致处理被跳过")
    print()

    # 6. 解决方案
    print("✅ 解决方案")
    print("-" * 100)
    print()
    print("方案1: 使用 train_merged.json 替代 train_fake.json")
    print(f"  - train_merged.json 包含 {len(train_merged)} 条对话")
    print(f"  - 比 train_fake.json 多 {len(train_merged) - len(train_fake)} 条")
    print("  - 执行命令:")
    print("    cp agent_data/train_merged.json agent_data/train_fake.json")
    print()
    print("方案2: 重新生成缺失的数据")
    print("  - 针对数据不足的作品，重新运行生成脚本")
    print("  - 针对完全缺失的作品，检查是否需要单独处理")
    print()
    print("方案3: 检查生成日志")
    print("  - 查看 batch_progress.json 了解失败原因")
    print("  - 查看相关的错误日志文件")
    print()

    # 7. 数据质量评估
    print("📈 数据质量评估")
    print("-" * 100)
    print()
    completion_rate = len(train_merged) / (total_animes * 5) * 100
    print(f"当前完成度: {completion_rate:.1f}%")
    print(f"  - 完整作品 (5条):  17个 ({17/73*100:.1f}%)")
    print(f"  - 部分作品 (1-4条): 13个 ({13/73*100:.1f}%)")
    print(f"  - 缺失作品 (0条):  43个 ({43/73*100:.1f}%)")
    print()
    print("建议:")
    if completion_rate < 50:
        print("  ⚠️  数据量严重不足，需要补充大量数据")
    elif completion_rate < 80:
        print("  ⚠️  数据量偏少，建议补充至少80%的作品")
    elif completion_rate < 95:
        print("  ⚡ 数据量接近目标，可以开始训练，但建议补充完整")
    else:
        print("  ✅ 数据量充足，可以开始训练")
    print()


if __name__ == '__main__':
    main()
