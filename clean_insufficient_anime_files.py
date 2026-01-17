#!/usr/bin/env python3
"""
删除数据不足的作品文件，以便重新生成
"""

import os
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# 添加react_agent到路径
sys.path.insert(0, str(Path(__file__).parent / "react_agent"))
from filename_utils import sanitize_filename


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

    print("=" * 100)
    print("🗑️  清理数据不足的作品文件")
    print("=" * 100)
    print()

    # 加载动漫列表
    anime_list = load_anime_list(anime_list_file)

    # 统计训练数据
    anime_counts = count_anime_in_train(train_file, anime_list)

    # 找出数据不足的作品
    insufficient_anime = []
    for anime in anime_list:
        count = anime_counts.get(anime, 0)
        if count < 5:
            insufficient_anime.append(anime)

    print(f"找到 {len(insufficient_anime)} 个数据不足的作品\n")

    # 删除这些作品的文件
    deleted_files = []

    for anime in insufficient_anime:
        safe_name = sanitize_filename(anime)

        # 可能的文件
        files_to_delete = [
            agent_data_dir / f"{safe_name}_questions.jsonl",
            agent_data_dir / f"{safe_name}_questions.json",
            agent_data_dir / f"{safe_name}_train.json",
            agent_data_dir / f"answer_record/{safe_name}_full.txt",
        ]

        for file_path in files_to_delete:
            if file_path.exists():
                os.remove(file_path)
                deleted_files.append(str(file_path))
                print(f"✅ 已删除: {file_path.name}")

    print()
    print("=" * 100)
    print(f"📊 清理完成")
    print("=" * 100)
    print(f"删除文件数: {len(deleted_files)} 个")
    print()

    # 清空进度文件中这些作品的状态
    progress_file = agent_data_dir / "batch_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        results = progress.get('results', [])
        modified = False

        for result in results:
            anime = result.get('anime')
            if anime in insufficient_anime:
                # 重置状态
                result['qa_success'] = False
                result['train_success'] = False
                result['qa_questions'] = 0
                result['train_questions'] = 0
                result['train_data'] = None
                modified = True
                print(f"🔄 重置进度: {anime}")

        if modified:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print()
            print(f"✅ 已更新进度文件")

    print()
    print("=" * 100)
    print("🚀 下一步：运行批处理脚本")
    print("=" * 100)
    print()
    print("现在可以运行以下命令重新生成数据：")
    print()
    print("  cd /home/tcmofashi/LLaMA-Factory")
    print("  python3 react_agent/batch_process_all_anime.py")
    print()
    print("脚本会自动跳过完整的数据作品，只处理缺失的。")


if __name__ == "__main__":
    main()
