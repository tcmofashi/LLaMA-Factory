#!/usr/bin/env python3
"""
全流程批量处理脚本
从anime.txt读取动画列表，生成问题并回答，生成训练数据

功能：
1. 从anime.txt读取动画列表
2. 对每个动画生成QA问题（如果已存在则跳过）
3. 为问题生成答案
4. 生成训练数据（train_fake.json）

特点：
- 智能跳过已存在的问题文件
- 支持多provider负载均衡（GLM官方 + 硅基流动）
- 自动速率限制和错误重试
- 实时进度保存
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

# 添加react_agent到路径
sys.path.insert(0, str(Path(__file__).parent / "react_agent"))

from batch_process_all_anime import load_anime_list, process_single_anime

# 配置
ANIME_LIST_FILE = "/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt"
OUTPUT_DIR = "/home/tcmofashi/LLaMA-Factory/agent_data"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "batch_progress.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "batch_summary.json")


def save_progress(results: List[Dict]):
    """保存中间进度"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_progress() -> List[Dict]:
    """加载之前的进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_summary(results: List[Dict]):
    """保存最终摘要"""
    summary = {
        "total": len(results),
        "qa_success": sum(1 for r in results if r.get("qa_success")),
        "train_success": sum(1 for r in results if r.get("train_success")),
        "total_questions": sum(r.get("qa_questions", 0) for r in results),
        "total_training_data": sum(r.get("train_questions", 0) for r in results),
        "results": results
    }

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def print_banner():
    """打印横幅"""
    print(f"\n{'='*100}")
    print(f"# 全流程批量处理动画")
    print(f"# 数据源: {ANIME_LIST_FILE}")
    print(f"# 输出目录: {OUTPUT_DIR}")
    print(f"{'='*100}\n")


def print_final_statistics(results: List[Dict], total: int):
    """打印最终统计"""
    print(f"\n{'='*100}")
    print(f"📊 批量处理完成")
    print(f"{'='*100}\n")

    success_count = sum(1 for r in results if r.get("qa_success") and r.get("train_success"))
    failed_count = total - success_count

    print(f"总计: {total} 个动画")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {failed_count} 个")
    print(f"📝 生成问题: {sum(r.get('qa_questions', 0) for r in results)} 个")
    print(f"📚 生成训练数据: {sum(r.get('train_questions', 0) for r in results)} 条")

    # 列出失败的动画
    if failed_count > 0:
        print(f"\n❌ 失败的动画:")
        for r in results:
            if not (r.get("qa_success") and r.get("train_success")):
                print(f"   - {r['anime']}: {r.get('error', '未知错误')}")


def main():
    """主函数"""
    print_banner()

    # 加载动画列表
    anime_list = load_anime_list(ANIME_LIST_FILE)
    total = len(anime_list)

    print(f"📊 共 {total} 个动画需要处理\n")

    # 检查是否有之前的进度
    progress = load_progress()
    if progress:
        print(f"📂 发现已有的进度记录: {len(progress)} 个动画已处理")
        # 找到未完成的动画
        processed_anime = {r['anime'] for r in progress}
        remaining_anime = [a for a in anime_list if a not in processed_anime]
        if remaining_anime:
            print(f"📋 继续处理剩余 {len(remaining_anime)} 个动画\n")
            anime_list = remaining_anime
            results = progress
        else:
            print(f"✅ 所有动画已处理完成！\n")
            print_final_statistics(progress, len(progress))
            return
    else:
        results = []

    # 处理每个动画
    for i, anime_name in enumerate(anime_list, start=len(results) + 1):
        result = process_single_anime(anime_name, i, total)
        results.append(result)

        # 保存中间进度
        save_progress(results)

    # 最终统计
    print_final_statistics(results, total)

    # 保存最终结果
    summary = save_summary(results)
    print(f"\n📄 详细结果已保存到: {SUMMARY_FILE}")
    print(f"📄 进度文件已保存到: {PROGRESS_FILE}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，进度已保存")
        print(f"📄 进度文件: {PROGRESS_FILE}")
        print("💡 提示：重新运行脚本将继续处理剩余动画")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
