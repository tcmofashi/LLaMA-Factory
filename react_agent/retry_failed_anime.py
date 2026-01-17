#!/usr/bin/env python3
"""
自动识别并重新生成失败的动画
支持单独重试问题生成、训练数据生成，或完整流程
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Set

# 添加react_agent到路径
sys.path.insert(0, str(Path(__file__).parent))

from qa_pipeline_v2 import generate_questions_for_anime_v2
from generate_training_data import generate_training_data_from_questions
from filename_utils import sanitize_filename


class AnimeRetryManager:
    """动画重试管理器"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.anime_list_file = os.path.join(data_dir, "anime.txt")
        self.summary_file = os.path.join(data_dir, "batch_summary.json")

    def _questions_file(self, anime_name: str) -> str:
        return os.path.join(self.data_dir, f"{sanitize_filename(anime_name)}_questions.jsonl")

    def _train_file(self, anime_name: str) -> str:
        return os.path.join(self.data_dir, f"{sanitize_filename(anime_name)}_train.json")

    def load_anime_list(self) -> List[str]:
        """加载动画列表"""
        with open(self.anime_list_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_batch_summary(self) -> Dict:
        """加载批量处理摘要"""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"total": 0, "results": []}

    def identify_missing_questions(self) -> List[str]:
        """识别缺少问题文件的动画"""
        print(f"\n{'='*100}")
        print(f"🔍 识别缺少问题文件的动画")
        print(f"{'='*100}\n")

        anime_list = self.load_anime_list()
        missing = []

        for anime_name in anime_list:
            questions_file = self._questions_file(anime_name)

            if not os.path.exists(questions_file):
                missing.append(anime_name)
                print(f"  ❌ {anime_name}")
            else:
                # 检查文件是否为空或有问题
                with open(questions_file, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    if len(lines) == 0:
                        missing.append(anime_name)
                        print(f"  ⚠️  {anime_name} (问题文件为空)")
                    elif len(lines) < 5:
                        missing.append(anime_name)
                        print(f"  ⚠️  {anime_name} (问题数量不足: {len(lines)}/5)")
                    else:
                        print(f"  ✅ {anime_name}")

        print(f"\n📊 统计：")
        print(f"  总动画数: {len(anime_list)}")
        print(f"  缺少问题: {len(missing)} 个\n")

        return missing

    def identify_missing_train_data(self) -> List[str]:
        """识别缺少训练数据的动画"""
        print(f"\n{'='*100}")
        print(f"🔍 识别缺少训练数据的动画")
        print(f"{'='*100}\n")

        anime_list = self.load_anime_list()
        missing = []

        for anime_name in anime_list:
            # 首先检查问题文件是否存在
            questions_file = self._questions_file(anime_name)
            if not os.path.exists(questions_file):
                print(f"  ⏭️  {anime_name} (跳过：无问题文件)")
                continue

            # 检查训练数据文件
            train_file = self._train_file(anime_name)

            if not os.path.exists(train_file):
                missing.append(anime_name)
                print(f"  ❌ {anime_name}")
            else:
                # 检查训练数据是否完整
                with open(train_file, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                    if len(train_data) < 5:
                        missing.append(anime_name)
                        print(f"  ⚠️  {anime_name} (训练数据不足: {len(train_data)}/5)")
                    else:
                        print(f"  ✅ {anime_name}")

        print(f"\n📊 统计：")
        print(f"  总动画数: {len(anime_list)}")
        print(f"  有问题文件: {len(anime_list) - len([a for a in anime_list if not os.path.exists(self._questions_file(a))])}")
        print(f"  缺少训练数据: {len(missing)} 个\n")

        return missing

    def identify_failed_from_summary(self) -> Dict[str, List[str]]:
        """从batch_summary中识别失败的动画"""
        print(f"\n{'='*100}")
        print(f"🔍 从batch_summary.json中识别失败的动画")
        print(f"{'='*100}\n")

        summary = self.load_batch_summary()

        if not summary.get("results"):
            print(f"  ⚠️  batch_summary.json不存在或为空\n")
            return {"qa_failed": [], "train_failed": []}

        qa_failed = []
        train_failed = []

        for result in summary["results"]:
            anime_name = result.get("anime", "")

            if not result.get("qa_success", False):
                qa_failed.append(anime_name)
                print(f"  ❌ 问题生成失败: {anime_name}")

            if result.get("qa_success", False) and not result.get("train_success", False):
                train_failed.append(anime_name)
                print(f"  ❌ 训练数据生成失败: {anime_name}")

        print(f"\n📊 统计：")
        print(f"  问题生成失败: {len(qa_failed)} 个")
        print(f"  训练数据生成失败: {len(train_failed)} 个\n")

        return {"qa_failed": qa_failed, "train_failed": train_failed}

    def retry_questions(self, anime_list: List[str], max_rounds: int = 5) -> Dict:
        """重新生成问题"""
        print(f"\n{'='*100}")
        print(f"🔄 重新生成问题")
        print(f"{'='*100}\n")

        results = {
            "total": len(anime_list),
            "success": 0,
            "failed": 0,
            "details": []
        }

        for i, anime_name in enumerate(anime_list, 1):
            print(f"\n[{i}/{len(anime_list)}] 处理: {anime_name}")
            print(f"{'-'*100}\n")

            try:
                qa_result = generate_questions_for_anime_v2(
                    anime_name=anime_name,
                    output_dir=self.data_dir,
                    max_rounds=max_rounds
                )

                if qa_result.get("success") and qa_result.get("total_questions", 0) >= 5:
                    results["success"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "success",
                        "questions": qa_result.get("total_questions")
                    })
                    print(f"✅ 成功: {anime_name}\n")
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "failed",
                        "error": qa_result.get("error", "未知错误")
                    })
                    print(f"❌ 失败: {anime_name}\n")

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "anime": anime_name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"❌ 异常: {anime_name} - {e}\n")

        return results

    def retry_train_data(self, anime_list: List[str], max_workers: int = 5) -> Dict:
        """重新生成训练数据"""
        print(f"\n{'='*100}")
        print(f"🔄 重新生成训练数据")
        print(f"{'='*100}\n")

        results = {
            "total": len(anime_list),
            "success": 0,
            "failed": 0,
            "details": []
        }

        for i, anime_name in enumerate(anime_list, 1):
            print(f"\n[{i}/{len(anime_list)}] 处理: {anime_name}")
            print(f"{'-'*100}\n")

            questions_file = self._questions_file(anime_name)

            if not os.path.exists(questions_file):
                results["failed"] += 1
                results["details"].append({
                    "anime": anime_name,
                    "status": "no_questions",
                    "error": "问题文件不存在"
                })
                print(f"❌ 跳过: 问题文件不存在\n")
                continue

            try:
                train_data = generate_training_data_from_questions(
                    questions_file=questions_file,
                    output_dir=self.data_dir,
                    max_workers=max_workers,
                    anime_name=anime_name
                )

                if train_data and len(train_data) >= 5:
                    results["success"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "success",
                        "train_data": len(train_data)
                    })
                    print(f"✅ 成功: {anime_name} ({len(train_data)} 条)\n")
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "insufficient_data",
                        "error": f"训练数据不足: {len(train_data) if train_data else 0}/5"
                    })
                    print(f"❌ 失败: {anime_name} (训练数据不足)\n")

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "anime": anime_name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"❌ 异常: {anime_name} - {e}\n")

        return results

    def retry_full_pipeline(self, anime_list: List[str], max_rounds: int = 5, max_workers: int = 5) -> Dict:
        """重新生成完整流程（问题 + 训练数据）"""
        print(f"\n{'='*100}")
        print(f"🔄 重新生成完整流程")
        print(f"{'='*100}\n")

        results = {
            "total": len(anime_list),
            "success": 0,
            "qa_failed": 0,
            "train_failed": 0,
            "details": []
        }

        for i, anime_name in enumerate(anime_list, 1):
            print(f"\n{'='*100}")
            print(f"[{i}/{len(anime_list)}] 完整流程: {anime_name}")
            print(f"{'='*100}\n")

            # 步骤1: 生成问题
            try:
                qa_result = generate_questions_for_anime_v2(
                    anime_name=anime_name,
                    output_dir=self.data_dir,
                    max_rounds=max_rounds
                )

                if not qa_result.get("success") or qa_result.get("total_questions", 0) < 5:
                    results["qa_failed"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "qa_failed",
                        "error": qa_result.get("error", "问题生成失败")
                    })
                    print(f"❌ 问题生成失败: {anime_name}\n")
                    continue

                print(f"✅ 问题生成成功: {anime_name}\n")

            except Exception as e:
                results["qa_failed"] += 1
                results["details"].append({
                    "anime": anime_name,
                    "status": "qa_error",
                    "error": str(e)
                })
                print(f"❌ 问题生成异常: {anime_name} - {e}\n")
                continue

            # 步骤2: 生成训练数据
            questions_file = self._questions_file(anime_name)

            try:
                train_data = generate_training_data_from_questions(
                    questions_file=questions_file,
                    output_dir=self.data_dir,
                    max_workers=max_workers,
                    anime_name=anime_name
                )

                if train_data and len(train_data) >= 5:
                    results["success"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "success",
                        "questions": qa_result.get("total_questions"),
                        "train_data": len(train_data)
                    })
                    print(f"✅ 完整流程成功: {anime_name}\n")
                else:
                    results["train_failed"] += 1
                    results["details"].append({
                        "anime": anime_name,
                        "status": "train_failed",
                        "error": f"训练数据不足: {len(train_data) if train_data else 0}/5"
                    })
                    print(f"❌ 训练数据生成失败: {anime_name}\n")

            except Exception as e:
                results["train_failed"] += 1
                results["details"].append({
                    "anime": anime_name,
                    "status": "train_error",
                    "error": str(e)
                })
                print(f"❌ 训练数据生成异常: {anime_name} - {e}\n")

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="自动识别并重新生成失败的动画")
    parser.add_argument("--data-dir", type=str,
                       default="/home/tcmofashi/LLaMA-Factory/agent_data",
                       help="数据目录（默认: agent_data）")
    parser.add_argument("--mode", type=str, choices=["auto", "questions", "train", "full"],
                       default="auto",
                       help="重试模式: auto(自动识别), questions(仅问题), train(仅训练数据), full(完整流程)")
    parser.add_argument("--max-rounds", type=int, default=5,
                       help="QA生成的最大审批轮数（默认: 5）")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="训练数据生成的最大并发数（默认: 5）")

    args = parser.parse_args()

    print(f"\n{'#'*100}")
    print(f"# 动画重试管理器")
    print(f"# 数据目录: {args.data_dir}")
    print(f"# 重试模式: {args.mode}")
    print(f"{'#'*100}")

    manager = AnimeRetryManager(args.data_dir)

    if args.mode == "auto":
        # 自动识别并处理
        print(f"\n🤖 自动模式：识别所有失败的动画并尝试修复\n")

        # 识别缺少问题的动画
        missing_questions = manager.identify_missing_questions()

        # 识别缺少训练数据的动画
        missing_train_data = manager.identify_missing_train_data()

        # 从summary中识别失败
        failed_from_summary = manager.identify_failed_from_summary()

        # 合并结果
        qa_failed = list(set(missing_questions + failed_from_summary.get("qa_failed", [])))
        train_failed = list(set(missing_train_data + failed_from_summary.get("train_failed", [])))

        # 去重（train_failed中的某些可能也在qa_failed中）
        train_failed = [a for a in train_failed if a not in qa_failed]

        print(f"\n📋 重试计划：")
        print(f"  需要重新生成问题: {len(qa_failed)} 个")
        print(f"  需要重新生成训练数据: {len(train_failed)} 个")

        if qa_failed:
            print(f"\n🔄 重新生成问题...")
            qa_results = manager.retry_questions(qa_failed, args.max_rounds)

        if train_failed:
            print(f"\n🔄 重新生成训练数据...")
            train_results = manager.retry_train_data(train_failed, args.max_workers)

        # 保存结果
        retry_result = {
            "mode": "auto",
            "qa_retry": qa_results if qa_failed else {"total": 0, "success": 0, "failed": 0},
            "train_retry": train_results if train_failed else {"total": 0, "success": 0, "failed": 0}
        }

    elif args.mode == "questions":
        # 仅重新生成问题
        missing_questions = manager.identify_missing_questions()
        failed_from_summary = manager.identify_failed_from_summary()
        qa_failed = list(set(missing_questions + failed_from_summary.get("qa_failed", [])))

        print(f"\n🔄 重新生成问题...")
        retry_result = manager.retry_questions(qa_failed, args.max_rounds)

    elif args.mode == "train":
        # 仅重新生成训练数据
        missing_train_data = manager.identify_missing_train_data()
        failed_from_summary = manager.identify_failed_from_summary()
        train_failed = list(set(missing_train_data + failed_from_summary.get("train_failed", [])))

        print(f"\n🔄 重新生成训练数据...")
        retry_result = manager.retry_train_data(train_failed, args.max_workers)

    elif args.mode == "full":
        # 完整流程
        missing_questions = manager.identify_missing_questions()
        failed_from_summary = manager.identify_failed_from_summary()
        qa_failed = list(set(missing_questions + failed_from_summary.get("qa_failed", [])))

        print(f"\n🔄 重新生成完整流程...")
        retry_result = manager.retry_full_pipeline(qa_failed, args.max_rounds, args.max_workers)

    # 保存重试结果
    result_file = os.path.join(args.data_dir, "retry_result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(retry_result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*100}")
    print(f"📄 重试结果已保存到: {result_file}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
