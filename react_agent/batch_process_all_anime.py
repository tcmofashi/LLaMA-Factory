#!/usr/bin/env python3
"""
全流程批量处理所有动画
对anime.txt中的每个动画执行：
1. 生成问题（QA Pipeline）
2. 生成训练数据（Training Data）
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

# 添加react_agent到路径
sys.path.insert(0, str(Path(__file__).parent))

from qa_pipeline_v2 import generate_questions_for_anime_v2
from generate_training_data import generate_training_data_from_questions

# 配置
ANIME_LIST_FILE = "/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt"
OUTPUT_DIR = "/home/tcmofashi/LLaMA-Factory/agent_data"
MAX_ROUNDS = 3  # QA生成的最大审批轮数


def load_anime_list(file_path: str) -> List[str]:
    """加载动画列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def process_single_anime(anime_name: str, index: int, total: int) -> Dict[str, any]:
    """
    处理单个动画的完整流程
    
    Args:
        anime_name: 动画名称
        index: 当前索引
        total: 总数
    
    Returns:
        处理结果字典
    """
    result = {
        "anime": anime_name,
        "index": index,
        "total": total,
        "qa_success": False,
        "qa_questions": 0,
        "qa_file": None,
        "train_success": False,
        "train_questions": 0,
        "error": None
    }
    
    print(f"\n{'='*100}")
    print(f"🎬 处理动画 [{index}/{total}]: {anime_name}")
    print(f"{'='*100}\n")
    
    # 步骤1: 生成问题（或读取已有问题）
    # 检查是否已存在问题文件
    existing_questions_file = os.path.join(OUTPUT_DIR, f"{anime_name}_questions.jsonl")
    
    if os.path.exists(existing_questions_file):
        # 读取已有问题
        print(f"📝 步骤1: 发现已存在的问题文件")
        print(f"{'-'*100}\n")
        print(f"ℹ️  文件路径: {existing_questions_file}")
        
        # 读取问题数量
        question_count = 0
        with open(existing_questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    question_count += 1
        
        result["qa_success"] = True
        result["qa_questions"] = question_count
        result["qa_file"] = existing_questions_file
        
        print(f"\n✅ 跳过问题生成，使用已有问题: {question_count}个问题\n")
    
    else:
        # 生成新问题
        print(f"📝 步骤1: 生成QA问题")
        print(f"{'-'*100}\n")
        
        try:
            qa_result = generate_questions_for_anime_v2(
                anime_name=anime_name,
                output_dir=OUTPUT_DIR,
                max_rounds=MAX_ROUNDS
            )
            
            result["qa_success"] = True
            result["qa_questions"] = qa_result.get("total_questions", 0)
            result["qa_file"] = qa_result.get("questions_file")
            
            print(f"\n✅ QA生成完成: {result['qa_questions']}个问题")
        
        except Exception as e:
            result["error"] = f"QA生成失败: {str(e)}"
            print(f"❌ QA生成失败: {e}\n")
            return result
    
    # 步骤2: 生成训练数据
    try:
        questions_file = result["qa_file"]
        if not questions_file or not os.path.exists(questions_file):
            result["error"] = "问题文件不存在"
            print(f"❌ 问题文件不存在: {questions_file}\n")
            return result
        
        print(f"\n📚 步骤2: 生成训练数据")
        print(f"{'-'*100}\n")
        
        generate_training_data_from_questions(
            questions_file=questions_file,
            output_dir=OUTPUT_DIR
        )
        
        result["train_success"] = True
        # 统计生成的问题数量（从train_fake.json读取）
        train_file = os.path.join(OUTPUT_DIR, "train_fake.json")
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    result["train_questions"] = len(data)
        
        print(f"\n✅ 训练数据生成完成")
    
    except Exception as e:
        result["error"] = f"训练数据生成失败: {str(e)}"
        print(f"❌ 训练数据生成失败: {e}\n")
    
    return result


def save_summary(results: List[Dict], output_file: str):
    """保存处理摘要"""
    summary = {
        "total": len(results),
        "qa_success": sum(1 for r in results if r["qa_success"]),
        "train_success": sum(1 for r in results if r["train_success"]),
        "total_questions": sum(r.get("qa_questions", 0) for r in results),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def main():
    """主函数"""
    print(f"\n{'#'*100}")
    print(f"# 全流程批量处理动画")
    print(f"# 数据源: {ANIME_LIST_FILE}")
    print(f"# 输出目录: {OUTPUT_DIR}")
    print(f"{'#'*100}\n")
    
    # 加载动画列表
    anime_list = load_anime_list(ANIME_LIST_FILE)
    total = len(anime_list)
    
    print(f"📊 共 {total} 个动画需要处理\n")
    
    # 处理每个动画
    results = []
    
    for i, anime_name in enumerate(anime_list, 1):
        result = process_single_anime(anime_name, i, total)
        results.append(result)
        
        # 保存中间结果
        summary_file = os.path.join(OUTPUT_DIR, "batch_progress.json")
        save_summary(results, summary_file)
    
    # 最终统计
    print(f"\n{'='*100}")
    print(f"📊 批量处理完成")
    print(f"{'='*100}\n")
    
    success_count = sum(1 for r in results if r["qa_success"] and r["train_success"])
    failed_count = total - success_count
    
    print(f"总计: {total} 个动画")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {failed_count} 个")
    print(f"📝 生成问题: {sum(r.get('qa_questions', 0) for r in results)} 个")
    
    # 保存最终结果
    final_summary = save_summary(results, os.path.join(OUTPUT_DIR, "batch_summary.json"))
    print(f"\n📄 详细结果已保存到: {os.path.join(OUTPUT_DIR, 'batch_summary.json')}")
    
    # 列出失败的动画
    if failed_count > 0:
        print(f"\n❌ 失败的动画:")
        for r in results:
            if not (r["qa_success"] and r["train_success"]):
                print(f"   - {r['anime']}: {r.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
