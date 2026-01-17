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
from typing import List, Dict, Optional

# 添加react_agent到路径
sys.path.insert(0, str(Path(__file__).parent))

from qa_pipeline_v2 import generate_questions_for_anime_v2
from generate_training_data_improved import generate_training_data_from_questions
from filename_utils import sanitize_filename

# 配置
ANIME_LIST_FILE = "/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt"
OUTPUT_DIR = "/home/tcmofashi/LLaMA-Factory/agent_data"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "batch_progress.json")
MERGE_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_merge.json")

# 问题/训练数据生成参数
MAX_QA_ROUNDS = 50  # 对齐最新QA流水线
MAX_QA_ATTEMPTS = 3  # QA整体重试次数（含整组重跑）
MAX_TRAIN_ATTEMPTS = 3  # 训练数据重试次数
REQUIRED_QUESTIONS = 5
REQUIRED_TRAIN_DATA = 5

# 训练数据生成并发与重试
TRAIN_MAX_WORKERS = 3
TRAIN_MAX_RETRIES_PER_Q = 3


def load_anime_list(file_path: str) -> List[str]:
    """加载动画列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_progress(file_path: str) -> Dict[str, Dict]:
    """加载进度文件，支持断点续跑"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results") if isinstance(data, dict) else data
            if not results:
                return {}
            return {r.get("anime"): r for r in results if r.get("anime")}
    except Exception:
        return {}


def count_questions(file_path: str) -> int:
    """统计问题文件行数"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_train_items(file_path: str) -> int:
    """统计训练数据条数（JSON列表）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
    except Exception:
        return 0
    return 0


def load_train_data(file_path: str) -> List[Dict]:
    """加载训练数据列表，失败返回空列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def validate_questions_file(file_path: str, required: int) -> bool:
    """校验问题文件是否存在、行数足够且格式包含question/type"""
    if not os.path.exists(file_path):
        return False
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict) or "question" not in obj or "type" not in obj:
                        return False
                    items.append(obj)
    except Exception:
        return False
    return len(items) == required


def validate_train_file(file_path: str, required: int) -> bool:
    """校验训练数据文件：存在、条数足够、每条包含 messages 且有 system/user/assistant"""
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) < required:
            return False
        for item in data:
            msgs = item.get("messages") if isinstance(item, dict) else None
            if not isinstance(msgs, list) or len(msgs) < 3:
                return False
            roles = {m.get("role") for m in msgs if isinstance(m, dict)}
            if not {"system", "user", "assistant"}.issubset(roles):
                return False
        return True
    except Exception:
        return False


def process_single_anime(anime_name: str, index: int, total: int, prev: Optional[Dict] = None) -> Dict[str, any]:
    """
    处理单个动画的完整流程

    Args:
        anime_name: 动画名称
        index: 当前索引
        total: 总数

    Returns:
        处理结果字典（包含生成的训练数据）
    """
    safe_anime_name = sanitize_filename(anime_name)

    result = {
        "anime": anime_name,
        "index": index,
        "total": total,
        "qa_success": False,
        "qa_questions": 0,
        "qa_file": None,
        "train_success": False,
        "train_questions": 0,
        "train_data": None,  # 新增：保存该动画生成的训练数据
        "skipped": False,
        "error": None
    }

    # 如果已有成功记录且满足5问/5答，直接复用（断点续跑）
    if prev and prev.get("qa_success") and prev.get("train_success") \
            and prev.get("qa_questions", 0) >= REQUIRED_QUESTIONS \
            and prev.get("train_questions", 0) >= REQUIRED_TRAIN_DATA:
        print(f"➡️  检测到已有完整结果，跳过重跑: {anime_name}\n")
        return prev

    print(f"\n{'='*100}")
    print(f"🎬 处理动画 [{index}/{total}]: {anime_name}")
    print(f"{'='*100}\n")

    # 步骤1: 生成问题（或读取已有问题）
    existing_questions_file = os.path.join(OUTPUT_DIR, f"{safe_anime_name}_questions.jsonl")

    def need_regenerate(count: int) -> bool:
        return count != REQUIRED_QUESTIONS

    # 读取或生成问题，带智能重试
    question_attempt = 0
    while question_attempt < MAX_QA_ATTEMPTS:
        question_attempt += 1

        # 优先复用已有文件（存在且通过校验）
        if validate_questions_file(existing_questions_file, REQUIRED_QUESTIONS):
            question_count = count_questions(existing_questions_file)
            result["qa_success"] = True
            result["qa_questions"] = question_count
            result["qa_file"] = existing_questions_file
            print(f"\n✅ 使用已有问题文件（{question_count}题，校验通过）\n")
            break
        elif os.path.exists(existing_questions_file):
            question_count = count_questions(existing_questions_file)
            print(f"⚠️  现有问题文件不合法或数量不足 {REQUIRED_QUESTIONS}（{question_count}），重新生成\n")

        # 生成新问题
        print(f"📝 步骤1: 生成QA问题 (尝试 {question_attempt}/{MAX_QA_ATTEMPTS})")
        print(f"{'-'*100}\n")

        try:
            qa_result = generate_questions_for_anime_v2(
                anime_name=anime_name,
                output_dir=OUTPUT_DIR,
                max_rounds=MAX_QA_ROUNDS
            )

            if qa_result.get("skipped"):
                result["skipped"] = True
                result["error"] = qa_result.get("reason", "萌娘百科信息不足，已跳过")
                print(f"⚠️  Agent判定信息不足，跳过该动画\n")
                return result

            if qa_result.get("success") and qa_result.get("total_questions") == REQUIRED_QUESTIONS:
                result["qa_success"] = True
                result["qa_questions"] = qa_result.get("total_questions", 0)
                result["qa_file"] = qa_result.get("questions_file")
                print(f"\n✅ QA生成完成: {result['qa_questions']} 个问题\n")
                break
            else:
                print(f"⚠️  QA生成未达标（题数 {qa_result.get('total_questions')}），重试\n")
        except Exception as e:
            print(f"❌ QA生成失败: {e}\n")
            result["error"] = f"QA生成失败: {str(e)}"

        if question_attempt >= MAX_QA_ATTEMPTS:
            print(f"❌ QA多次失败，放弃该动画\n")
            return result

    # 步骤2: 生成训练数据
    try:
        questions_file = result["qa_file"]
        if not questions_file or not os.path.exists(questions_file):
            result["error"] = "问题文件不存在"
            print(f"❌ 问题文件不存在: {questions_file}\n")
            return result

        # 确保问题数量满足要求
        current_q_count = count_questions(questions_file)
        if current_q_count < REQUIRED_QUESTIONS:
            result["error"] = f"问题数量不足 {REQUIRED_QUESTIONS}"
            print(f"❌ 问题数量不足: {current_q_count}\n")
            return result

        print(f"\n📚 步骤2: 生成训练数据")
        print(f"{'-'*100}\n")

        existing_train_file = os.path.join(OUTPUT_DIR, f"{safe_anime_name}_train.json")
        existing_count = count_train_items(existing_train_file) if os.path.exists(existing_train_file) else 0

        # 如已有满足要求且通过校验的训练数据，直接复用并写入合并文件
        if validate_train_file(existing_train_file, REQUIRED_TRAIN_DATA):
            train_data_existing = load_train_data(existing_train_file)
            result["train_success"] = True
            result["train_questions"] = existing_count
            result["train_data"] = None  # 节省内存
            print(f"✅ 复用已有训练数据: {existing_train_file} ({existing_count} 条，校验通过)\n")
            # 写入合并文件（缺什么补什么）
            append_to_merge(train_data_existing)
            return result
        elif os.path.exists(existing_train_file):
            print(f"⚠️  现有训练数据不合法或数量不足 {REQUIRED_TRAIN_DATA}（{existing_count}），重新生成\n")

        train_attempt = 0
        while train_attempt < MAX_TRAIN_ATTEMPTS:
            train_attempt += 1
            print(f"🤖 训练数据生成尝试 {train_attempt}/{MAX_TRAIN_ATTEMPTS}\n")
            train_data = generate_training_data_from_questions(
                questions_file=questions_file,
                output_dir=OUTPUT_DIR,
                anime_name=anime_name,
                max_workers=TRAIN_MAX_WORKERS,
                max_retries=TRAIN_MAX_RETRIES_PER_Q
            )

            if train_data and len(train_data) >= REQUIRED_TRAIN_DATA:
                result["train_success"] = True
                result["train_questions"] = len(train_data)
                result["train_data"] = train_data
                print(f"\n✅ 训练数据生成完成: {len(train_data)} 条\n")
                append_to_merge(train_data)
                break
            else:
                print(f"⚠️  训练数据不足 {REQUIRED_TRAIN_DATA} 条 (当前 {len(train_data) if train_data else 0})，重试\n")

        if not result["train_success"]:
            result["error"] = f"训练数据不足 {REQUIRED_TRAIN_DATA} 条"
            print(f"❌ 训练数据生成未达标，放弃该动画\n")

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
        "skipped": sum(1 for r in results if r.get("skipped")),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def get_data_signature(item: Dict) -> str:
    """生成训练数据的唯一签名（用于去重）"""
    try:
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            return None

        # 提取 user 和 assistant 消息作为签名依据
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


def append_to_merge(train_data: List[Dict]):
    """将本次训练数据追加到 train_merge.json，自动去重文件不存在时创建"""
    if not train_data:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged = []

    if os.path.exists(MERGE_TRAIN_FILE):
        merged = load_train_data(MERGE_TRAIN_FILE)

    # 去重：基于签名判断是否已存在
    existing_signatures = set()
    for item in merged:
        sig = get_data_signature(item)
        if sig:
            existing_signatures.add(sig)

    # 只追加不重复的数据
    added_count = 0
    for item in train_data:
        sig = get_data_signature(item)
        if sig and sig not in existing_signatures:
            merged.append(item)
            existing_signatures.add(sig)
            added_count += 1
        elif sig:
            print(f"⚠️  检测到重复数据，已跳过: {item.get('messages', [{}])[1].get('content', '')[:50]}...")

    if added_count > 0:
        print(f"📝 追加 {added_count} 条新数据到 {MERGE_TRAIN_FILE}")
        with open(MERGE_TRAIN_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    else:
        print(f"ℹ️  没有新数据需要追加（所有数据已存在）")


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

    # 断点续跑：尝试加载已有进度
    existing = load_progress(PROGRESS_FILE)
    results = []

    for i, anime_name in enumerate(anime_list, 1):
        prev = existing.get(anime_name)
        result = process_single_anime(anime_name, i, total, prev)
        results.append(result)

        # 保存中间结果
        save_summary(results, PROGRESS_FILE)

    # 最终统计
    print(f"\n{'='*100}")
    print(f"📊 批量处理完成")
    print(f"{'='*100}\n")

    success_count = sum(1 for r in results if r["qa_success"] and r["train_success"])
    skipped_count = sum(1 for r in results if r.get("skipped"))
    failed_count = total - success_count - skipped_count

    print(f"总计: {total} 个动画")
    print(f"✅ 成功: {success_count} 个")
    print(f"⏭️  跳过: {skipped_count} 个")
    print(f"❌ 失败: {failed_count} 个")
    print(f"📝 生成问题: {sum(r.get('qa_questions', 0) for r in results)} 个")

    # 保存最终结果
    final_summary = save_summary(results, os.path.join(OUTPUT_DIR, "batch_summary.json"))
    print(f"\n📄 详细结果已保存到: {os.path.join(OUTPUT_DIR, 'batch_summary.json')}")

    # 列出失败的动画
    if failed_count > 0:
        print(f"\n❌ 失败的动画:")
        for r in results:
            if not (r.get("skipped") or (r["qa_success"] and r["train_success"])):
                print(f"   - {r['anime']}: {r.get('error', '未知错误')}")

    # ========== 累积所有训练数据到 train_merge.json（过程内已逐步追加，这里提示一下） ==========
    print(f"\n{'='*100}")
    print(f"💾 累积训练数据文件: {MERGE_TRAIN_FILE}")
    print(f"{'='*100}\n")
    merged_count = count_train_items(MERGE_TRAIN_FILE) if os.path.exists(MERGE_TRAIN_FILE) else 0
    if merged_count:
        print(f"✅ 当前合并训练数据总计: {merged_count} 条")
    else:
        print(f"⚠️  合并训练数据为空")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
