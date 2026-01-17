#!/usr/bin/env python3
"""
为问题生成答案（训练数据）- 改进版
添加重试机制，确保每个问题都能成功生成答案
"""

import os
import json
import subprocess
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filename_utils import sanitize_filename

# 导入训练数据处理工具
from training_data_utils import (
    generate_diverse_system_prompt,
    postprocess_answer,
    validate_answer_quality
)


def generate_answer_prompt(question: str) -> str:
    """生成答案的prompt"""
    prompt = f"""你是一个专业的动画知识助手。请使用萌娘百科搜索工具查找相关信息，然后回答以下问题：

问题：{question}

重要输出要求：
1. **最终答案必须是自然的段落形式**，就像你本来就知道这些信息一样
2. **绝对不要在答案中提到"我使用了搜索工具"、"根据萌娘百科"等任何工具使用痕迹**
3. **不要使用标题格式（如"##"、"###"等）**，直接用自然段落叙述
4. 答案应该看起来像是**你本身就知道的ACG知识**，而不是通过查询获得的
5. 先简要介绍作品信息，然后回答具体问题
6. 使用自然的口语化表达，避免机械的"首先、其次、最后"等结构

请记住：你的回答应该像一个真正的ACG爱好者在聊天时自然说出的，而不是一个AI在查询数据库后给出的报告。

请开始：
"""
    return prompt


def call_agent_for_answer(question: str, timeout: int = 600) -> str:
    """
    调用Agent生成答案（带超时控制）

    Args:
        question: 问题文本
        timeout: 超时时间（秒），默认10分钟

    Returns:
        答案文本

    Raises:
        RuntimeError: Agent执行失败
        TimeoutError: 执行超时
    """
    prompt = generate_answer_prompt(question)

    # 确保传递萌娘百科API环境变量
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt,
         "--max-tokens", "131072", "--max-iterations", "50"],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        env=env,
        timeout=timeout  # 添加超时控制
    )

    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "未知错误"
        raise RuntimeError(f"Agent执行失败: {error_msg}")

    return result.stdout


def process_single_question_with_retry(
    question: str,
    index: int,
    max_retries: int = 3,
    initial_timeout: int = 600
) -> Dict[str, any]:
    """
    处理单个问题，带重试机制和指数退避

    Args:
        question: 问题文本
        index: 问题索引
        max_retries: 最大重试次数
        initial_timeout: 初始超时时间（秒）

    Returns:
        包含答案和元数据的字典
        {
            "success": True/False,
            "answer": "答案文本",
            "attempts": 尝试次数,
            "error": "错误信息（如果失败）"
        }
    """
    for attempt in range(max_retries):
        try:
            # 指数退避：每次重试增加超时时间
            current_timeout = initial_timeout * (2 ** attempt)

            print(f"🤖 尝试生成答案 (尝试 {attempt + 1}/{max_retries})...")
            print(f"   超时设置: {current_timeout} 秒\n")

            answer = call_agent_for_answer(question, timeout=current_timeout)

            # 验证答案质量
            if answer and len(answer.strip()) > 100:
                print(f"✅ 答案生成成功 (长度: {len(answer)} 字符)\n")
                return {
                    "success": True,
                    "answer": answer,
                    "attempts": attempt + 1,
                    "error": None
                }
            else:
                print(f"⚠️  答案质量不达标 (长度: {len(answer) if answer else 0} 字符)")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 10  # 指数退避：10s, 20s, 40s
                    print(f"⏳ 等待 {wait_time} 秒后重试...\n")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 达到最大重试次数\n")

        except subprocess.TimeoutExpired as e:
            print(f"❌ 尝试 {attempt + 1} 超时 (>{initial_timeout * (2 ** attempt)}秒)")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10
                print(f"⏳ 等待 {wait_time} 秒后重试...\n")
                time.sleep(wait_time)
            else:
                print(f"❌ 达到最大重试次数\n")

        except RuntimeError as e:
            print(f"❌ 尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10
                print(f"⏳ 等待 {wait_time} 秒后重试...\n")
                time.sleep(wait_time)
            else:
                print(f"❌ 达到最大重试次数\n")

        except Exception as e:
            print(f"❌ 尝试 {attempt + 1} 发生意外错误: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10
                print(f"⏳ 等待 {wait_time} 秒后重试...\n")
                time.sleep(wait_time)
            else:
                print(f"❌ 达到最大重试次数\n")

    # 所有重试都失败
    return {
        "success": False,
        "answer": None,
        "attempts": max_retries,
        "error": "经过 {} 次重试后仍然失败".format(max_retries)
    }


def generate_training_data_from_questions(
    questions_file: str,
    output_dir: str,
    max_workers: int = 3,  # 降低并发数，避免API限流
    anime_name: str = None,
    max_retries: int = 3  # 新增：每个问题的最大重试次数
):
    """
    从问题文件生成训练数据（改进版：带重试机制）

    Args:
        questions_file: 问题文件路径（jsonl格式）
        output_dir: 输出目录
        max_workers: 最大并发数（降低到3以避免API限流）
        anime_name: 动画名称（可选，用于文件命名）
        max_retries: 每个问题的最大重试次数

    Returns:
        生成的训练数据列表（OpenAI chat格式）
    """

    print(f"\n{'='*100}")
    print(f"{'#'*35} 开始生成训练数据 (改进版)")
    print(f"{'='*100}\n")

    print(f"问题文件: {questions_file}")
    print(f"输出目录: {output_dir}")
    print(f"最大并发数: {max_workers}")
    print(f"最大重试次数: {max_retries}\n")

    # 读取问题
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"📊 共 {len(questions)} 个问题需要回答\n")

    # 创建输出目录
    answer_record_dir = os.path.join(output_dir, "answer_record")
    os.makedirs(answer_record_dir, exist_ok=True)

    # 创建日志目录
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    # 记录失败的问题
    failed_questions = []
    success_count = 0
    total_attempts = 0

    # 串行处理以确保稳定性（或者使用降低的并发数）
    if max_workers == 1:
        print(f"🔄 使用串行处理以确保稳定性\n")
        results = []

        for i, question_obj in enumerate(questions, 1):
            question = question_obj["question"]
            print(f"\n{'='*100}")
            print(f"📝 处理问题 {i}/{len(questions)}")
            print(f"{'='*100}")
            print(f"问题: {question}\n")

            # 带重试的处理
            result = process_single_question_with_retry(
                question, i, max_retries=max_retries
            )

            total_attempts += result["attempts"]

            if result["success"]:
                results.append({
                    "question": question_obj["question"],
                    "answer": result["answer"],
                    "type": question_obj.get("type", "unknown")
                })
                success_count += 1
                print(f"✅ 问题 {i} 处理成功 (尝试次数: {result['attempts']})\n")
            else:
                failed_questions.append({
                    "index": i,
                    "question": question,
                    "error": result["error"]
                })
                print(f"❌ 问题 {i} 处理失败: {result['error']}\n")

    else:
        # 并行处理（但降低并发数）
        print(f"🔄 使用并行处理（最多{max_workers}个并发）\n")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, question_obj in enumerate(questions, 1):
                question = question_obj["question"]
                future = executor.submit(
                    process_single_question_with_retry,
                    question, i, max_retries
                )
                futures[future] = (question_obj, i)

                print(f"[进度: {i}/{len(questions)}] 提交问题 {i}...")

            print(f"\n⏳ 等待所有任务完成...\n")

            for future in as_completed(futures):
                question_obj, idx = futures[future]

                try:
                    result = future.result()
                    total_attempts += result["attempts"]

                    if result["success"]:
                        results.append({
                            "question": question_obj["question"],
                            "answer": result["answer"],
                            "type": question_obj.get("type", "unknown")
                        })
                        success_count += 1
                        print(f"✅ 问题 {idx} 处理成功 (尝试次数: {result['attempts']})\n")
                    else:
                        failed_questions.append({
                            "index": idx,
                            "question": question_obj["question"],
                            "error": result["error"]
                        })
                        print(f"❌ 问题 {idx} 处理失败: {result['error']}\n")

                except Exception as e:
                    print(f"❌ 问题 {idx} 发生意外错误: {e}\n")
                    failed_questions.append({
                        "index": idx,
                        "question": question_obj["question"],
                        "error": f"意外错误: {str(e)}"
                    })

    # 统计信息
    print(f"\n{'='*100}")
    print(f"📊 生成统计")
    print(f"{'='*100}\n")
    print(f"总问题数: {len(questions)}")
    print(f"成功生成: {success_count}")
    print(f"失败数量: {len(failed_questions)}")
    print(f"总尝试次数: {total_attempts}")
    print(f"平均尝试次数: {total_attempts / len(questions) if questions else 0:.1f}\n")

    # 保存失败记录
    if failed_questions:
        failed_name = sanitize_filename(anime_name) if anime_name else "failed_questions"
        failed_log_file = log_dir / f"{failed_name}_failed.json"
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            json.dump(failed_questions, f, ensure_ascii=False, indent=2)
        print(f"⚠️  失败记录已保存到: {failed_log_file}\n")

    # 保存结果并返回训练数据
    if results:
        train_data = save_training_data(results, output_dir, answer_record_dir, anime_name)
        return train_data
    else:
        print(f"❌ 所有问题都生成失败，无法生成训练数据\n")
        return []


def save_training_data(results: List[Dict], output_dir: str, answer_record_dir: str, anime_name: str = None):
    """保存训练数据（带后处理和system prompt）"""

    print(f"\n{'='*100}")
    print(f"📊 训练数据生成完成")
    print(f"{'='*100}\n")

    # 1. 后处理答案并生成system prompt
    print(f"🔄 正在后处理答案并生成system prompt...\n")

    processed_results = []
    for i, result in enumerate(results):
        question = result["question"]
        raw_answer = result["answer"]

        # 清理答案格式
        cleaned_answer = postprocess_answer(raw_answer, question)

        # 生成多样化的system prompt
        system_prompt = generate_diverse_system_prompt(question, cleaned_answer)

        # 验证答案质量
        quality_check = validate_answer_quality(cleaned_answer)

        processed_results.append({
            "question": question,
            "answer": cleaned_answer,
            "system_prompt": system_prompt,
            "quality": quality_check
        })

        if quality_check["issues"]:
            print(f"  ⚠️  问题 {i+1}: {', '.join(quality_check['issues'])}")
        else:
            print(f"  ✅ 问题 {i+1}: 质量验证通过")

    print()

    # 2. 保存完整格式（TXT）
    if anime_name is None:
        anime_name = os.path.basename(output_dir).replace("_questions.json.jsonl", "").replace("_questions.jsonl", "")
    safe_anime_name = sanitize_filename(anime_name)
    full_format_file = os.path.join(answer_record_dir, f"{safe_anime_name}_full.txt")

    with open(full_format_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(processed_results, 1):
            f.write(f"问题 {i}: {result['question']}\n")
            f.write(f"System: {result['system_prompt']}\n")
            f.write(f"答案: {result['answer']}\n")
            f.write("\n" + "="*100 + "\n\n")

    print(f"✅ 完整格式: {full_format_file} ({len(processed_results)} 条) - TXT格式")

    # 3. 保存单个动画的训练数据（单独的JSON文件）
    anime_train_file = os.path.join(output_dir, f"{safe_anime_name}_train.json")

    fake_data = []
    for result in processed_results:
        fake_data.append({
            "messages": [
                {"role": "system", "content": result["system_prompt"]},
                {"role": "user", "content": result["question"]},
                {"role": "assistant", "content": result["answer"]}
            ]
        })

    with open(anime_train_file, 'w', encoding='utf-8') as f:
        json.dump(fake_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 单个动画训练数据: {anime_train_file} ({len(processed_results)} 条) - JSON格式")

    # 4. 保存问题集合（JSON）
    questions_file = os.path.join(output_dir, f"{safe_anime_name}_questions.json")

    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)

    print(f"✅ 问题集合: {questions_file} ({len(processed_results)} 个问题总计) - JSON格式\n")

    print(f"✅ 训练数据生成完成\n")

    return fake_data  # 返回生成的训练数据，用于累积


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python3 generate_training_data_improved.py <questions_file> [max_retries]")
        sys.exit(1)

    questions_file = sys.argv[1]
    output_dir = "/home/tcmofashi/LLaMA-Factory/agent_data"
    max_retries = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    generate_training_data_from_questions(
        questions_file,
        output_dir,
        max_workers=1,  # 默认使用串行处理
        max_retries=max_retries
    )
