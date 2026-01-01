#!/usr/bin/env python3
"""
为问题生成答案（训练数据）
"""

import os
import json
import subprocess
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入训练数据处理工具
from training_data_utils import (
    generate_diverse_system_prompt,
    postprocess_answer,
    validate_answer_quality
)


def generate_answer_prompt(question: str) -> str:
    """
    生成答案的prompt
    """
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


def call_agent_for_answer(question: str) -> str:
    """调用Agent生成答案"""
    prompt = generate_answer_prompt(question)

    # 确保传递萌娘百科API环境变量
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, "--max-tokens", "131072", "--max-iterations", "50"],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        env=env
    )

    if result.returncode != 0:
        raise RuntimeError(f"Agent执行失败: {result.stderr}")

    return result.stdout


def generate_training_data_from_questions(
    questions_file: str,
    output_dir: str,
    max_workers: int = 5
):
    """
    从问题文件生成训练数据

    Args:
        questions_file: 问题文件路径（jsonl格式）
        output_dir: 输出目录
        max_workers: 最大并发数，默认5
    """

    print(f"\n{'='*100}")
    print(f"{'#'*35} 开始生成训练数据")
    print(f"{'='*100}\n")

    print(f"问题文件: {questions_file}")
    print(f"输出目录: {output_dir}\n")

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

    # 并行处理
    actual_workers = min(max_workers, len(questions))
    print(f"🔄 使用并行处理（最多{actual_workers}个并发）\n")

    results = []

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_question = {}
        
        for i, question_obj in enumerate(questions, 1):
            question = question_obj["question"]
            future = executor.submit(process_single_question, question, i)
            future_to_question[future] = (question_obj, i)
            
            print(f"[进度: {i}] 开始处理问题 {i+1}: {question[:80]}...")
            print(f"{'='*100}")
            print(f"🤖 回答问题 {i+1}: {question}")
            print(f"{'='*100}\n")
            print(f"⏳ 正在调用ReAct Agent生成回答...\n")
        
        for future in as_completed(future_to_question):
            question_obj, idx = future_to_question[future]
            
            try:
                answer = future.result()
                
                print(f"✅ 回答生成完成\n")
                print(f"✅ 问题 {idx} 处理完成\n")
                
                results.append({
                    "question": question_obj["question"],
                    "answer": answer,
                    "type": question_obj.get("type", "unknown")
                })
                
            except Exception as e:
                print(f"❌ 问题 {idx} 处理失败")
                print(f"❌ 生成回答失败: {e}\n")
    
    # 保存结果
    save_training_data(results, output_dir, answer_record_dir)


def process_single_question(question: str, index: int) -> str:
    """处理单个问题"""
    return call_agent_for_answer(question)


def save_training_data(results: List[Dict], output_dir: str, answer_record_dir: str):
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
            print(f"  ✅ 问题 {i+1}: 质量验证通过 (system_prompt: {system_prompt[:30]}...)")

    print()

    # 2. 保存完整格式（TXT）
    anime_name = os.path.basename(output_dir).replace("_questions.json.jsonl", "").replace("_questions.jsonl", "")
    full_format_file = os.path.join(answer_record_dir, f"{anime_name}_full.txt")

    with open(full_format_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(processed_results, 1):
            f.write(f"问题 {i}: {result['question']}\n")
            f.write(f"System: {result['system_prompt']}\n")
            f.write(f"答案: {result['answer']}\n")
            f.write("\n" + "="*100 + "\n\n")

    print(f"✅ 完整格式: {full_format_file} ({len(processed_results)} 条) - TXT格式")

    # 3. 保存伪造格式（JSON - OpenAI chat格式，带system prompt）
    fake_format_file = os.path.join(output_dir, "train_fake.json")

    fake_data = []
    for result in processed_results:
        fake_data.append({
            "messages": [
                {"role": "system", "content": result["system_prompt"]},
                {"role": "user", "content": result["question"]},
                {"role": "assistant", "content": result["answer"]}
            ]
        })

    with open(fake_format_file, 'w', encoding='utf-8') as f:
        json.dump(fake_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 伪造格式: {fake_format_file} ({len(processed_results)} 条总计) - JSON格式（含system prompt）")

    # 4. 保存问题集合（JSON）
    questions_file = os.path.join(output_dir, "questions.json")

    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)

    print(f"✅ 问题集合: {questions_file} ({len(processed_results)} 个问题总计) - JSON格式\n")

    print(f"✅ 训练数据生成完成\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 generate_training_data.py <questions_file>")
        sys.exit(1)
    
    questions_file = sys.argv[1]
    output_dir = "/home/tcmofashi/LLaMA-Factory/agent_data"
    
    generate_training_data_from_questions(questions_file, output_dir)
