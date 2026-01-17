#!/usr/bin/env python3
"""
QA生成Pipeline V2
使用GLM-4.7 Agent生成问题，DS V3.2进行质量审批
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from filename_utils import sanitize_filename


def call_glm_agent(prompt: str, max_tokens: int = 131072, max_iterations: int = 50) -> str:
    """调用GLM Agent生成响应"""
    # 确保传递萌娘百科API环境变量
    import os
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, "--max-tokens", str(max_tokens), "--max-iterations", str(max_iterations)],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        env=env
    )

    if result.returncode != 0:
        raise RuntimeError(f"GLM Agent执行失败: {result.stderr}")

    return result.stdout


def call_ds_judge(prompt: str, max_retries: int = 3) -> Dict:
    """调用DS V3.2进行问题评分，失败时重试，不再默认放行"""
    import toml
    from pathlib import Path
    from openai import OpenAI

    last_err: Optional[Exception] = None

    # 加载配置
    config_path = Path("/home/tcmofashi/LLaMA-Factory/config.toml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    provider_config = config["providers"]["deepseek"]

    client = OpenAI(
        api_key=provider_config["api_key"],
        base_url=provider_config["base_url"]
    )
    model = provider_config["model"]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            result = response.choices[0].message.content.strip()

            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(result)

        except Exception as e:  # noqa: BLE001 - 打印后再重试
            last_err = e
            print(f"⚠️  DS评分失败（第{attempt}次尝试/{max_retries}）：{e}")
            if attempt < max_retries:
                continue

    # 重试仍失败，抛出异常让上层决定重试整组
    raise RuntimeError(f"DS评分连续失败：{last_err}")


def generate_qa_prompt(anime_name: str) -> str:
    """
    生成问题生成的prompt（只生成问题，不生成答案）

    这是核心prompt，定义了问题的结构和要求
    """
    prompt = f"""请为动画《{anime_name}》生成5个高质量的问题。

**注意：你只需要生成问题，不需要生成答案。**

要求：
1. 使用萌娘百科搜索工具查找该动画的详细信息
2. 基于**真实、准确的信息**生成问题，不要编造
3. 5个问题必须按照以下结构：

**问题1-3**：整合多个信息点的事实性问题（type: "factual"）
- 每个问题都必须整合多个相关联的信息点
- 可以是：播出信息、制作信息、角色/声优、剧情、音乐、原作等任意组合
- 三个问题之间应该关注不同的信息组合，避免重复
- 灵活组合各种维度，创造有价值的多信息点问题
- **重要**：问题可以适当包含背景信息以使问题更完整、不那么干巴，但**严禁在问题中直接或间接包含答案**

**问题4**：内容概括性问题（type: "summary"）
- 请概括《{anime_name}》的核心内容和主要情节
- 包括作品的主题、故事背景、主要冲突等
- 要求：内容完整、条理清晰、重点突出
- **重要**：问题应该具体，而不是简单的"请概括这部动画"，可以包含适当的背景信息

**问题5**：主题和艺术分析（type: "analysis"）
- 分析《{anime_name}》的主题思想、艺术表现手法、创作特色等
- 可以涉及：导演风格、画面表现、音乐、象征隐喻等
- 要求：有深度、有见解、基于事实
- **重要**：问题应该引导深度分析，而不是表面的描述

**问题质量要求**：
1. 问题必须完整、可回答，不能是反问句或陈述句
2. 问题可以包含必要的背景信息，使问题更具体、更有深度
3. **严禁在问题中出现答案**（无论是直接还是间接暗示）
4. 问题应该有深度，避免过于直白或表面化
5. 问题应该引发思考，而不是简单的事实罗列

**问题示例对比**：

❌ 不好的问题（过于直白、干巴、包含答案）：
- "《莉兹与青鸟》的导演是谁？"（过于简单、干巴）
- "山田尚子执导的《莉兹与青鸟》讲述了什么故事？"（包含答案）
- "《莉兹与青鸟》是不是一部关于音乐和友情的动画？"（反问句）

✅ 好的问题（有深度、有背景、不含答案）：
- "在京阿尼众多动画作品中，《莉兹与青鸟》以其独特的视觉语言和音乐叙事著称。请结合作品的制作背景、导演风格，以及其对原著《吹响吧！上低音号》系列的延续与突破，分析这部剧场版动画在艺术表现和主题表达上的独特之处。" (type: "analysis")
- "《莉兹与青鸟》作为山田尚子独立执导的剧场版，请分析其在音乐表现形式、镜头语言运用、角色心理刻画等方面的独特性，以及作品如何通过音乐与影像的结合展现两位主人公的内心世界和关系变化。" (type: "analysis")

重要提示：
1. **必须先使用搜索工具**查找信息，不要直接回答
2. 如果萌娘百科中没有该动画的相关信息，请使用 skip_anime 工具跳过
3. 问题必须基于**真实、准确**的萌娘百科数据
4. 每个问题都应该具体、详细、有价值、有深度
5. **只生成问题，不要生成答案**
6. 输出格式必须严格按照以下JSON格式：

```json
[
  {{
    "question": "问题1的完整内容",
    "type": "factual"
  }},
  {{
    "question": "问题2的完整内容",
    "type": "factual"
  }},
  {{
    "question": "问题3的完整内容",
    "type": "factual"
  }},
  {{
    "question": "问题4的完整内容",
    "type": "summary"
  }},
  {{
    "question": "问题5的完整内容",
    "type": "analysis"
  }}
]
```

**⚠️ 重要：输出格式说明**
- 你需要使用搜索工具查找信息（Thought → Action → Observation）
- **最后一步**：直接输出JSON格式的问题，**不要使用"Answer:"前缀**
- ❌ **错误格式**：Answer: ```json [...] ```
- ✅ **正确格式**：```json [...] ```
- ❌ **不要**在Observation后就停止，必须继续生成JSON
- ❌ **不要**输出空的JSON代码块
- ✅ **必须**输出包含5个问题的完整JSON

**输出示例（正确的格式）**：
```json
[
  {{"question": "问题1", "type": "factual"}},
  ...
]
```

请开始工作：
"""

    return prompt


def generate_distribution_prompt(questions_json: str, anime_name: str) -> str:
    """生成仅用于分布守卫的prompt"""
    prompt = f"""请检查以下关于动画《{anime_name}》的5个问题是否满足**类型分布**要求。

待检查的问题：
{questions_json}

**分布要求**（必须满足）：
- 问题1-3必须是 factual 类型
- 问题4必须是 summary 类型
- 问题5必须是 analysis 类型

**如果分布不满足**：
- 直接判定不通过
- 生成具体的重新生成指导（regenerate_instructions），明确说明需要如何调整类型位置或问题设计，以便重新生成整组问题

**返回JSON格式**：
```json
{{
    "distribution_check": {{
        "passed": true/false,
        "details": "类型分布检查说明"
    }},
    "regenerate_instructions": "如果不通过，给出如何调整分布的指导；通过则简要写 OK"
}}
```
"""

    return prompt


def generate_quality_prompt(questions_json: str, anime_name: str) -> str:
    """生成仅用于质量守卫的prompt（假定分布已通过）"""
    prompt = f"""请严格审批以下关于动画《{anime_name}》的5个问题质量（分布已预检通过，无需再验证类型分布）。

待审批的问题：
{questions_json}

**质量审批标准**：
1. **问题深度要求**：
     - 90-100分：问题有深度、引发思考、包含适当背景信息但不包含答案
     - 70-89分：问题基本合格，但深度稍欠或背景信息不足
     - 60-69分：问题过于直白或干巴，但尚可回答
     - 0-59分：问题质量不合格（反问句、陈述句、包含答案、无法回答等）

2. **问题完整性**：
     - 问题必须是完整的疑问句，不能是反问句或陈述句
     - 问题必须可以被回答，不能过于模糊或宽泛

3. **背景信息vs答案**：
     - ✅ 允许：在问题中包含必要的背景信息，使问题更完整、有深度
     - ❌ 严禁：在问题中直接或间接包含答案

**评分规则**：
- 90-100分：优秀，完全符合要求
- 70-89分：良好，基本符合要求，但有小瑕疵
- 60-69分：及格，问题过于直白或干巴，但可回答
- 0-59分：不及格，必须重新生成

**返回JSON格式**：
```json
{{
    "evaluations": [
        {{
            "question": 1,
            "score": 95,
            "reason": "问题深度充分，包含适当背景信息但不包含答案，引发思考",
            "should_regenerate": false
        }},
        {{
            "question": 2,
            "score": 45,
            "reason": "问题过于直白干巴，缺少背景信息，深度不足",
            "should_regenerate": true,
            "regenerate_instructions": "请重新生成问题，使其更有深度。可以在问题中加入制作背景、角色设定等信息，使问题更完整。注意不要在问题中包含答案。"
        }},
        ...
    ],
    "approved": [1, 3, 4, 5],  # 及格的问题编号（分数>=60）
    "need_regenerate": [2],     # 需要重新生成的问题编号（分数<60）
    "total_score": 385
}}
```

**重要**：
- 如果问题分数 < 60，必须标记 should_regenerate=true
- 必须提供具体的 regenerate_instructions，告诉GLM Agent如何改进这个问题
- regenerate_instructions应该具体说明问题哪里不好，应该如何改进
"""

    return prompt


def parse_questions_from_output(output: str) -> Optional[List[Dict]]:
    """
    从GLM输出中解析问题列表（改进版）

    Returns:
        问题列表，如果解析失败返回None
    """
    # 情况1: 完全没有JSON标记
    if '"question"' not in output and '```json' not in output:
        print(f"   ❌ 解析失败: 未检测到JSON格式输出")
        print(f"   📝 输出长度: {len(output)} 字符")
        if len(output) < 200:
            print(f"   📝 输出内容: {output[:200]}...")
        return None

    # 情况2: 检查是否有JSON代码块
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()

        # 检查是否为空代码块
        if not json_str or json_str == "":
            print(f"   ❌ 解析失败: 检测到JSON代码块但内容为空")
            print(f"   📝 这通常意味着GLM Agent在输出JSON前被中断")
            return None

        # 尝试解析
        try:
            questions = json.loads(json_str)
            if isinstance(questions, list) and len(questions) == 5:
                return questions
            else:
                print(f"   ❌ 解析失败: JSON格式正确但问题数量不是5个（实际: {len(questions) if isinstance(questions, list) else '非列表'}）")
                return None
        except json.JSONDecodeError as e:
            print(f"   ❌ 解析失败: JSON格式错误 - {e}")
            print(f"   📝 JSON内容前200字符: {json_str[:200]}...")
            return None
    else:
        # 情况3: 有JSON标记但没有代码块
        if '```json' in output:
            print(f"   ❌ 解析失败: 检测到JSON开始标记但缺少结束标记")
            print(f"   📝 这通常意味着GLM Agent生成JSON时被中断")
            return None

    # 情况4: 尝试直接解析整个输出
    try:
        questions = json.loads(output)
        if isinstance(questions, list) and len(questions) == 5:
            return questions
        else:
            print(f"   ❌ 解析失败: 直接解析成功但问题数量不是5个")
            return None
    except json.JSONDecodeError as e:
        print(f"   ❌ 解析失败: 直接解析也失败 - {e}")
        return None


def regenerate_single_question(
    anime_name: str,
    question_index: int,
    question_type: str,
    instructions: str
) -> Dict:
    """
    重新生成单个问题（只生成问题，不生成答案）

    Args:
        anime_name: 动画名称
        question_index: 问题编号（1-5）
        question_type: 问题类型（factual/summary/analysis）
        instructions: DS给出的改进指令

    Returns:
        重新生成的问题（只包含question和type）
    """
    prompt = f"""请为动画《{anime_name}》重新生成第{question_index}个问题。

**问题类型**：{question_type}

**改进要求**：
{instructions}

**重要提醒**：
1. 使用萌娘百科搜索工具查找该动画的详细信息
2. 问题可以包含适当的背景信息，使问题更完整、有深度
3. **问题类型不得变化，必须保持为 {question_type}**
4. **严禁在问题中直接或间接包含答案**
5. 问题必须完整、可回答，不能是反问句或陈述句
6. 问题应该有深度，避免过于直白或表面化
7. **只生成问题，不要生成答案**

请只返回第{question_index}个问题，格式如下：
```json
{{
  "question": "重新生成的问题内容",
  "type": "{question_type}"
}}
```

请开始：
"""

    try:
        output = call_glm_agent(prompt, max_tokens=65536, max_iterations=50)

        # 解析JSON结果
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            question_data = json.loads(json_match.group(1))
        else:
            question_data = json.loads(output)

        return question_data

    except Exception as e:
        print(f"⚠️  重新生成问题{question_index}失败: {e}")
        return None


def run_distribution_guard(questions: List[Dict], anime_name: str) -> Dict:
    """运行分布守卫，返回评估结果和指令"""
    questions_json = json.dumps(questions, ensure_ascii=False, indent=2)
    prompt = generate_distribution_prompt(questions_json, anime_name)

    try:
        evaluation = call_ds_judge(prompt)
        distribution = evaluation.get("distribution_check", {}) or {}
        passed = bool(distribution.get("passed", False))
        instructions = evaluation.get("regenerate_instructions") or distribution.get("details", "")

        return {
            "passed": passed,
            "instructions": instructions,
            "raw": evaluation
        }
    except RuntimeError as e:
        print(f"⚠️  分布守卫调用DS失败，整组重试：{e}")
        return {
            "passed": False,
            "instructions": f"DS调用失败，请整组重试。原因: {e}",
            "raw": {"error": str(e)}
        }


def run_quality_guard(questions: List[Dict], anime_name: str) -> Dict:
    """运行质量守卫，返回需要重生的题目等信息"""
    questions_json = json.dumps(questions, ensure_ascii=False, indent=2)
    prompt = generate_quality_prompt(questions_json, anime_name)

    try:
        evaluation = call_ds_judge(prompt)

        need_regenerate = set(evaluation.get("need_regenerate", []))
        evaluations = evaluation.get("evaluations", []) or []

        for item in evaluations:
            q_num = item.get("question")
            if q_num is None:
                continue
            score = item.get("score", 0)
            should_regenerate = item.get("should_regenerate", False)
            if score < 60 or should_regenerate:
                need_regenerate.add(q_num)

        approved = evaluation.get("approved", []) or []

        return {
            "need_regenerate": sorted(need_regenerate),
            "approved": approved,
            "evaluations": evaluations,
            "raw": evaluation
        }
    except RuntimeError as e:
        print(f"⚠️  质量守卫调用DS失败，整组重试：{e}")
        return {
            "error": str(e),
            "need_regenerate": list(range(1, 6)),
            "approved": [],
            "evaluations": [],
            "raw": {"error": str(e)}
        }


def generate_questions_for_anime_v2(
    anime_name: str,
    output_dir: str,
    max_rounds: int = 50
) -> Dict[str, any]:
    """
    为动画生成问题（V2版本：GLM生成 + DS审批 + 智能重新生成）

    Args:
        anime_name: 动画名称
        output_dir: 输出目录
        max_rounds: 最大审批轮数

    Returns:
        包含生成结果的字典
    """

    print(f"\n{'='*100}")
    print(f"#{' '*40}QA生成系统: {anime_name}{' '*40}")
    print(f"# GLM4.7 Agent 生成 + DS V3.2 审批 + 智能重新生成")
    print(f"{'='*100}\n")

    safe_anime_name = sanitize_filename(anime_name)

    final_questions: List[Dict] = []
    total_regenerations = 0
    max_regenerations = 50
    success = False

    for round_num in range(1, max_rounds + 1):
        print(f"{'='*100}")
        print(f"🔄 第 {round_num} 轮（整组生成）")
        print(f"{'='*100}\n")

        prompt = generate_qa_prompt(anime_name)
        print(f"🤖 GLM4.7 Agent 正在生成初始5题...\n")
        output = call_glm_agent(prompt, max_iterations=50)
        questions = parse_questions_from_output(output)

        if not questions:
            if "⏭️  跳过该动画" in output or "skip_anime" in output:
                print(f"\n⚠️  Agent检测到萌娘百科信息不足，跳过该动画")
                if "原因:" in output:
                    reason_start = output.find("原因:")
                    reason_end = output.find("\n", reason_start)
                    if reason_end != -1:
                        reason = output[reason_start:reason_end].strip()
                        print(f"   {reason}")

                return {
                    "anime_name": anime_name,
                    "skipped": True,
                    "reason": "萌娘百科信息不足"
                }

            print(f"\n⚠️  第{round_num}轮：无法解析GLM输出，整组重试\n")
            continue

        print(f"\n✅ GLM生成了 {len(questions)} 个问题，进入守卫流程\n")

        while True:
            # 分布守卫
            dist_result = run_distribution_guard(questions, anime_name)
            if dist_result.get("passed"):
                print(f"📐 分布守卫通过\n")
            else:
                print(f"⚠️  分布守卫不通过，整组问题需要重生成")
                if dist_result.get("instructions"):
                    print(f"   分布改进指令: {dist_result['instructions']}")
                break  # 退出内层，开启下一轮整组生成

            # 质量守卫
            quality_result = run_quality_guard(questions, anime_name)

            if quality_result.get("error"):
                print(f"⚠️  质量守卫因DS失败中断，整组重试\n")
                break
            evaluations = quality_result.get("evaluations", [])
            need_regenerate = quality_result.get("need_regenerate", [])
            approved = quality_result.get("approved", [])

            print(f"📊 质量评分结果：")
            for item in evaluations:
                q_num = item.get("question")
                score = item.get("score")
                reason = item.get("reason", "")
                status = "✅ 通过" if score is not None and score >= 60 else "❌ 需重生"
                print(f"  问题{q_num}: {score}分 - {status}")
                print(f"    原因: {reason}")
                if item.get("should_regenerate"):
                    print(f"    改进指令: {item.get('regenerate_instructions', '无')}")
            print()

            # 无需重生且全部5题通过
            if not need_regenerate and len(approved) == 5:
                final_questions = questions
                success = True
                break

            # 无需重生但通过数量不足，整组重来避免死循环
            if not need_regenerate and len(approved) < 5:
                print(f"⚠️  评分通过数量不足（{len(approved)}/5），但无待重生项，整组重新生成\n")
                break

            if total_regenerations >= max_regenerations:
                print(f"⚠️  达到最大单题重生次数 ({max_regenerations})，停止本轮\n")
                break

            # 逐题重生
            print(f"🔄 进入单题重生：{need_regenerate}\n")
            for q_num in need_regenerate:
                if total_regenerations >= max_regenerations:
                    print(f"⚠️  已到达单题重生上限，停止重生\n")
                    break

                eval_item = next((e for e in evaluations if e.get("question") == q_num), None)
                instructions = "请提升问题深度、保持类型不变，避免包含答案。"
                if eval_item and eval_item.get("regenerate_instructions"):
                    instructions = eval_item["regenerate_instructions"]

                question_type = questions[q_num - 1].get("type") if q_num - 1 < len(questions) else "factual"

                print(f"🤖 重生问题{q_num} (type={question_type})")
                new_question = regenerate_single_question(
                    anime_name,
                    q_num,
                    question_type,
                    instructions
                )

                if new_question:
                    questions[q_num - 1] = new_question
                    total_regenerations += 1
                    print(f"✅ 问题{q_num}重生完成 (累计重生 {total_regenerations})\n")
                else:
                    print(f"❌ 问题{q_num}重生失败，保留原问题\n")

            # 重生后再次进入分布->质量守卫循环
            continue

        # 内循环结束后，如果已满足条件则跳出外循环
        if success:
            print(f"\n{'='*100}")
            print(f"✅ 第{round_num}轮完成，全部守卫通过！")
            print(f"{'='*100}")
            print(f"生成问题数: 5")
            print(f"总重新生成次数: {total_regenerations}")
            print()
            break
        else:
            print(f"⚠️  第{round_num}轮未通过守卫，进入下一轮整组生成\n")

    # 保存问题
    if success and len(final_questions) == 5:
        questions_file = os.path.join(output_dir, f"{safe_anime_name}_questions.jsonl")

        with open(questions_file, 'w', encoding='utf-8') as f:
            for q in final_questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

        print(f"✅ 问题已保存到: {questions_file}\n")

        return {
            "success": True,
            "anime_name": anime_name,
            "questions": final_questions,
            "total_questions": len(final_questions),
            "questions_file": questions_file,
            "total_rounds": round_num,
            "total_regenerations": total_regenerations
        }

    print(f"❌ 未能生成满足要求的5个问题\n")
    return {
        "success": False,
        "anime_name": anime_name,
        "questions": final_questions,
        "total_questions": len(final_questions),
        "error": "无法生成满足守卫要求的5个问题"
    }


# 导入re模块
import re
