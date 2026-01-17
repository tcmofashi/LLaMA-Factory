#!/usr/bin/env python3
"""
训练数据处理工具
- 生成多样化的system prompt
- 使用DeepSeek-V3.2清洗答案（带Agent调用能力）
"""

import re
import random
import subprocess
from typing import Dict, List
import toml
from pathlib import Path


def load_deepseek_client():
    """加载DeepSeek客户端"""
    config_path = Path("/home/tcmofashi/LLaMA-Factory/config.toml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    provider_config = config["providers"]["deepseek"]

    from openai import OpenAI
    client = OpenAI(
        api_key=provider_config["api_key"],
        base_url=provider_config["base_url"]
    )

    return client, provider_config["model"]


def generate_diverse_system_prompt(question: str, answer: str) -> str:
    """
    生成通用的ACG领域system prompt（与具体作品无关）

    Args:
        question: 用户问题（保留参数以兼容接口，但不再使用）
        answer: 生成的答案（保留参数以兼容接口，但不再使用）

    Returns:
        生成的高质量system prompt
    """
    # 通用的ACG领域system prompt列表（与具体作品无关，专业正式）
    acg_system_prompts = [
        # 角色定位类
        "你是一位资深的ACG爱好者，对动画、漫画、游戏等作品非常了解。",
        "你是二次元文化研究者，专注于ACG领域的各类作品分析。",
        "你是一位动漫评论家，擅长分析动画作品的剧情、角色和表现手法。",
        "你是ACG领域的百科全书，对动画、漫画、游戏等作品了如指掌。",
        "你是一名专业的动漫博主，长期撰写ACG相关的内容。",
        "你是ACG文化的传播者，热爱分享动漫、游戏等作品。",

        # 专业兴趣类
        "你对日本动画产业有深入了解，熟悉制作公司、声优、音乐等各个环节。",
        "你是动漫制作的专业人士，了解动画制作的整个流程和细节。",
        "你是声优文化爱好者，对声优的代表作、配音风格等有深入研究。",
        "你是ACG音乐达人，熟悉动漫OP、ED、插入歌等相关音乐作品。",
        "你专注于日本动画的历史发展，了解各个时期的代表作品。",
        "你是动画制片人视角的观察者，了解ACG产业的商业运作。",

        # 兴趣爱好类
        "你热爱观看各类动画作品，无论是日常番还是战斗番都有涉猎。",
        "你是萌系动画的忠实观众，对芳文社、四格漫画改编作品情有独钟。",
        "你是轻小说阅读爱好者，熟悉各类轻小说及其改编作品。",
        "你是游戏玩家，对ACG相关的游戏作品有丰富体验。",
        "你对原创动画作品有浓厚兴趣，对原创剧本和导演风格有独到见解。",
        "你是长期追番的观众，每季都会关注多部新番动画。",

        # 分析视角类
        "你擅长分析动画的叙事结构、角色关系和主题表达。",
        "你专注于研究动画的视觉表现手法，包括作画、演出、摄影等方面。",
        "你对动画中的音乐运用和音效制作有独到的见解。",
        "你善于从文学和艺术角度分析动画作品的深层内涵。",
        "你关注动画中的社会文化现象和意识形态表达。",
        "你擅长分析ACG作品的商业成功要素和受众定位。",

        # 综合能力类
        "你熟悉ACG作品的制作团队、声优阵容、音乐制作等各个方面。",
        "你对动画、漫画、游戏、轻小说等ACG载体都有深入了解。",
        "你是ACG领域的通才，各个领域的作品都有涉猎。",
        "你关注ACG作品的跨媒体改编和IP运营。",
        "你了解ACG作品在日本的流行趋势和文化影响。",

        # 专业表达类（替代原来的轻浮口语）
        "你是ACG领域的长期关注者，对各类作品都有深入研究和积累。",
        "你对动漫文化有深刻理解，能够从多个角度分析作品价值。",
        "你是ACG作品的系统性研究者，注重作品的完整性和艺术性。",
        "你对日本动漫产业的发展历程和现状有全面的认识。",
        "你是动画艺术的鉴赏者，擅长从专业角度评价作品质量。",
        "你对ACG亚文化现象有敏锐的洞察力，能够进行深度分析。"
    ]

    # 随机选择一个返回
    return random.choice(acg_system_prompts)


def call_glm_agent_regenerate(question: str, instructions: str) -> str:
    """
    调用GLM Agent重新生成答案

    Args:
        question: 原始问题
        instructions: DeepSeek给出的改进指令

    Returns:
        重新生成的答案
    """
    try:
        import os
        env = os.environ.copy()
        env['USE_MOEGIRL_API'] = 'true'

        # 构建重新生成的prompt
        prompt = f"""请重新回答以下问题。

问题：{question}

改进要求：
{instructions}

重要提示：
1. 使用萌娘百科搜索工具查找信息
2. 最终答案必须是自然段落形式
3. 不要在答案中提到工具使用
4. 看起来要像你本身就懂这个知识
5. 先介绍作品信息，再回答问题

请开始："""

        result = subprocess.run(
            ["python3", "react_agent/agent.py", "--query", prompt, "--max-tokens", "131072", "--max-iterations", "10"],
            capture_output=True,
            text=True,
            cwd="/home/tcmofashi/LLaMA-Factory",
            timeout=180,
            env=env
        )

        if result.returncode != 0:
            return f"重新生成失败: {result.stderr}"

        # 提取最终答案（去掉工具痕迹等）
        output = result.stdout
        answer_start = output.find("最终答案:")
        if answer_start != -1:
            answer = output[answer_start + 5:].strip()
        else:
            answer = output.strip()

        return answer

    except Exception as e:
        return f"重新生成失败: {str(e)}"


def clean_answer_with_deepseek(answer: str, question: str, max_retries: int = 2) -> str:
    """
    使用DeepSeek-V3.2清洗答案，必要时要求GLM Agent重新生成

    Args:
        answer: 原始答案
        question: 问题
        max_retries: 最大重试次数

    Returns:
        清洗后的答案
    """
    for attempt in range(max_retries):
        try:
            client, model = load_deepseek_client()

            if attempt == 0:
                # 第一次尝试：直接清洗
                clean_prompt = f"""你是一个专业的ACG问答质量评估和改进专家。

任务：评估并改进以下AI助手生成的答案

问题：{question}

原始答案：
{answer}

请进行以下工作：

1. **质量评估**：
   - 答案是否有效回答了问题？
   - 是否包含了足够的信息？
   - 信息是否准确相关？

2. **清理改进**：
   - 如果答案中包含工具使用痕迹（如"我使用了搜索"、"根据萌娘百科"、"Thought:"、"Action:"、"Observation:"等），完全去除
   - 去除所有Markdown格式标记（如###、##、**等）
   - 将答案改写成自然的段落形式
   - 使用自然的口语化表达
   - 先介绍作品信息，然后回答问题

3. **重新生成判断**：
   - 如果原始答案**完全没有回答问题**或**答案质量太差**（如内容空洞、完全跑题、信息严重不足），请输出：
     ```
     NEED_REGENERATE: <具体的重新生成要求>
     ```
     例如：`NEED_REGENERATE: 原答案没有回答问题，请重新生成一个详细完整的答案，先介绍作品背景，然后回答具体问题。`

4. **输出格式**：
   - 如果不需要重新生成，直接输出改进后的答案
   - 如果需要重新生成，必须以`NEED_REGENERATE:`开头（注意是英文冒号）

请直接输出结果（不要有其他解释）："""
            else:
                # 后续尝试：处理重新生成的答案
                clean_prompt = f"""你是一个专业的文本编辑，负责清理ACG问答的答案。

以下是一个重新生成的答案，请确保：

1. 完全去除工具使用痕迹
2. 去除Markdown格式
3. 改写为自然段落
4. 使用口语化表达

问题：{question}

重新生成的答案：
{answer}

请直接输出清理后的答案："""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": clean_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            result = response.choices[0].message.content.strip()

            # 检查是否需要重新生成
            if result.startswith("NEED_REGENERATE:") or result.startswith("NEED_REGENERATE："):
                # 提取重新生成的指令
                if ":" in result or "：" in result:
                    instructions = result.split(":", 1)[1].split("：", 1)[0].strip()
                else:
                    instructions = result[15:].strip()

                print(f"  🔄 DeepSeek要求重新生成: {instructions[:50]}...")

                # 调用GLM Agent重新生成
                new_answer = call_glm_agent_regenerate(question, instructions)

                # 递归调用自己来清洗新答案
                return clean_answer_with_deepseek(new_answer, question, max_retries - 1)

            # 如果结果太短，使用fallback
            if not result or len(result) < 20:
                return clean_answer_fallback(answer, question)

            return result

        except Exception as e:
            print(f"  ⚠️  DeepSeek尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                return clean_answer_fallback(answer, question)

    # 如果所有尝试都失败，使用fallback
    return clean_answer_fallback(answer, question)


def clean_answer_fallback(answer: str, question: str) -> str:
    """
    Fallback答案清理方法（当DeepSeek不可用时）
    """
    # 去掉工具调用痕迹
    answer = re.sub(r'🌐 使用萌娘百科API服务模式[^\n]*\n?', '', answer)
    answer = re.sub(r'最终答案:\n*', '', answer)

    # 去掉ReAct标记
    answer = re.sub(r'Thought:[^\n]*\n?', '', answer)
    answer = re.sub(r'Action:[^\n]*\n?', '', answer)
    answer = re.sub(r'Observation:[^\n]*\n?', '', answer)
    answer = re.sub(r'Answer:[^\n]*\n?', '', answer)

    # 去掉搜索结果格式
    answer = re.sub(r'\[\d+\][^\n]*\n?', '', answer)
    answer = re.sub(r'   路径:[^\n]*\n?', '', answer)
    answer = re.sub(r'   Index:[^\n]*\n?', '', answer)
    answer = re.sub(r'找到 \d+ 个[^\n]*\n?', '', answer)

    # 去掉Markdown格式
    answer = re.sub(r'^#+\s+', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'\*\*', '', answer)
    answer = re.sub(r'`', '', answer)

    # 去掉外部信源表述
    answer = re.sub(r'基于萌娘百科[^\n]*\n?', '', answer)
    answer = re.sub(r'根据萌娘百科[^\n]*\n?', '', answer)
    answer = re.sub(r'为了回答.*?我使用[^\n]*\n?', '', answer, flags=re.DOTALL)
    answer = re.sub(r'我首先使用[^\n]*?\n?', '', answer, flags=re.DOTALL)

    # 去掉列表符号
    lines = answer.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'^[\*\-\+\d]+\.\s*', '', line)
        line = re.sub(r'^\*\s*', '', line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    answer = '\n'.join(cleaned_lines)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()

    # 如果答案太短，返回默认答案
    if len(answer) < 30:
        anime_match = re.search(r'《(.*?)》', question)
        if anime_match:
            anime_name = anime_match.group(1)
            answer = f"{anime_name}是一部优秀的动画作品。由于信息有限，暂时无法提供更详细的介绍。"
        else:
            answer = "抱歉，当前无法提供详细的答案。"

    return answer


# 保留原来的函数名作为接口
def clean_answer(answer: str, question: str) -> str:
    """清理答案格式（使用DeepSeek-V3.2）"""
    return clean_answer_with_deepseek(answer, question)


def postprocess_answer(answer: str, question: str) -> str:
    """
    完整的答案后处理流程

    Args:
        answer: 原始答案
        question: 问题

    Returns:
        处理后的答案
    """
    # 直接使用clean_answer，它已经包含了所有清理逻辑
    return clean_answer(answer, question)


def validate_answer_quality(answer: str) -> Dict[str, any]:
    """
    验证答案质量

    Args:
        answer: 答案

    Returns:
        验证结果
    """
    issues = []

    # 检查是否有工具痕迹
    if '🌐' in answer or '萌娘百科API' in answer:
        issues.append("存在工具调用痕迹")

    # 检查是否有外部信源表述
    if '基于萌娘百科' in answer or '根据萌娘百科' in answer:
        issues.append("存在外部信源表述")

    # 检查是否有过多的结构化标记
    if answer.count('###') > 3 or answer.count('##') > 3:
        issues.append("存在过多的标题格式")

    # 检查是否过于机械
    mechanical_patterns = ['首先，', '其次，', '再次，', '最后，']
    pattern_count = sum(answer.count(p) for p in mechanical_patterns)
    if pattern_count > 5:
        issues.append(f"过于机械（{pattern_count}个机械过渡词）")

    # 检查长度
    if len(answer) < 50:
        issues.append("答案过短")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "length": len(answer)
    }
