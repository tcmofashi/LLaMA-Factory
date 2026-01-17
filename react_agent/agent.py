#!/usr/bin/env python3
"""
ReAct Agent 实现
支持工具调用和推理-行动循环
"""

import re
import json
import toml
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 检查是否使用API服务模式
USE_MOEGIRL_API = os.getenv("USE_MOEGIRL_API", "false").lower() == "true"
MOEGIRL_API_URL = os.getenv("MOEGIRL_API_URL", "http://localhost:8765")

# 导入SkipAnimeTool（两种模式都需要）
from tools import SkipAnimeTool


def extract_response_content(message) -> str:
    """
    从OpenAI响应消息中提取内容
    支持标准content字段和reasoning字段（某些provider使用）
    """
    # 优先使用content字段
    if message.content:
        return message.content

    # 某些provider（如rinkoai的GLM-4.7）使用reasoning字段
    if hasattr(message, 'reasoning') and message.reasoning:
        return message.reasoning

    # 如果都没有，返回空字符串
    return ""

if USE_MOEGIRL_API:
    print(f"🌐 使用萌娘百科API服务模式: {MOEGIRL_API_URL}")
    from moegirl_api_client import (
        MoegirlTitleSearchTool_API,
        MoegirlKeywordSearchTool_API,
        MoegirlGetEntryTool_API,
    )
else:
    from tools import (
        ToolManager,
        MoegirlWikiTool,
        MoegirlTitleSearchTool,
        MoegirlKeywordSearchTool,
        MoegirlGetEntryTool,
    )


class ReActAgent:
    """ReAct Agent - 推理与行动的循环"""

    def __init__(
        self,
        config_path: str = "/home/tcmofashi/LLaMA-Factory/config.toml",
        tools: Optional[Any] = None,  # ToolManager或None
        max_iterations: int = 5,  # 默认5轮迭代，适合大多数场景
        max_new_tokens: int = 131072,  # 默认128k tokens，GLM-4.7最大支持
        verbose: bool = True,
        max_context_tokens: int = 150000,  # 上下文窗口估算阈值
    ):
        self.config_path = config_path
        self.max_iterations = max_iterations
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.verbose = verbose

        # 加载配置
        self.config = self._load_config()

        # 初始化负载均衡器
        self.load_balancer = None

        # 初始化工具管理器
        if USE_MOEGIRL_API:
            # API模式：不需要ToolManager，直接使用API客户端
            self.tool_manager = None
            self._register_api_tools()
        else:
            # 本地模式：使用ToolManager
            self.tool_manager = tools or ToolManager()
            self._register_default_tools()

        # 初始化模型
        self.model = None
        self.tokenizer = None
        self.client = None
        self.providers = {}  # 所有providers
        self.provider_order = []  # provider优先级顺序

    def _load_config(self) -> Dict[str, Any]:
        """加载TOML配置文件"""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)

        return config

    def _register_default_tools(self):
        """注册默认工具（本地模式）"""
        # 注册萌娘百科搜索工具套件
        title_search_tool = MoegirlTitleSearchTool()
        keyword_search_tool = MoegirlKeywordSearchTool()
        get_entry_tool = MoegirlGetEntryTool()
        skip_tool = SkipAnimeTool()

        self.tool_manager.register_tool(title_search_tool)
        self.tool_manager.register_tool(keyword_search_tool)
        self.tool_manager.register_tool(get_entry_tool)
        self.tool_manager.register_tool(skip_tool)

    def _register_api_tools(self):
        """注册API工具（API模式）"""
        # 使用API客户端工具
        self.title_search_tool = MoegirlTitleSearchTool_API(MOEGIRL_API_URL)
        self.keyword_search_tool = MoegirlKeywordSearchTool_API(MOEGIRL_API_URL)
        self.get_entry_tool = MoegirlGetEntryTool_API(MOEGIRL_API_URL)
        self.skip_tool = SkipAnimeTool()  # Skip工具不需要API

    def initialize_model(self):
        """初始化语言模型（支持标准化provider配置和负载均衡）"""
        # 从配置中读取provider信息
        agent_config = self.config.get("agent", {})
        providers_config = self.config.get("providers", {})

        # 获取负载均衡策略
        load_balancing_strategy = agent_config.get("load_balancing", "weighted")

        # 这里使用OpenAI兼容的API
        # 如果使用的是本地模型，需要使用 transformers
        try:
            from openai import OpenAI
            import httpx
            from rate_limiter import RateLimiter
            from load_balancer import LoadBalancer

            # 初始化负载均衡器
            self.load_balancer = LoadBalancer(strategy=load_balancing_strategy)

            # 初始化所有配置的providers
            for provider_key in providers_config:
                provider_config = providers_config[provider_key]

                # 检查必需字段
                if not all(k in provider_config for k in ["model", "api_key", "base_url"]):
                    if self.verbose:
                        print(f"⚠️  Provider '{provider_key}' 配置不完整，跳过")
                    continue

                # 创建客户端
                timeout = provider_config.get("timeout", 300)
                timeout_config = httpx.Timeout(timeout, connect=60.0)

                client = OpenAI(
                    base_url=provider_config["base_url"],
                    api_key=provider_config["api_key"],
                    timeout=timeout_config,
                )

                # 获取RPM并创建速率限制器
                rpm = provider_config.get("rpm", 60)  # 默认60次/分钟
                rate_limiter = RateLimiter(rate=rpm, period=60)

                # 保存provider信息
                self.providers[provider_key] = {
                    "client": client,
                    "model": provider_config["model"],
                    "name": provider_config.get("name", provider_key),
                    "rpm": rpm,
                    "rate_limiter": rate_limiter,
                }

                # 添加到负载均衡器
                self.load_balancer.add_provider(provider_key, rpm)

                # 添加到优先级列表
                self.provider_order.append(provider_key)

                if self.verbose:
                    print(f"✅ 已加载Provider '{provider_key}':")
                    print(f"   名称: {provider_config.get('name', provider_key)}")
                    print(f"   模型: {provider_config['model']}")
                    print(f"   API: {provider_config['base_url']}")
                    print(f"   RPM: {rpm}次/分钟")

            # 检查是否有可用的providers
            if not self.providers:
                raise RuntimeError("❌ 没有可用的provider配置")

            # 为了向后兼容，设置默认属性
            first_provider = self.provider_order[0]
            self.client = self.providers[first_provider]["client"]
            self.model_name = self.providers[first_provider]["model"]
            self.rate_limiter = self.providers[first_provider]["rate_limiter"]

            if self.verbose:
                print(f"\n📊 Provider配置完成:")
                print(f"   已加载Providers: {len(self.providers)}")
                self.load_balancer.print_status()

        except ImportError:
            # 如果没有openai库，使用本地模型
            self._initialize_local_model()

    def _initialize_local_model(self):
        """初始化本地模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = self.config.get("model", {})
        model_path = model_config.get("path", "Qwen/Qwen2.5-7B-Instruct")

        if self.verbose:
            print(f"⏳ 加载本地模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model_name = model_path
        self.client = None  # 使用本地推理

        if self.verbose:
            print(f"✅ 本地模型加载完成")

    def _generate_with_openai(self, messages: List[Dict[str, str]]) -> str:
        """使用OpenAI API生成（支持负载均衡和智能切换）"""
        import openai
        import time

        agent_config = self.config.get("agent", {})
        max_retries = agent_config.get("max_retries", 5)
        base_delay = agent_config.get("retry_base_delay", 2)

        # 跟踪已经尝试失败的provider，避免重复尝试
        failed_providers = set()

        # 尝试使用可用的provider
        for attempt in range(max_retries * len(self.providers)):  # 总尝试次数 = 最大重试 × provider数量
            # 使用负载均衡器选择provider（跳过已失败的）
            provider_key = None
            for _ in range(len(self.providers) * 2):  # 尝试多次以找到未失败的provider
                candidate_key = self.load_balancer.get_provider()
                if candidate_key and candidate_key not in failed_providers:
                    provider_key = candidate_key
                    break
                elif len(failed_providers) >= len(self.providers):
                    # 所有provider都失败了
                    raise RuntimeError(f"❌ 所有provider都已失败: {', '.join(failed_providers)}")

            if not provider_key:
                raise RuntimeError("❌ 没有可用的provider")

            try:
                # 获取provider信息
                provider = self.providers[provider_key]
                client = provider["client"]
                model_name = provider["model"]
                rate_limiter = provider["rate_limiter"]

                # 速率限制：等待获取令牌
                if rate_limiter:
                    if self.verbose:
                        status = rate_limiter.get_status()
                        print(f"⏳ [{provider['name']}] 速率限制检查: 剩余令牌 {status['available_tokens']}/{status['rate']}")

                    # 等待令牌，最多等待30秒
                    if not rate_limiter.acquire(timeout=30):
                        # 30秒内无法获取令牌，标记为失败并切换到其他provider
                        if self.verbose:
                            print(f"⚠️  [{provider['name']}] 速率限制等待超时，切换到其他provider...")
                        failed_providers.add(provider_key)
                        continue

                # 调用API
                if self.verbose:
                    print(f"🔄 使用Provider: {provider['name']} ({model_name})")

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=self.max_new_tokens,
                )
                return extract_response_content(response.choices[0].message)

            except openai.RateLimitError as e:
                # API并发限制（429错误）
                provider = self.providers[provider_key]
                if self.verbose:
                    print(f"⚠️  [{provider['name']}] 遇到并发限制(429)，尝试其他provider...")

                # 标记当前provider失败
                failed_providers.add(provider_key)

                # 如果还有其他provider，继续尝试
                if len(failed_providers) < len(self.providers):
                    continue
                else:
                    # 所有provider都失败了
                    raise RuntimeError(f"❌ 所有provider都遇到并发限制: {', '.join(failed_providers)}")

            except openai.APITimeoutError:
                # 超时错误，标记当前provider失败，尝试其他provider
                provider = self.providers[provider_key]
                if self.verbose:
                    print(f"⚠️  [{provider['name']}] API调用超时，尝试其他provider...")
                failed_providers.add(provider_key)

                # 如果还有其他provider，继续尝试
                if len(failed_providers) < len(self.providers):
                    continue
                else:
                    raise RuntimeError(f"❌ 所有provider都超时: {', '.join(failed_providers)}")

            except openai.APIError as e:
                # API错误，对于5xx错误可以尝试切换provider，4xx错误则直接失败
                provider = self.providers[provider_key]

                # 检查错误类型，5xx错误可以重试，4xx错误不应该重试
                error_str = str(e).lower()
                is_server_error = any(code in error_str for code in ['500', '502', '503', '504'])

                if is_server_error and len(self.providers) > 1:
                    # 服务器错误，尝试其他provider
                    if self.verbose:
                        print(f"⚠️  [{provider['name']}] 服务器错误，尝试其他provider...")
                    failed_providers.add(provider_key)

                    if len(failed_providers) < len(self.providers):
                        continue
                    else:
                        raise RuntimeError(f"❌ 所有provider都遇到服务器错误: {', '.join(failed_providers)}")
                elif is_server_error:
                    # 只有一个provider，使用指数退避重试
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if self.verbose:
                            print(f"⚠️  [{provider['name']}] 服务器错误，等待 {delay} 秒后重试... (尝试 {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"❌ [{provider['name']}] 服务器错误：已重试 {max_retries} 次: {e}")
                else:
                    # 客户端错误（4xx），不应该重试
                    raise RuntimeError(f"❌ [{provider['name']}] 客户端错误: {e}")

            except openai.APIConnectionError as e:
                # 连接错误，尝试切换到其他provider
                provider = self.providers[provider_key]
                if self.verbose:
                    print(f"⚠️  [{provider['name']}] 连接失败，尝试其他provider...")
                failed_providers.add(provider_key)

                # 如果还有其他provider，继续尝试
                if len(failed_providers) < len(self.providers):
                    continue
                else:
                    raise RuntimeError(f"❌ 所有provider都无法连接: {', '.join(failed_providers)}")

            except Exception as e:
                provider = self.providers[provider_key]
                raise RuntimeError(f"❌ [{provider['name']}] 未知错误: {type(e).__name__}: {e}")

    def _generate_with_local(self, messages: List[Dict[str, str]]) -> str:
        """使用本地模型生成"""
        import torch

        # 构建prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        # 解码
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成响应"""
        if self.client:
            return self._generate_with_openai(messages)
        else:
            return self._generate_with_local(messages)

    # ========== 上下文压缩辅助 ==========
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """粗略估算tokens（字符/4），避免依赖额外tokenizer"""
        text = json.dumps(messages, ensure_ascii=False)
        return int(len(text) / 4)

    def _summarize_context(self, prefix_messages: List[Dict[str, str]], model_name: str) -> Optional[str]:
        """使用当前模型压缩前序对话，返回摘要文本"""
        try:
            if not self.client:
                return None  # 本地模式暂不压缩，直接依赖窗口裁剪

            prompt = (
                "请用中文总结以下对话历史，保留所有已查到的关键信息、已使用的工具和结论，"
                "避免遗漏事实。总结后用于继续对话，无需重复未完成的行动。"
            )
            summary_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(prefix_messages, ensure_ascii=False)},
            ]

            response = self.client.chat.completions.create(
                model=model_name,
                messages=summary_messages,
                temperature=0.2,
                max_tokens=150000,
            )
            return extract_response_content(response.choices[0].message).strip()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  上下文压缩失败，将退化为窗口裁剪: {e}")
            return None

    def _compact_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """当上下文超过阈值时压缩：>90k先摘要+保留最近5轮，仍>50k则保留2轮"""
        # 粗略估算
        est_tokens = self._estimate_tokens(messages)
        if est_tokens <= 90000:
            return messages

        if self.verbose:
            print(f"⚠️  上下文过长，估算 {est_tokens} tokens，开始压缩")

        # 识别最近对话窗口：跳过系统消息
        system_msg = messages[0]
        recent_pairs = []
        body = messages[1:]

        # 将末尾消息按 assistant+user 配对回溯
        pair_buffer = []
        for msg in reversed(body):
            pair_buffer.append(msg)
            if len(pair_buffer) >= 2:  # 粗略配对
                recent_pairs.append(list(reversed(pair_buffer)))
                pair_buffer = []
            if len(recent_pairs) >= 5:
                break

        recent_pairs = list(reversed(recent_pairs))  # 恢复正序
        recent_flat = [m for pair in recent_pairs for m in pair]

        # 待摘要的前序消息
        kept_set = set(id(m) for m in recent_flat)
        prefix_messages = [m for m in body if id(m) not in kept_set]

        summary_text = self._summarize_context(prefix_messages, getattr(self, "model_name", ""))
        summary_msg = {"role": "system", "content": f"对话摘要：{summary_text}"} if summary_text else None

        new_messages = [system_msg]
        if summary_msg:
            new_messages.append(summary_msg)
        new_messages.extend(recent_flat)

        est_tokens_after = self._estimate_tokens(new_messages)
        if est_tokens_after <= 50000:
            return new_messages

        if self.verbose:
            print(f"⚠️  压缩后仍估算 {est_tokens_after} tokens，进一步缩窗至2轮")

        # 仅保留最近2轮
        recent_pairs_short = []
        pair_buffer = []
        for msg in reversed(body):
            pair_buffer.append(msg)
            if len(pair_buffer) >= 2:
                recent_pairs_short.append(list(reversed(pair_buffer)))
                pair_buffer = []
            if len(recent_pairs_short) >= 2:
                break
        recent_pairs_short = list(reversed(recent_pairs_short))
        recent_flat_short = [m for pair in recent_pairs_short for m in pair]

        final_messages = [system_msg]
        if summary_msg:
            final_messages.append(summary_msg)
        final_messages.extend(recent_flat_short)

        if self.verbose:
            print(f"✅ 上下文压缩完成，估算 {self._estimate_tokens(final_messages)} tokens")

        return final_messages

    def _format_tools(self) -> str:
        """格式化工具列表为字符串"""
        if USE_MOEGIRL_API:
            tools = [
                self.title_search_tool,
                self.keyword_search_tool,
                self.get_entry_tool,
                self.skip_tool
            ]
        else:
            tools = self.tool_manager.tools.values()

        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        return "\n".join(tool_descriptions)

    def run(self, query: str) -> str:
        """运行ReAct循环"""
        # 初始化模型（如果还未初始化）
        if self.client is None and self.model is None:
            self.initialize_model()

        # 构建系统消息
        system_message = {
            "role": "system",
            "content": f"""你是一个专业的动画知识助手，擅长使用工具查找和分析萌娘百科中的动画相关信息。

可用工具：
{self._format_tools()}

请按照以下格式工作：
1. Thought: 思考当前需要做什么
2. Action: 使用工具，格式为 Action: 工具名称(参数)
3. Observation: 观察工具返回的结果
4. 重复步骤1-3，直到找到足够的信息
5. Answer: 基于收集的信息给出最终答案

示例：
Thought: 我需要查找《孤独摇滚！》的相关信息
Action: title_search(query="孤独摇滚")
Observation: 找到相关条目...
Thought: 我需要获取更详细的信息
Action: get_entry(path="孤独摇滚！")
Observation: 获取到详细信息...
Answer: 《孤独摇滚！》是...

重要提示：
- **skip_anime工具只能在超过10轮迭代后使用**（第11轮及以后）
- 在前10轮迭代中，请努力尝试各种搜索策略（不同关键词、不同搜索方式）
- 只有在确实无法找到任何相关信息时，才能在10轮后使用skip_anime工具
- 在搜索时，尝试使用不同的关键词（中文标题、日文标题、英文标题等）
- 在给出答案前，确保已经收集到足够的信息
- Answer必须基于工具返回的事实信息，不要编造

现在请处理用户的问题："""
        }

        # 构建消息历史
        messages = [system_message, {"role": "user", "content": query}]

        # 迭代循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"迭代 {iteration + 1}/{self.max_iterations}")
                print(f"{'='*80}\n")

            # 上下文压缩（超过90k触发，目标<=50k）
            messages = self._compact_messages(messages)

            # 生成响应
            response = self.generate(messages)

            if self.verbose:
                print(f"模型响应:\n{response}\n")

            # 解析响应
            action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', response, re.DOTALL)
            answer_match = re.search(r'Answer:\s*(.*?)(?:\n|$)', response, re.DOTALL)

            if answer_match and not action_match:
                # 找到答案，没有行动
                final_answer = answer_match.group(1).strip()
                if self.verbose:
                    print(f"✅ 最终答案: {final_answer[:200]}...")
                    print(f"{'='*80}\n")
                return final_answer

            elif action_match:
                # 执行行动
                action_name = action_match.group(1)
                action_params_str = action_match.group(2).strip()

                # 解析参数
                try:
                    if action_params_str:
                        # 尝试解析为字典
                        action_params = eval(f"dict({action_params_str})")
                    else:
                        action_params = {}
                except:
                    action_params = {}

                if self.verbose:
                    print(f"🎬 执行行动: {action_name}")
                    print(f"   参数: {action_params}\n")

                # 执行工具
                try:
                    if USE_MOEGIRL_API:
                        # API模式
                        if action_name == "title_search":
                            if "title" not in action_params or not action_params.get("title"):
                                action_params["title"] = query  # 缺省时用原始查询兜底
                            result = self.title_search_tool.execute(**action_params)
                        elif action_name == "keyword_search":
                            if "keyword" not in action_params or not action_params.get("keyword"):
                                action_params["keyword"] = query
                            result = self.keyword_search_tool.execute(**action_params)
                        elif action_name == "get_entry":
                            result = self.get_entry_tool.execute(**action_params)
                        elif action_name == "skip_anime":
                            # 检查是否超过10轮迭代
                            if iteration < 10:
                                result = f"错误：skip_anime工具只能在超过10轮迭代后使用。当前迭代次数: {iteration + 1}。请继续尝试搜索信息。"
                                if self.verbose:
                                    print(f"⚠️  {result}\n")
                                    print(f"💡 提示：请继续使用搜索工具查找信息，不要过早放弃\n")
                            else:
                                result = self.skip_tool.execute(**action_params)
                        else:
                            result = f"错误：未知工具 '{action_name}'"
                    else:
                        # 本地模式
                        if action_name == "skip_anime":
                            # 检查是否超过10轮迭代
                            if iteration < 10:
                                result = f"错误：skip_anime工具只能在超过10轮迭代后使用。当前迭代次数: {iteration + 1}。请继续尝试搜索信息。"
                                if self.verbose:
                                    print(f"⚠️  {result}\n")
                                    print(f"💡 提示：请继续使用搜索工具查找信息，不要过早放弃\n")
                            else:
                                tool = self.tool_manager.get_tool(action_name)
                                if tool:
                                    result = tool.execute(**action_params)
                                else:
                                    result = f"错误：未知工具 '{action_name}'"
                        else:
                            tool = self.tool_manager.get_tool(action_name)
                            if tool:
                                result = tool.execute(**action_params)
                            else:
                                result = f"错误：未知工具 '{action_name}'"

                    if self.verbose:
                        print(f"📊 工具返回:\n{str(result)[:500]}...\n")

                    # 添加助手消息（包含行动和观察）
                    assistant_message = {
                        "role": "assistant",
                        "content": response
                    }
                    user_message = {
                        "role": "user",
                        "content": f"Observation: {str(result)}"
                    }
                    messages.extend([assistant_message, user_message])

                except Exception as e:
                    error_msg = f"执行工具时出错: {str(e)}"
                    if self.verbose:
                        print(f"❌ {error_msg}\n")

                    # 添加错误观察
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {error_msg}"
                    })

            else:
                # 没有找到行动或答案，使用整个响应作为答案
                if self.verbose:
                    print(f"✅ 使用模型响应作为最终答案\n")
                    print(f"{'='*80}\n")
                return response

        # 达到最大迭代次数
        if self.verbose:
            print(f"⚠️  达到最大迭代次数 ({self.max_iterations})，返回最后的结果\n")
            print(f"{'='*80}\n")

        return response


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="ReAct Agent - 动画知识助手")
    parser.add_argument("--query", type=str, help="要处理的问题")
    parser.add_argument("--config", type=str, default="/home/tcmofashi/LLaMA-Factory/config.toml", help="配置文件路径")
    parser.add_argument("--max-tokens", type=int, default=131072, help="最大生成token数")
    parser.add_argument("--max-iterations", type=int, default=50, help="最大迭代次数")
    parser.add_argument("--max-context", type=int, default=150000, help="上下文token估算阈值")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    # 创建Agent
    agent = ReActAgent(
        config_path=args.config,
        max_new_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        max_context_tokens=args.max_context,
        verbose=args.verbose
    )

    # 运行
    try:
        response = agent.run(args.query)
        print(f"\n最终答案:\n{response}")
    except RuntimeError as e:
        # RuntimeError表示无法恢复的错误（如所有provider都失败）
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # 其他未知错误
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
