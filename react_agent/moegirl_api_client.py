#!/usr/bin/env python3
"""
萌娘百科API客户端
使用FastAPI服务进行搜索
"""

import requests
from typing import Dict, List, Any


class MoegirlTitleSearchTool_API:
    """标题搜索工具（API版本）"""
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.name = "title_search"
        self.description = "搜索萌娘百科中与标题匹配的条目"

    def execute(self, title: str, top_k: int = 5) -> str:
        try:
            response = requests.post(
                f"{self.api_url}/title_search",
                json={"title": title, "top_k": top_k},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("results"):
                return f"未找到与 '{title}' 相关的条目"

            output = [f"找到 {len(result['results'])} 个相关条目:\n"]
            for i, r in enumerate(result["results"], 1):
                output.append(f"{i}. {r['title']}")
                output.append(f"   路径: {r['path']}")
                output.append(f"   Index: {r['index']}")
                output.append("")

            return "\n".join(output)
        except Exception as e:
            return f"搜索失败: {str(e)}"


class MoegirlKeywordSearchTool_API:
    """关键词搜索工具（API版本）"""
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.name = "keyword_search"
        self.description = "在萌娘百科中搜索包含特定关键词的条目"

    def execute(self, keyword: str, top_k: int = 10) -> str:
        try:
            response = requests.post(
                f"{self.api_url}/keyword_search",
                json={"keyword": keyword, "top_k": top_k},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("results"):
                return f"未找到包含 '{keyword}' 的条目"

            output = [f"找到 {len(result['results'])} 个包含关键词的条目:\n"]
            for i, r in enumerate(result["results"], 1):
                output.append(f"{i}. {r['title']}")
                output.append(f"   路径: {r['path']}")
                output.append(f"   Index: {r['index']}")
                output.append("")

            return "\n".join(output)
        except Exception as e:
            return f"搜索失败: {str(e)}"


class MoegirlGetEntryTool_API:
    """获取完整条目工具（API版本）"""
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.name = "get_entry"
        self.description = "获取萌娘百科条目的完整内容。使用从title_search或keyword_search获取的index来调用此工具。"

    def execute(self, index: int) -> str:
        try:
            response = requests.post(
                f"{self.api_url}/get_entry",
                json={"index": index},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

            # API返回的是content字段，不是text字段
            content = result.get('content', '')
            path = result.get('path', '')

            # 如果内容被截断，添加提示
            truncated = result.get('truncated', False)
            if truncated:
                truncation_note = f"\n\n[注: 内容已截断，完整内容长度: {result.get('total_length', 0)} 字符]"
            else:
                truncation_note = ""

            return f"路径: {path}\n\n内容:\n{content}{truncation_note}"
        except Exception as e:
            return f"获取条目失败: {str(e)}"
