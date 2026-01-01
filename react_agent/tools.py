#!/usr/bin/env python3
"""
萌娘百科搜索工具
支持本地数据集加载和搜索
"""

import os
import glob
import json
from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict


class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class SkipAnimeTool(Tool):
    """跳过动画工具"""
    def __init__(self):
        super().__init__(
            name="skip_anime",
            description="当萌娘百科中没有该动画的相关信息时，使用此工具跳过该动画"
        )
    
    def execute(self, reason: str = "萌娘百科信息不足") -> str:
        return f"⏭️  跳过该动画\n原因: {reason}"


class MoegirlWikiTool:
    """萌娘百科数据集工具"""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        self.dataset = self._load_dataset(cache_dir)
    
    def _load_dataset(self, cache_base: str) -> Dataset:
        """加载萌娘百科数据集"""
        print("⏳ 正在加载萌娘百科数据集...")
        
        # 查找所有arrow文件
        arrow_dirs = glob.glob(os.path.join(cache_base, "*/*/*/moe_girl_wiki-*.arrow"), recursive=False)
        
        if not arrow_dirs:
            raise FileNotFoundError(f"未找到萌娘百科数据集文件")
        
        # 按split分组
        split_files = {}
        for arrow_file in arrow_dirs:
            parts = arrow_file.split('/')
            if "train" in parts[-1]:
                split_name = "train"
            elif "validation" in parts[-1]:
                split_name = "validation"
            elif "test" in parts[-1]:
                split_name = "test"
            else:
                split_name = "unknown"
            
            if split_name not in split_files:
                split_files[split_name] = []
            split_files[split_name].append(arrow_file)
        
        # 构建DatasetDict
        from datasets import concatenate_datasets
        datasets = {}
        for split_name, files in split_files.items():
            if files:
                shard_datasets = [Dataset.from_file(f) for f in files]
                datasets[split_name] = concatenate_datasets(shard_datasets)
        
        dataset = DatasetDict(datasets)
        
        print(f"✅ 数据集加载完成")
        return dataset
    
    def search_by_title(self, query: str, top_k: int = 5) -> List[Dict]:
        """按标题搜索"""
        if "train" in self.dataset:
            split = self.dataset["train"]
        else:
            split = list(self.dataset.values())[0]
        
        results = []
        query_lower = query.lower()
        
        for item in split:
            title = item.get("title", "")
            if query_lower in title.lower():
                results.append({
                    "title": title,
                    "path": item.get("path", ""),
                    "text": item.get("text", "")[:500],
                })
        
        return results[:top_k]
    
    def search_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict]:
        """按关键词搜索"""
        if "train" in self.dataset:
            split = self.dataset["train"]
        else:
            split = list(self.dataset.values())[0]
        
        results = []
        keyword_lower = keyword.lower()
        
        for item in split:
            text = item.get("text", "")
            title = item.get("title", "")
            
            if keyword_lower in title.lower() or keyword_lower in text.lower():
                results.append({
                    "title": title,
                    "path": item.get("path", ""),
                    "text": text[:500],
                })
        
        return results[:top_k]
    
    def get_entry(self, path: str) -> Optional[Dict]:
        """获取完整条目"""
        if "train" in self.dataset:
            split = self.dataset["train"]
        else:
            split = list(self.dataset.values())[0]
        
        for item in split:
            if item.get("path", "") == path:
                return {
                    "title": item.get("title", ""),
                    "path": item.get("path", ""),
                    "text": item.get("text", "")
                }
        
        return None


class MoegirlTitleSearchTool(Tool):
    """标题搜索工具"""
    def __init__(self, wiki_tool: MoegirlWikiTool = None):
        super().__init__(
            name="title_search",
            description="搜索萌娘百科中与标题匹配的条目"
        )
        self.wiki_tool = wiki_tool or MoegirlWikiTool()
    
    def execute(self, query: str, top_k: int = 5) -> str:
        results = self.wiki_tool.search_by_title(query, top_k)
        
        if not results:
            return f"未找到与 '{query}' 相关的条目"
        
        output = [f"找到 {len(results)} 个相关条目:\n"]
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['title']}")
            output.append(f"   路径: {r['path']}")
            output.append("")
        
        return "\n".join(output)


class MoegirlKeywordSearchTool(Tool):
    """关键词搜索工具"""
    def __init__(self, wiki_tool: MoegirlWikiTool = None):
        super().__init__(
            name="keyword_search",
            description="在萌娘百科中搜索包含特定关键词的条目"
        )
        self.wiki_tool = wiki_tool or MoegirlWikiTool()
    
    def execute(self, keyword: str, top_k: int = 10) -> str:
        results = self.wiki_tool.search_by_keyword(keyword, top_k)
        
        if not results:
            return f"未找到包含 '{keyword}' 的条目"
        
        output = [f"找到 {len(results)} 个包含关键词的条目:\n"]
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['title']}")
            output.append(f"   路径: {r['path']}")
            output.append("")
        
        return "\n".join(output)


class MoegirlGetEntryTool(Tool):
    """获取完整条目工具"""
    def __init__(self, wiki_tool: MoegirlWikiTool = None):
        super().__init__(
            name="get_entry",
            description="获取萌娘百科条目的完整内容"
        )
        self.wiki_tool = wiki_tool or MoegirlWikiTool()
    
    def execute(self, path: str) -> str:
        entry = self.wiki_tool.get_entry(path)
        
        if not entry:
            return f"未找到路径为 '{path}' 的条目"
        
        output = [
            f"标题: {entry['title']}",
            f"路径: {entry['path']}",
            f"\n内容:\n{entry['text']}"
        ]
        
        return "\n".join(output)


class ToolManager:
    """工具管理器"""
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        wiki_tool = MoegirlWikiTool()
        
        self.register_tool(MoegirlTitleSearchTool(wiki_tool))
        self.register_tool(MoegirlKeywordSearchTool(wiki_tool))
        self.register_tool(MoegirlGetEntryTool(wiki_tool))
        self.register_tool(SkipAnimeTool())
    
    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
