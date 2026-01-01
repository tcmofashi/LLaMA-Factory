#!/usr/bin/env python3
"""
ReAct Agent Package
"""

__version__ = "0.1.0"

# 延迟导入，避免模块加载失败
__all__ = [
    "ReActAgent",
    "Tool",
    "ToolManager",
    "MoegirlWikiTool",
    "MoegirlTitleSearchTool",
    "MoegirlKeywordSearchTool",
    "MoegirlGetEntryTool",
    "MCPTool",
]


def __getattr__(name):
    if name == "ReActAgent":
        from .agent import ReActAgent
        return ReActAgent
    elif name == "Tool":
        from .tools import Tool
        return Tool
    elif name == "ToolManager":
        from .tools import ToolManager
        return ToolManager
    elif name == "MoegirlWikiTool":
        from .tools import MoegirlWikiTool
        return MoegirlWikiTool
    elif name == "MoegirlTitleSearchTool":
        from .tools import MoegirlTitleSearchTool
        return MoegirlTitleSearchTool
    elif name == "MoegirlKeywordSearchTool":
        from .tools import MoegirlKeywordSearchTool
        return MoegirlKeywordSearchTool
    elif name == "MoegirlGetEntryTool":
        from .tools import MoegirlGetEntryTool
        return MoegirlGetEntryTool
    elif name == "MCPTool":
        from .tools import MCPTool
        return MCPTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
