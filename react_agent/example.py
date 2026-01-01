#!/usr/bin/env python3
"""
ReAct Agent 使用示例
"""

from agent import ReActAgent
from tools import Tool, ToolManager, MoegirlWikiTool


def example_basic_query():
    """示例1：基本查询"""
    print("\n" + "=" * 80)
    print("示例1：基本查询")
    print("=" * 80)

    agent = ReActAgent(verbose=True)

    questions = [
        "种崎敦美的生日是什么时候？",
        "Comic Girls的主角有谁？",
        "长谷川育美有哪些代表作品？",
    ]

    for question in questions:
        print(f"\n🔹 问题: {question}")
        response = agent.run(question)
        print(f"✅ 答案: {response}\n")
        print("-" * 80)


def example_tool_usage():
    """示例2：直接使用工具"""
    print("\n" + "=" * 80)
    print("示例2：直接使用萌娘百科工具")
    print("=" * 80)

    tool = MoegirlWikiTool()

    # 按标题搜索
    print("🔍 标题搜索: Comic Girls")
    result = tool.run(title="Comic Girls", max_results=3)
    print(result)

    print("\n" + "-" * 80 + "\n")

    # 按关键词搜索
    print("🔍 关键词搜索: 种崎敦美")
    result = tool.run(keyword="种崎敦美", max_results=3)
    print(result)


def example_custom_tool():
    """示例3：添加自定义工具"""
    print("\n" + "=" * 80)
    print("示例3：自定义工具")
    print("=" * 80)

    class CalculatorTool(Tool):
        """简单的计算器工具"""

        @property
        def name(self) -> str:
            return "calculator"

        @property
        def description(self) -> str:
            return """执行基本数学运算。
参数：
- expression: 数学表达式（如 "2 + 3"）
"""

        def run(self, expression: str = "", **kwargs) -> str:
            try:
                # 安全的数学表达式计算
                allowed_chars = set("0123456789+-*/(). ")
                if not all(c in allowed_chars for c in expression):
                    return "❌ 表达式包含非法字符"

                result = eval(expression)
                return f"计算结果: {expression} = {result}"
            except Exception as e:
                return f"❌ 计算失败: {str(e)}"

    # 创建Agent并注册自定义工具
    agent = ReActAgent()
    agent.tool_manager.register_tool(CalculatorTool())

    # 测试自定义工具
    response = agent.run("帮我计算 25 * 4 + 10")
    print(f"✅ 答案: {response}")


def example_batch_queries():
    """示例4：批量查询"""
    print("\n" + "=" * 80)
    print("示例4：批量查询（静默模式）")
    print("=" * 80)

    agent = ReActAgent(verbose=False)

    queries = [
        "种崎敦美的生日",
        "Comic Girls的主配角",
        "桐谷华是谁",
    ]

    results = []
    for query in queries:
        print(f"查询中: {query}...")
        response = agent.run(query)
        results.append({"query": query, "answer": response})

    # 打印结果摘要
    print("\n" + "=" * 80)
    print("查询结果摘要:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] 问题: {result['query']}")
        print(f"    答案: {result['answer'][:200]}...")


def example_mcp_integration():
    """示例5：MCP工具集成（示例代码）"""
    print("\n" + "=" * 80)
    print("示例5：MCP工具集成")
    print("=" * 80)

    # 注意：这只是一个示例框架
    # 实际使用需要根据具体的MCP服务器进行调整

    from tools import MCPTool

    class MockMCPClient:
        """模拟的MCP客户端"""

        def call_tool(self, tool_name: str, parameters: dict):
            # 这里应该是实际的MCP调用
            return f"模拟MCP工具 {tool_name} 的返回结果，参数: {parameters}"

    # 创建Agent
    agent = ReActAgent()

    # 创建MCP客户端
    mcp_client = MockMCPClient()

    # 注册MCP工具
    web_search_tool = MCPTool(
        mcp_client=mcp_client,
        tool_name="web_search",
        tool_description="网络搜索工具，可用于搜索最新信息"
    )

    agent.tool_manager.register_tool(web_search_tool)

    # 使用MCP工具
    response = agent.run("搜索最新的动画信息")
    print(f"✅ 答案: {response}")


if __name__ == "__main__":
    import sys

    print("\n🚀 ReAct Agent 示例程序")
    print("=" * 80)

    examples = {
        "1": ("基本查询", example_basic_query),
        "2": ("直接使用工具", example_tool_usage),
        "3": ("自定义工具", example_custom_tool),
        "4": ("批量查询", example_batch_queries),
        "5": ("MCP集成", example_mcp_integration),
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\n运行示例{example_num}: {name}")
            func()
        else:
            print(f"\n❌ 未找到示例 {example_num}")
            print("\n可用示例:")
            for num, (name, _) in examples.items():
                print(f"  {num}. {name}")
    else:
        print("\n可用示例:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\n使用方法: python example.py [示例编号]")
        print("例如: python example.py 1")
