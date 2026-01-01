# ReAct Agent

基于ReAct（Reasoning + Acting）范式的AI Agent，支持工具调用和推理-行动循环。

## 最新更新 (v0.2.0)

✨ **工具解耦**：萌娘百科搜索已解耦为三个独立工具
- `moegirl_title_search` - 标题搜索（精确查找）
- `moegirl_keyword_search` - 关键词搜索（内容检索）
- `moegirl_get_entry` - 获取完整条目（查看详情）

详见：[萌娘百科工具指南](MOEGIRL_TOOLS_GUIDE.md) | [快速参考](QUICKREF.md)

---

## 功能特性

- 🔧 **工具调用**：萌娘百科搜索（3种方式）、MCP工具等
- 🤔 **推理循环**：交替进行思考和行动
- 🔌 **灵活配置**：通过TOML配置文件管理模型和API
- 💬 **交互模式**：支持命令行交互问答
- ⚡ **高性能**：数据集单例模式，避免重复加载

## 目录结构

```
react_agent/
├── agent.py           # ReAct Agent主程序
├── tools.py           # 工具集定义
├── config.toml        # 配置文件（指向上级目录）
├── README.md          # 说明文档
└── example.py         # 使用示例
```

## 配置文件

配置文件位于 `/home/tcmofashi/LLaMA-Factory/config.toml`：

```toml
[api]
provider = "Pro/zai-org/GLM-4.7"
api_key = "your-api-key"
base_url = "https://api.siliconflow.cn/v1"

[model]
name = "GLM-4.7"
organization = "Pro/zai-org"
```

## 安装依赖

```bash
pip install openai datasets toml
```

## 使用方法

### 1. 命令行查询

```bash
# 直接查询
python agent.py --query "种崎敦美的代表作是什么？"

# 使用简短参数
python agent.py -q "长谷川育美的生日是哪天？"

# 指定配置文件
python agent.py -q "Comic Girls的主角是谁" -c /path/to/config.toml
```

### 2. 交互模式

```bash
# 进入交互模式
python agent.py --interactive

# 或使用简短参数
python agent.py -i
```

交互模式示例：
```
🔹 你: 种崎敦美为什么被称为华哥？
[Agent会自动使用萌娘百科工具搜索并回答]

🔹 你: Comic Girls这部动画讲的是什么？
[Agent会搜索相关条目并总结]

🔹 你: quit
👋 退出交互模式
```

### 3. Python API

```python
from agent import ReActAgent

# 创建Agent
agent = ReActAgent(
    config_path="/home/tcmofashi/LLaMA-Factory/config.toml",
    max_iterations=10,
    verbose=True,
)

# 查询问题
response = agent.run("种崎敦美的生日和代表作品是什么？")
print(response)
```

## 可用工具

### 萌娘百科搜索工具套件

#### 1. moegirl_title_search (标题搜索)

在条目标题中精确搜索，适用于已知确切名称的场景。

**参数：**
- `title`: 标题关键词（必需）
- `max_results`: 最大返回结果数（默认5）

**示例：**
```python
from react_agent import MoegirlTitleSearchTool

tool = MoegirlTitleSearchTool()
result = tool.run(title="Comic Girls", max_results=3)
```

#### 2. moegirl_keyword_search (关键词搜索)

在条目正文中搜索关键词，适用于不确定标题的场景。

**参数：**
- `keyword`: 搜索关键词（必需）
- `max_results`: 最大返回结果数（默认5）

**示例：**
```python
from react_agent import MoegirlKeywordSearchTool

tool = MoegirlKeywordSearchTool()
result = tool.run(keyword="桐谷华", max_results=3)
```

#### 3. moegirl_get_entry (获取完整条目)

获取指定索引的完整条目内容。

**参数：**
- `index`: 条目索引（必需）
- `max_length`: 最大返回字符数（默认5000）

**示例：**
```python
from react_agent import MoegirlGetEntryTool

tool = MoegirlGetEntryTool()
result = tool.run(index=126814, max_length=2000)
```

### MCP工具 (MCPTool)

通过MCP协议调用外部工具。

---

详细使用指南请参考：[萌娘百科工具指南](MOEGIRL_TOOLS_GUIDE.md)

## ReAct循环工作原理

```
用户问题
    ↓
[思考] → 分析问题，决定需要什么信息
    ↓
[行动] → 选择并执行工具
    ↓
[观察] → 获取工具返回结果
    ↓
[思考] → 分析结果，判断是否需要更多信息
    ↓
[行动] → 继续使用工具 或 [回答] → 给出最终答案
```

## 扩展工具

### 添加自定义工具

```python
from tools import Tool, ToolManager

class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "我的自定义工具"

    def run(self, **kwargs):
        return "工具执行结果"

# 注册工具
agent = ReActAgent()
agent.tool_manager.register_tool(MyCustomTool())
```

### 添加MCP工具

```python
from tools import MCPTool

# 假设有一个MCP客户端
mcp_client = ...  # 初始化MCP客户端

# 创建MCP工具包装器
mcp_tool = MCPTool(
    mcp_client=mcp_client,
    tool_name="mcp_search",
    tool_description="MCP搜索工具"
)

# 注册到Agent
agent.tool_manager.register_tool(mcp_tool)
```

## 注意事项

1. **配置文件**：确保config.toml中的API密钥有效
2. **网络连接**：使用在线API需要网络连接
3. **数据集路径**：萌娘百科数据集需要预先下载到指定路径
4. **内存占用**：本地模型需要足够的显存

## 故障排查

### 模型加载失败
- 检查config.toml配置
- 确认API密钥有效
- 查看网络连接

### 工具调用失败
- 检查数据集路径
- 确认工具参数格式正确（JSON格式）
- 查看详细错误信息（去掉--quiet参数）

## 许可证

MIT License
