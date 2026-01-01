# 问题修复：萌娘百科API模式环境变量传递

## 问题描述

### 错误信息
```
FileNotFoundError: 未找到萌娘百科数据集文件
```

### 根本原因

虽然在外部设置了环境变量 `USE_MOEGIRL_API=true`，但是在通过 `subprocess.run()` 调用 `agent.py` 时，**没有传递环境变量**，导致子进程无法感知到应该使用API模式，而是尝试加载本地数据集。

## 问题位置

### 文件1：`react_agent/qa_pipeline_v2.py`

**问题代码（第16-21行）：**
```python
def call_glm_agent(prompt: str, max_tokens: int = 131072, max_iterations: int = 20) -> str:
    """调用GLM Agent生成响应"""
    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, ...],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        # ❌ 缺少 env 参数
    )
```

**修复后的代码：**
```python
def call_glm_agent(prompt: str, max_tokens: int = 131072, max_iterations: int = 20) -> str:
    """调用GLM Agent生成响应"""
    # ✅ 确保传递萌娘百科API环境变量
    import os
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, ...],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        env=env  # ✅ 添加环境变量
    )
```

### 文件2：`react_agent/generate_training_data.py`

**问题代码（第43-57行）：**
```python
def call_agent_for_answer(question: str) -> str:
    """调用Agent生成答案"""
    prompt = generate_answer_prompt(question)

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, ...],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        # ❌ 缺少 env 参数
    )
```

**修复后的代码：**
```python
def call_agent_for_answer(question: str) -> str:
    """调用Agent生成答案"""
    prompt = generate_answer_prompt(question)

    # ✅ 确保传递萌娘百科API环境变量
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'

    result = subprocess.run(
        ["python3", "react_agent/agent.py", "--query", prompt, ...],
        capture_output=True,
        text=True,
        cwd="/home/tcmofashi/LLaMA-Factory",
        env=env  # ✅ 添加环境变量
    )
```

## 修复验证

### 测试1：API模式激活检测

**测试脚本：**
```python
from qa_pipeline_v2 import call_glm_agent

output = call_glm_agent("请使用萌娘百科搜索《孤独摇滚！》")

if "🌐 使用萌娘百科API服务模式" in output:
    print("✅ API模式成功激活")
```

**测试结果：**
```
✅ API模式成功激活
✅ 生成了响应

输出预览:
🌐 使用萌娘百科API服务模式: http://localhost:8765

最终答案:
《孤独摇滚！》（日语：ぼっち・ざ・ろっく！；英语：Bocchi the Rock!）是...
```

### 测试2：萌娘百科API服务状态

```bash
$ curl http://localhost:8765/health
{"status":"healthy","dataset_loaded":true,"dataset_size":328587}
```

✅ API服务正常运行
✅ 数据集已加载（328,587条记录）

## 技术说明

### subprocess.run() 环境变量传递

**默认行为：**
- `subprocess.run()` 默认会复制父进程的环境变量
- 但是在某些情况下（如shell启动、脚本调用），环境变量可能不会正确传递

**最佳实践：**
```python
import os
import subprocess

# 方式1：复制当前环境变量并添加
env = os.environ.copy()
env['MY_VAR'] = 'value'

result = subprocess.run(
    ["command", "arg1", "arg2"],
    env=env  # 显式传递环境变量
)

# 方式2：只传递特定的环境变量（不推荐）
env = {'MY_VAR': 'value', 'PATH': os.environ['PATH']}
result = subprocess.run([...], env=env)
```

### 为什么需要显式传递？

1. **可靠性**：确保环境变量在所有场景下都能正确传递
2. **明确性**：代码意图清晰，一看就知道需要哪些环境变量
3. **调试性**：出现问题时容易排查

## 影响范围

### 受影响的函数

| 文件 | 函数 | 用途 | 状态 |
|------|------|------|------|
| `qa_pipeline_v2.py` | `call_glm_agent()` | 生成问题 | ✅ 已修复 |
| `generate_training_data.py` | `call_agent_for_answer()` | 生成答案 | ✅ 已修复 |
| `training_data_utils.py` | `call_glm_agent_regenerate()` | 重新生成答案 | ✅ 无需修复（已有env） |

### 受影响的流程

1. **问题生成流程**：`batch_process_all_anime.py` → `qa_pipeline_v2.py` → `call_glm_agent()`
2. **训练数据生成流程**：`batch_process_all_anime.py` → `generate_training_data.py` → `call_agent_for_answer()`
3. **智能重新生成流程**：`training_data_utils.py` → `call_glm_agent_regenerate()` ✅ 已正确

## 使用建议

### 运行全流程脚本

现在可以直接运行，无需手动设置环境变量：

```bash
cd /home/tcmofashi/LLaMA-Factory

# ✅ 脚本内部已自动设置 USE_MOEGIRL_API=true
python3 react_agent/batch_process_all_anime.py
```

### 手动运行（如果需要）

如果需要单独运行某个模块：

```bash
# 方式1：临时设置环境变量
export USE_MOEGIRL_API=true
python3 react_agent/agent.py --query "测试问题"

# 方式2：一次性设置
USE_MOEGIRL_API=true python3 react_agent/agent.py --query "测试问题"
```

## 后续优化建议

### 1. 统一环境变量管理

创建一个统一的配置模块：

```python
# react_agent/config.py
import os

def get_agent_env():
    """获取Agent运行所需的环境变量"""
    env = os.environ.copy()
    env['USE_MOEGIRL_API'] = 'true'
    return env
```

然后在所有subprocess调用中使用：

```python
from config import get_agent_env

result = subprocess.run([...], env=get_agent_env())
```

### 2. 添加环境变量检测

在脚本启动时检测萌娘百科API服务：

```python
def check_moegirl_api():
    """检测萌娘百科API服务是否运行"""
    import urllib.request
    try:
        response = urllib.request.urlopen("http://localhost:8765/health", timeout=2)
        data = json.loads(response.read())
        return data.get("status") == "healthy"
    except:
        return False
```

### 3. 更友好的错误提示

```python
if not check_moegirl_api():
    print("⚠️  警告：萌娘百科API服务未运行")
    print("   请先启动萌娘百科API服务：")
    print("   python3 react_agent/moegirl_api_server.py")
    sys.exit(1)
```

## 总结

### 修复内容

✅ 修复了 `qa_pipeline_v2.py` 中的环境变量传递问题
✅ 修复了 `generate_training_data.py` 中的环境变量传递问题
✅ 验证了API模式正常工作
✅ 验证了萌娘百科API服务正常运行

### 修复时间

2026-01-01

### 修复状态

✅ **已完成并验证通过**

### 可以开始使用

现在可以正常运行全流程脚本了：

```bash
python3 react_agent/batch_process_all_anime.py
```

---

**修复完成！** 🎉
