# GLM-4.7 调用萌娘百科超时问题分析

## 问题现象
GLM-4.7 Agent在调用萌娘百科工具时超时，导致问题生成失败。

## 超时流程分析

### 1. 调用链路
```
qa_pipeline_v2.py
  └─> subprocess.run(agent.py)
       └─> ReActAgent.run()
            └─> _generate_with_openai()
                 └─> client.chat.completions.create()  ← 这里超时
                      ├─> 智谱AI API调用
                      └─> 等待响应...
```

### 2. 超时位置
**文件**: `react_agent/agent.py:89-92`
```python
self.client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    # ❌ 没有设置 timeout 参数！
)
```

### 3. 默认超时设置
OpenAI SDK默认超时：`httpx.Timeout(timeout=5.0, connect=5.0)`
- **连接超时**: 5秒
- **读取超时**: 5秒

### 4. 为什么会超时？

#### 原因1: 萌娘百科数据集过大
- **数据集**: `KomeijiForce/moe_girl_wiki`
- **大小**: 可能有数十万条条目
- **搜索方式**: 线性遍历整个数据集
- **时间复杂度**: O(n)

#### 原因2: GLM-4.7响应慢
- GLM-4.7需要理解prompt
- 决定使用哪个工具
- 生成工具调用JSON
- 可能超过5秒

#### 原因3: 工具执行时间
- 萌娘百科工具是线性搜索
- 搜索"向阳素描×☆☆☆"可能需要遍历很多条目
- 累积时间超过5秒

## 错误处理现状

### 当前错误处理
❌ **没有超时错误处理**
- OpenAI客户端会抛出 `ReadTimeout` 异常
- Agent会崩溃
- subprocess返回错误码
- qa_pipeline_v2.py捕获到错误，但不知道是超时

### 错误传播
```
agent.py: OpenAI ReadTimeout
  └─> subprocess.run() 返回非0退出码
       └─> qa_pipeline_v2.py: result.returncode != 0
            └─> 返回 {"status": "failed", "stage": "question_generation"}
```

## 解决方案

### 方案1: 增加OpenAI客户端超时（推荐）
```python
from openai import OpenAI

self.client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=300.0,  # 5分钟超时
)
```

### 方案2: 使用httpx.Timeout对象
```python
from openai import OpenAI
import httpx

self.client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=httpx.Timeout(300.0, connect=60.0),
)
```

### 方案3: 优化萌娘百科搜索（长期）
- 建立索引
- 使用更快的搜索算法
- 预处理数据集

### 方案4: 添加重试机制
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_with_retry():
    response = self.client.chat.completions.create(...)
```

### 方案5: 更好的错误处理
```python
try:
    response = self.client.chat.completions.create(...)
except openai.ReadTimeout:
    return "❌ API调用超时，请稍后重试"
except openai.APIError as e:
    return f"❌ API错误: {e}"
```

## 推荐实施
1. **立即**: 增加timeout参数到300秒
2. **短期**: 添加更详细的错误处理和日志
3. **中期**: 优化萌娘百科搜索性能
4. **长期**: 考虑缓存搜索结果

## 测试建议
```bash
# 测试是否是超时问题
timeout 60 python react_agent/agent.py --query "测试" --max-tokens 100

# 监控实际响应时间
time python react_agent/qa_pipeline_v2.py --anime "孤独摇滚！"
```
