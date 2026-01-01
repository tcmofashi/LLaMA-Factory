# GLM-4.7 调用萌娘百科超时问题 - 完整分析报告

## 问题现象
用户报告：GLM-4.7调用萌娘百科时超时，导致问题生成失败，返回0个问题。

---

## 超时流程分析

### 1. 调用链路
```
qa_pipeline_v2.py (subprocess调用)
  └─> agent.py (ReActAgent)
       ├─> 初始化OpenAI客户端
       ├─> 调用GLM-4.7 API
       ├─> GLM-4.7决定使用工具
       └─> 执行moegirl_title_search工具
            ├─> load_dataset()  ← **这里卡住了**
            │   ├─> 尝试连接 HuggingFace
            │   ├─> 重试5次（1s, 2s, 4s, 8s, 8s）
            │   └─> 每次超时 → 累计30秒+
            └─> 工具执行失败 → GLM-4.7无法回答
```

### 2. 超时位置
**第一层超时**：`subprocess.run()` - 外层调用（用户设置的timeout）
**第二层超时**：`agent.py` - OpenAI客户端调用（原默认5秒）
**第三层超时**：`tools.py` - 数据集加载（实际发生位置）

---

## 根本原因

### 原因1: HuggingFace网络不可达
```
Network is unreachable (errno 101)
```
- 数据集尝试连接 `huggingface.co` 检查更新
- 网络不可达，导致重试
- 每次重试间隔：1s → 2s → 4s → 8s → 8s = 23秒+

### 原因2: 数据集加载策略问题
**原始代码**：
```python
dataset = load_dataset("KomeijiForce/moe_girl_wiki", cache_dir=...)
```
- **默认行为**：每次都会尝试连接HuggingFace检查更新
- **问题**：即使有缓存，也会联网验证
- **结果**：网络不可达时每次都超时

### 原因3: OpenAI客户端无超时设置
**原始代码**：
```python
self.client = OpenAI(base_url=base_url, api_key=api_key)
# ❌ 没有设置timeout参数
```
- **默认超时**：5秒（connect）+ 5秒（read）
- **问题**：GLM-4.7处理工具调用可能需要更长时间

---

## 已实施的修复

### 修复1: 增加OpenAI客户端超时 ✅
**文件**: `react_agent/agent.py:90-104`

```python
import httpx

timeout_config = httpx.Timeout(300.0, connect=60.0)
self.client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=timeout_config,  # 连接60秒，读取300秒
)
```

### 修复2: 添加OpenAI错误处理 ✅
**文件**: `react_agent/agent.py:138-160`

```python
try:
    response = self.client.chat.completions.create(...)
except openai.APITimeoutError:
    raise RuntimeError(f"❌ API调用超时（超过300秒）")
except openai.APIError as e:
    raise RuntimeError(f"❌ API错误: {e}")
```

### 修复3: 本地缓存加载萌娘百科数据集 ✅
**文件**: `react_agent/tools.py:53-121`

```python
# 直接从缓存加载，避免联网检查
arrow_files = glob.glob(os.path.join(cache_base, "*/*/*/*.arrow"))
for arrow_file in arrow_files:
    # 解析split名称
    if '-train-' in filename:
        split_files['train'].append(arrow_file)
    # 加载到DatasetDict
    datasets[split_name] = Dataset.from_file(files[0])
```

---

## 当前错误处理机制

### OpenAI API层
```python
try:
    response = client.chat.completions.create(...)
except openai.APITimeoutError:
    return "❌ API调用超时（超过300秒）"
except openai.APIError:
    return "❌ API错误"
except openai.RateLimitError:
    return "❌ API速率限制"
```

### Agent层
```python
try:
    tool_result = tool.run(**params)
except Exception as e:
    return f"❌ 工具执行失败: {e}"
```

### 数据集层
```python
try:
    dataset = load_from_cache()
except Exception as e:
    print(f"⚠️  数据集加载失败: {e}")
    return None  # 工具返回"❌ 数据集加载失败"
```

### GLM-4.7 Agent响应
- 工具失败 → GLM-4.7收到错误消息
- 调用 `skip_anime` 工具
- 跳过该动画

### qa_pipeline_v2.py层
```python
if result.returncode != 0:
    return {"status": "failed", "stage": "question_generation"}
```

---

## 剩余问题与建议

### 问题1: 大数据集加载慢
**现状**：萌娘百科数据集约1.5GB，从.arrow文件加载需要时间
**建议**：
- 实施数据集预加载/预热
- 使用更快的存储（SSD）
- 或者使用更小的索引数据集

### 问题2: GLM-4.7响应慢
**现状**：GLM-4.7处理复杂prompt需要较长时间
**建议**：
- 简化prompt
- 使用更快的模型（如DeepSeek-V3）
- 增加超时时间（已设置300秒）

### 问题3: 没有subprocess timeout
**现状**：`subprocess.run()`没有设置timeout参数
**建议**：
```python
result = subprocess.run(
    [...],
    timeout=600,  # 添加10分钟超时
    capture_output=True,
)
```

---

## 测试建议

### 1. 测试数据集加载
```bash
python -c "
from react_agent.tools import MoegirlTitleSearchTool
tool = MoegirlTitleSearchTool()
result = tool.run('孤独摇滚', max_results=5)
print(result)
"
```

### 2. 测试Agent超时
```bash
# 应该在300秒内完成
timeout 300 python react_agent/agent.py \
  --query "测试" \
  --max-tokens 100 \
  --max-iterations 1
```

### 3. 测试完整流程
```bash
# 监控执行时间
time python react_agent/qa_pipeline_v2.py \
  --anime "孤独摇滚！"
```

---

## 总结

**超时的三个层次**：
1. **数据集加载超时**（30秒+）← 主要原因
2. **OpenAI客户端超时**（原5秒，现已修复为300秒）
3. **subprocess超时**（无限制，建议添加）

**错误处理现状**：
- ✅ OpenAI层：有详细错误分类
- ✅ Agent层：有异常捕获
- ✅ 工具层：返回友好错误信息
- ✅ 应用层：status标记
- ❌ 但都只是**被动失败**，没有**主动重试**

**推荐下一步**：
1. 测试修复后的数据集加载速度
2. 如果仍然慢，考虑使用索引或预加载
3. 为subprocess添加timeout参数
