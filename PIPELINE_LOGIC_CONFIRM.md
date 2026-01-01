# 全流程脚本逻辑确认

## 脚本位置
`/home/tcmofashi/LLaMA-Factory/react_agent/batch_process_all_anime.py`

## 执行流程

### 总体流程
```
开始
  ↓
加载动画列表 (anime.txt)
  ↓
对每个动画：
  ├─ 步骤1: 生成问题 (或使用已有问题)
  │   ├─ 检查 {anime_name}_questions.jsonl 是否存在
  │   ├─ 存在 → 跳过生成，读取已有问题
  │   └─ 不存在 → 调用 GLM-4.7 生成问题
  │
  └─ 步骤2: 生成训练数据
      ├─ 使用步骤1的问题文件
      └─ 调用 DeepSeek 清洗 + 生成训练数据
  ↓
结束
```

## 核心逻辑代码

### 步骤1：问题生成（或读取）

```python
# 检查是否已存在问题文件
existing_questions_file = os.path.join(OUTPUT_DIR, f"{anime_name}_questions.jsonl")

if os.path.exists(existing_questions_file):
    # 读取已有问题
    print(f"✅ 跳过问题生成，使用已有问题: {question_count}个问题")
else:
    # 生成新问题
    qa_result = generate_questions_for_anime_v2(
        anime_name=anime_name,
        output_dir=OUTPUT_DIR,
        max_rounds=MAX_ROUNDS
    )
```

### 步骤2：训练数据生成

```python
# 生成训练数据
generate_training_data_from_questions(
    questions_file=questions_file,  # 使用步骤1的问题文件
    output_dir=OUTPUT_DIR
)
```

## 逻辑确认

### ✅ 执行顺序正确

**1. 先生成问题**
- 调用 `qa_pipeline_v2.py` 的 `generate_questions_for_anime_v2()`
- 使用 GLM-4.7 Agent 生成 5 个问题
- 保存到 `{anime_name}_questions.jsonl`

**2. 再生成训练数据**
- 调用 `generate_training_data.py` 的 `generate_training_data_from_questions()`
- 读取步骤1生成的问题文件
- 使用 GLM-4.7 Agent 生成答案
- 使用 DeepSeek-V3.2 清洗答案（必要时触发重新生成）
- 生成多样化 system prompt
- 保存到 `train_fake.json`

### ✅ 已有问题检测正确

**检测逻辑：**
- 文件路径：`{OUTPUT_DIR}/{anime_name}_questions.jsonl`
- 如果文件存在 → 跳过问题生成
- 如果文件不存在 → 生成新问题

**实际测试：**
```bash
# 测试1：没有问题文件的动画
向阳素描×365 → 问题文件不存在 → 生成问题 → 生成训练数据

# 测试2：有问题文件的动画
孤独摇滚！ → 问题文件存在（5个问题）→ 跳过问题生成 → 直接生成训练数据
```

## 已生成问题的动画列表

以下动画已有问题文件，运行全流程脚本时会**跳过问题生成，直接生成训练数据**：

| 序号 | 动画名称 | 问题文件 | 问题数量 |
|------|----------|----------|----------|
| 1 | BanG Dream! It's MyGO!!!!! | ✅ 已存在 | - |
| 2 | NEW GAME!! | ✅ 已存在 | - |
| 3 | Re：从零开始的异世界生活 | ✅ 已存在 | - |
| 4 | 孤独摇滚！ | ✅ 已存在 | 5 |
| 5 | 摇曳露营△ | ✅ 已存在 | - |
| 6 | 向阳素描×365 | ✅ 已存在 | 5 |
| ... | (更多动画) | ✅ 已存在 | - |

## 使用示例

### 场景1：处理全新动画（没有问题）

```bash
# anime.txt 包含：
# 《魔法少女小圆》

python3 react_agent/batch_process_all_anime.py
```

**执行流程：**
1. 检查 `魔法少女小圆_questions.jsonl` → 不存在
2. 调用 GLM-4.7 生成问题 → 保存到 `魔法少女小圆_questions.jsonl`
3. 调用 GLM-4.7 生成答案 + DeepSeek 清洗 → 保存到 `train_fake.json`

### 场景2：处理已有问题的动画

```bash
# anime.txt 包含：
# 《孤独摇滚！》

python3 react_agent/batch_process_all_anime.py
```

**执行流程：**
1. 检查 `孤独摇滚！_questions.jsonl` → **已存在**
2. **跳过问题生成**，读取已有的 5 个问题
3. 调用 GLM-4.7 生成答案 + DeepSeek 清洗 → 保存到 `train_fake.json`

### 场景3：混合处理

```bash
# anime.txt 包含：
# 《魔法少女小圆》（没有问题）
# 《孤独摇滚！》（有问题）

python3 react_agent/batch_process_all_anime.py
```

**执行流程：**
- 魔法少女小圆：
  1. 生成问题 → 保存
  2. 生成训练数据

- 孤独摇滚！：
  1. **跳过问题生成**，读取已有问题
  2. 生成训练数据

## 优势

### ✅ 避免重复生成
- 已生成问题的动画不会重新生成问题
- 节省时间和API调用成本

### ✅ 支持断点续传
- 如果中途中断，下次运行会跳过已完成的动画
- 进度保存在 `batch_progress.json`

### ✅ 灵活重新生成
- 如果需要重新生成问题，只需删除对应的问题文件
- 下次运行时会重新生成

## 关键文件

| 文件 | 说明 |
|------|------|
| `react_agent/batch_process_all_anime.py` | 全流程脚本 |
| `react_agent/qa_pipeline_v2.py` | 问题生成逻辑 |
| `react_agent/generate_training_data.py` | 训练数据生成逻辑 |
| `react_agent/training_data_utils.py` | DeepSeek清洗工具 |
| `agent_data/anime.txt` | 动画列表 |
| `agent_data/{anime}_questions.jsonl` | 问题文件 |
| `agent_data/train_fake.json` | 训练数据 |
| `agent_data/batch_progress.json` | 进度记录 |
| `agent_data/batch_summary.json` | 最终摘要 |

## 总结

**全流程脚本的执行逻辑完全符合要求：**

✅ **先生成问题，再生成训练数据**
- 步骤1：问题生成（或读取已有）
- 步骤2：训练数据生成

✅ **对于已生成问题的作品，直接生成训练数据**
- 自动检测 `{anime}_questions.jsonl` 是否存在
- 存在则跳过问题生成，直接使用
- 不存在则先生成问题，再生成训练数据

✅ **支持批量处理和断点续传**
- 一次处理多个动画
- 中断后可继续，不会重复生成

---

**确认时间**: 2026-01-01
**确认状态**: ✅ 逻辑正确
**可以放心使用**: ✅ 是
