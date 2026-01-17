# 全流程脚本训练数据累积说明

**更新日期**: 2026-01-01
**修改文件**:
- `/home/tcmofashi/LLaMA-Factory/react_agent/generate_training_data.py`
- `/home/tcmofashi/LLaMA-Factory/react_agent/batch_process_all_anime.py`

---

## 修改目的

让所有动画的训练数据累积到一个总的 `train_fake.json` 文件中，而不是每次覆盖。

---

## 修改前的问题

### 原始流程

```
处理动画1 → 生成训练数据 → 写入 train_fake.json
处理动画2 → 生成训练数据 → 覆盖 train_fake.json ❌
处理动画3 → 生成训练数据 → 覆盖 train_fake.json ❌
```

**问题**：每次处理新动画时，`train_fake.json` 都被覆盖，只保留最后一个动画的数据。

---

## 修改后的方案

### 新流程

```
处理动画1 → 生成训练数据 → 保存到 {anime1}_train.json
处理动画2 → 生成训练数据 → 保存到 {anime2}_train.json
处理动画3 → 生成训练数据 → 保存到 {anime3}_train.json
                            ↓
                    全部完成后累积
                            ↓
              合并所有 → train_fake.json ✅
```

### 关键改进

1. **每个动画保存单独文件**：`{anime_name}_train.json`
2. **返回训练数据**：`generate_training_data_from_questions()` 返回生成的数据
3. **累积合并**：在所有动画处理完成后，合并所有数据到 `train_fake.json`

---

## 详细修改

### 1. `generate_training_data.py`

#### 修改 `save_training_data()` 函数

**修改前**：
```python
def save_training_data(results: List[Dict], output_dir: str, answer_record_dir: str):
    ...
    # 保存伪造格式（JSON - OpenAI chat格式，带system prompt）
    fake_format_file = os.path.join(output_dir, "train_fake.json")

    with open(fake_format_file, 'w', encoding='utf-8') as f:
        json.dump(fake_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 伪造格式: {fake_format_file} ({len(fake_data)} 条)")
```

**修改后**：
```python
def save_training_data(results: List[Dict], output_dir: str, answer_record_dir: str, anime_name: str = None):
    ...
    # 保存单个动画的训练数据（单独的JSON文件）
    anime_train_file = os.path.join(output_dir, f"{anime_name}_train.json")

    with open(anime_train_file, 'w', encoding='utf-8') as f:
        json.dump(fake_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 单个动画训练数据: {anime_train_file} ({len(fake_data)} 条)")

    return fake_data  # 返回生成的训练数据，用于累积
```

**改进点**：
- ✅ 保存到 `{anime_name}_train.json` 而不是 `train_fake.json`
- ✅ 返回 `fake_data` 供累积使用
- ✅ 添加 `anime_name` 参数用于文件命名

#### 修改 `generate_training_data_from_questions()` 函数

**修改前**：
```python
def generate_training_data_from_questions(
    questions_file: str,
    output_dir: str,
    max_workers: int = 5
):
    ...
    save_training_data(results, output_dir, answer_record_dir)
```

**修改后**：
```python
def generate_training_data_from_questions(
    questions_file: str,
    output_dir: str,
    max_workers: int = 5,
    anime_name: str = None  # 新增参数
):
    ...
    # 保存结果并返回训练数据
    train_data = save_training_data(results, output_dir, answer_record_dir, anime_name)

    return train_data  # 返回训练数据
```

**改进点**：
- ✅ 添加 `anime_name` 参数
- ✅ 调用 `save_training_data` 时传入动画名称
- ✅ 返回生成的训练数据

---

### 2. `batch_process_all_anime.py`

#### 修改 `process_single_anime()` 函数

**新增字段**：
```python
result = {
    ...
    "train_data": None,  # 新增：保存该动画生成的训练数据
    ...
}
```

**接收训练数据**：
```python
# 调用训练数据生成函数，传入动画名称，并获取返回的训练数据
train_data = generate_training_data_from_questions(
    questions_file=questions_file,
    output_dir=OUTPUT_DIR,
    anime_name=anime_name  # 传入动画名称
)

result["train_success"] = True
result["train_questions"] = len(train_data)
result["train_data"] = train_data  # 保存训练数据
```

**改进点**：
- ✅ 接收并保存每个动画生成的训练数据
- ✅ 传入动画名称用于文件命名

#### 修改 `main()` 函数

**新增累积逻辑**：
```python
# ========== 累积所有训练数据到 train_fake.json ==========
print(f"\n{'='*100}")
print(f"💾 累积所有训练数据到 train_fake.json")
print(f"{'='*100}\n")

# 收集所有成功的训练数据
all_train_data = []
for r in results:
    if r["train_success"] and r["train_data"]:
        all_train_data.extend(r["train_data"])

# 保存到总的 train_fake.json
if all_train_data:
    train_fake_file = os.path.join(OUTPUT_DIR, "train_fake.json")
    with open(train_fake_file, 'w', encoding='utf-8') as f:
        json.dump(all_train_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {len(all_train_data)} 条训练数据到: {train_fake_file}")
    print(f"✅ 涵盖 {success_count} 个动画的训练数据")
```

**改进点**：
- ✅ 收集所有动画的训练数据
- ✅ 合并到一个总的 `train_fake.json`
- ✅ 显示累积的数据量

---

## 输出文件结构

### 处理3个动画后的文件结构

```
agent_data/
├── anime.txt                                    # 动画列表
├── train_fake.json                             # 总训练数据（15条 = 5×3）✅ 新增
│
├── 动画1_questions.jsonl                       # 动画1的问题
├── 动画1_train.json                            # 动画1的训练数据（5条）
├── 动画1_questions.json                         # 动画1的问题集合
│
├── 动画2_questions.jsonl                       # 动画2的问题
├── 动画2_train.json                            # 动画2的训练数据（5条）
├── 动画2_questions.json                         # 动画2的问题集合
│
├── 动画3_questions.jsonl                       # 动画3的问题
├── 动画3_train.json                            # 动画3的训练数据（5条）
├── 动画3_questions.json                         # 动画3的问题集合
│
├── answer_record/
│   ├── 动画1_full.txt                          # 动画1的完整记录
│   ├── 动画2_full.txt                          # 动画2的完整记录
│   └── 动画3_full.txt                          # 动画3的完整记录
│
├── batch_progress.json                         # 批量处理进度
└── batch_summary.json                          # 批量处理摘要
```

---

## 使用示例

### 运行全流程

```bash
cd /home/tcmofashi/LLaMA-Factory
python3 react_agent/batch_process_all_anime.py
```

### 输出示例

```
####################################################################################################
# 全流程批量处理动画
# 数据源: /home/tcmofashi/LLaMA-Factory/agent_data/anime.txt
# 输出目录: /home/tcmofashi/LLaMA-Factory/agent_data
####################################################################################################

📊 共 73 个动画需要处理

...

====================================================================================================
📊 批量处理完成
====================================================================================================

总计: 73 个动画
✅ 成功: 70 个
❌ 失败: 3 个
📝 生成问题: 350 个

====================================================================================================
💾 累积所有训练数据到 train_fake.json
====================================================================================================

✅ 已保存 350 条训练数据到: /home/tcmofashi/LLaMA-Factory/agent_data/train_fake.json
✅ 涵盖 70 个动画的训练数据
```

---

## 优势

### 1. 数据不丢失

- ✅ 每个动画的训练数据都保存在单独的文件中
- ✅ 中断后可以继续，不会丢失已处理的数据
- ✅ 支持断点续传

### 2. 灵活性

- ✅ 可以单独查看某个动画的训练数据
- ✅ 可以选择性合并某些动画的数据
- ✅ 便于调试和验证

### 3. 可追溯

- ✅ 每个动画的问题、答案、完整记录都有单独的文件
- ✅ 可以查看每个动画的处理进度
- ✅ 便于质量检查

---

## train_fake.json 格式

### 内容结构

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你是一位资深的ACG爱好者，对动画、漫画、游戏等作品非常了解。"
      },
      {
        "role": "user",
        "content": "问题内容..."
      },
      {
        "role": "assistant",
        "content": "答案内容..."
      }
    ]
  },
  {
    "messages": [
      {
        "role": "system",
        "content": "你是ACG音乐达人，熟悉动漫OP、ED、插入歌等相关音乐作品。"
      },
      {
        "role": "user",
        "content": "问题内容..."
      },
      {
        "role": "assistant",
        "content": "答案内容..."
      }
    ]
  },
  ...
]
```

### 特点

- ✅ **OpenAI Chat 格式**：标准的三轮对话格式
- ✅ **包含 System Prompt**：每个对话都有专业的 ACG 领域 system prompt
- ✅ **多样化**：41 个不同的 system prompt 随机选择
- ✅ **与具体作品无关**：system prompt 通用，不包含作品名

---

## 数据统计

### 73个动画的预期数据量

- **问题数**：73 × 5 = **365 个问题**
- **训练数据**：73 × 5 = **365 条对话**
- **文件大小估算**：约 10-20 MB（取决于答案长度）

### 单个动画的数据

- **问题文件**：`{anime_name}_questions.jsonl`（5行）
- **训练数据**：`{anime_name}_train.json`（5条对话）
- **完整记录**：`answer_record/{anime_name}_full.txt`
- **问题集合**：`{anime_name}_questions.json`

---

## 总结

### 核心改进

- ✅ **从覆盖模式 → 累积模式**
- ✅ **每个动画单独保存 + 最终合并**
- ✅ **支持断点续传**
- ✅ **数据可追溯**

### 使用方法

```bash
cd /home/tcmofashi/LLaMA-Factory
python3 react_agent/batch_process_all_anime.py
```

最终输出：
- **train_fake.json**：包含所有动画的训练数据（累积）
- **{anime_name}_train.json**：每个动画的单独训练数据

---

**文档位置**: `/home/tcmofashi/LLaMA-Factory/BATCH_PIPELINE_ACCUMULATION.md`
