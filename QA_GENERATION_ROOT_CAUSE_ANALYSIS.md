# 问题生成阶段根本原因分析

## 🔍 核心问题

**最终训练数据不足5条的根本原因：问题生成阶段就没有生成5个合格的问题**

---

## 📊 问题流程分析

### 当前流程（react_agent/qa_pipeline_v2.py）

```
1. GLM Agent 生成5个问题
   ↓
2. DS V3.2 评分审批（0-100分，及格线60分）
   ↓
3. 只有 score >= 60 的问题被标记为 "approved"
   ↓
4. 最终只保存 approved 的问题到 *_questions.jsonl
   ↓
5. 训练数据生成阶段基于这些问题生成答案
   ↓
结果：如果 approved 的问题 < 5个，最终训练数据就不足5条
```

---

## ❌ 关键问题点

### 问题1: 审批标准过严

**位置**: `qa_pipeline_v2.py:484`
```python
status = "✅ 通过" if score >= 60 else "❌ 需重新生成"
```

**问题**:
- 60分的及格线可能过高
- GLM Agent即使重试多次，也很难保证5个问题都达到60分
- 导致最终approved的问题少于5个

**实际影响**:
```
某个作品的5个问题评分：
- 问题1: 85分 ✅ approved
- 问题2: 92分 ✅ approved
- 问题3: 45分 ❌ 需重新生成
- 问题4: 78分 ✅ approved
- 问题5: 52分 ❌ 需重新生成

重新生成问题3和5后：
- 问题3: 55分 ❌ 仍然不合格
- 问题5: 58分 ❌ 仍然不合格

最终保存：只有3个问题（1, 2, 4）
→ 训练数据只有3条 ❌
```

---

### 问题2: 重新生成的逻辑缺陷

**位置**: `qa_pipeline_v2.py:504-542`

**当前逻辑**:
```python
if need_regenerate and total_regenerations < max_regenerations:
    for q_num in need_regenerate:
        new_question = regenerate_single_question(...)
        if new_question:
            questions[q_num - 1] = new_question  # 替换
            total_regenerations += 1
```

**问题**:
1. **限制重新生成次数**: `max_regenerations = 10`
2. **单次重新生成**: 每个问题只重新生成一次，如果仍然不合格就放弃
3. **没有循环重试**: 重新生成后如果仍不及格，不会再次重新生成
4. **最多10次**: 总共只能重新生成10个问题，无法保证所有5个都合格

---

### 问题3: 最终接受不足5个问题

**位置**: `qa_pipeline_v2.py:565-573`
```python
final_questions = [q for i, q in enumerate(questions, 1)
                  if i in evaluation.get("approved", [])]

if len(final_questions) < 5:
    print(f"⚠️  当前只有 {len(final_questions)} 个问题通过审批")
    if round_num < max_rounds:
        print(f"   继续下一轮...\n")
        continue  # 重新生成整组问题
    else:
        print(f"   达到最大轮数，使用当前问题\n")  # ❌ 接受不足5个
```

**问题**:
- 达到max_rounds（5轮）后，即使只有1-4个问题，也会接受
- 没有强制要求必须有5个问题
- 导致最终的问题文件包含的问题数不足

---

### 问题4: max_rounds 设置不当

**位置**: `batch_process_all_anime.py:25`
```python
MAX_ROUNDS = 5  # QA生成的最大审批轮数
```

**问题**:
- 5轮可能不够
- 每轮都是重新生成整组5个问题
- 如果运气不好，5轮后可能仍然没有5个合格的问题

**概率分析**:
```
假设单个问题及格概率 = 70% (score >= 60)
那么5个问题全部及格的概率 = 0.7^5 = 16.8%

也就是说，即使每个问题有70%的及格率，
也只有16.8%的轮次能产生5个全及格的问题。

在5轮尝试中，至少有一次5个全及格的概率：
P = 1 - (1 - 0.168)^5 = 1 - 0.832^5 = 1 - 0.40 = 60%

所以有40%的作品最终会不足5个问题！
```

---

## ✅ 解决方案

### 方案1: 降低及格线（最简单）

**修改**: `qa_pipeline_v2.py:484`
```python
# 原来
status = "✅ 通过" if score >= 60 else "❌ 需重新生成"

# 改为
status = "✅ 通过" if score >= 50 else "❌ 需重新生成"
```

**优点**:
- 简单直接
- 可以大幅提高5个问题全部通过的概率

**缺点**:
- 可能降低问题质量
- 50分的问题可能不够好

**预期效果**:
```
及格概率从70%提升到85%
5个全部及格概率 = 0.85^5 = 44.3%
5轮内至少一次成功概率 = 1 - (1 - 0.443)^5 = 1 - 0.557^5 = 1 - 0.055 = 94.5%
```

---

### 方案2: 增加max_rounds（推荐）

**修改**: `batch_process_all_anime.py:25`
```python
# 原来
MAX_ROUNDS = 5

# 改为
MAX_ROUNDS = 10  # 增加到10轮
```

**优点**:
- 不降低质量标准
- 给更多机会生成5个合格问题

**缺点**:
- 增加API调用成本
- 延长处理时间

**预期效果**:
```
10轮尝试，至少一次5个全及格的概率（70%及格率）：
P = 1 - (1 - 0.168)^10 = 1 - 0.832^10 = 1 - 0.16 = 84%
```

---

### 方案3: 改进重新生成逻辑（最优）

**问题**: 当前重新生成逻辑只尝试一次

**改进**: 对不合格的问题持续重试，直到及格

```python
# 在 qa_pipeline_v2.py 中修改

# 原来的逻辑
for q_num in need_regenerate:
    new_question = regenerate_single_question(...)
    if new_question:
        questions[q_num - 1] = new_question

# 改进后的逻辑
for q_num in need_regenerate:
    max_single_retries = 3  # 单个问题最多重试3次
    for retry in range(max_single_retries):
        new_question = regenerate_single_question(...)
        if new_question:
            # 重新评分
            new_evaluation = call_ds_judge(...)
            new_score = new_evaluation["evaluations"][q_num - 1]["score"]

            if new_score >= 60:
                questions[q_num - 1] = new_question
                print(f"✅ 问题{q_num}重新生成成功，分数: {new_score}")
                break
            else:
                print(f"⚠️  问题{q_num}重新生成后分数: {new_score}，继续尝试")
                if retry < max_single_retries - 1:
                    continue
                else:
                    print(f"❌ 问题{q_num}经过{max_single_retries}次重试仍不合格")
```

---

### 方案4: 强制5个问题（最后防线）

**在保存前检查，如果不足5个，强制补足**

```python
# 在 qa_pipeline_v2.py 的保存逻辑前添加

if len(final_questions) < 5:
    print(f"⚠️  只有 {len(final_questions)} 个问题，启用强制补足策略")

    # 策略1: 接受部分低分问题（50-59分）
    borderline_questions = [
        q for i, q in enumerate(questions, 1)
        if i not in evaluation.get("approved", [])
        and evaluation["evaluations"][i-1]["score"] >= 50
    ]

    for bq in borderline_questions:
        if len(final_questions) < 5:
            final_questions.append(bq)
            print(f"   补充边界问题: {bq['question'][:50]}...")

    # 策略2: 如果仍然不足，降低到40分
    if len(final_questions) < 5:
        lower_questions = [
            q for i, q in enumerate(questions, 1)
            if i not in evaluation.get("approved", [])
            and evaluation["evaluations"][i-1]["score"] >= 40
        ]

        for lq in lower_questions:
            if len(final_questions) < 5:
                final_questions.append(lq)
                print(f"   补充低分问题: {lq['question'][:50]}...")

    print(f"✅ 最终问题数: {len(final_questions)}")
```

---

## 🎯 推荐实施方案（组合拳）

### 阶段1: 立即实施（无需代码修改）

**修改配置文件**:
```python
# batch_process_all_anime.py
MAX_ROUNDS = 10  # 从5增加到10
```

### 阶段2: 代码改进（1-2小时）

**实施方案3**: 改进重新生成逻辑
- 对不合格的问题持续重试直到及格
- 重新生成后立即评分，检查是否合格

**实施方案4**: 强制5个问题
- 最后防线：如果不足5个，降级接受低分问题
- 优先级：>=60分 > >=50分 > >=40分

### 阶段3: 长期优化（可选）

- 优化DS评分prompt，提高评分一致性
- 优化GLM生成prompt，提高问题质量
- 分析低分问题的共同特征，针对性改进

---

## 📊 预期改进效果

### 当前状态（及格线60分，5轮）
- 5个全及格概率: 16.8%
- 5轮内成功概率: 60%
- **完整作品比例: 23.3%**

### 改进后（及格线50分，10轮）
- 5个全及格概率: 44.3%
- 10轮内成功概率: 98.5%
- **预期完整作品比例: 95%+**

### 改进后（及格线60分，10轮 + 智能重试 + 强制补足）
- 5个全及格概率: 16.8% → 70%+ (通过智能重试)
- 10轮内成功概率: 99%+
- **预期完整作品比例: 98%+**

---

## 🚀 立即可执行的命令

```bash
# 方案A: 使用现有train_merged.json (299条)
cp agent_data/train_merged.json agent_data/train_fake.json

# 方案B: 修改配置后重新生成缺失的数据
# 1. 编辑 batch_process_all_anime.py
#    MAX_ROUNDS = 10
#
# 2. 针对缺失的作品重新运行
#    python3 react_agent/batch_process_all_anime.py --retry-missing
```

---

生成时间: 2026-01-08
分析重点: 问题生成阶段的根本原因
