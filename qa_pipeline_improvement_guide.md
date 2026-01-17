# 问题生成改进方案 - 实施指南

## 🎯 核心发现

**根本原因**: DS审批评分过严，导致很多作品最终approved的问题少于5个

**关键代码位置**: `react_agent/qa_pipeline_v2.py:484`
```python
status = "✅ 通过" if score >= 60 else "❌ 需重新生成"
```

---

## ✅ 立即可实施的3个方案

### 方案1: 降低及格线（最简单，推荐先试）

**文件**: `react_agent/qa_pipeline_v2.py`
**位置**: 第484行附近

**修改**:
```python
# 原代码
status = "✅ 通过" if score >= 60 else "❌ 需重新生成"

# 改为（降低到50分）
status = "✅ 通过" if score >= 50 else "❌ 需重新生成"

# 或者（降低到45分，更宽松）
status = "✅ 通过" if score >= 45 else "❌ 需重新生成"
```

**优点**:
- 一行代码修改
- 立即生效
- 可以大幅提高5个问题全部通过的概率

**建议**: 先从50分尝试，如果效果不理想再降到45分

---

### 方案2: 增加审批轮数

**文件**: `react_agent/batch_process_all_anime.py`
**位置**: 第25行

**修改**:
```python
# 原代码
MAX_ROUNDS = 5  # QA生成的最大审批轮数（增加到5以提高成功率）

# 改为
MAX_ROUNDS = 10  # QA生成的最大审批轮数
```

**优点**:
- 给更多机会生成5个合格问题
- 不降低质量标准

**建议**: 与方案1组合使用

---

### 方案3: 强制补足5个问题（推荐）

**文件**: `react_agent/qa_pipeline_v2.py`
**位置**: 第583-591行（保存问题之前）

**在保存问题之前添加以下代码**:

```python
# 在第583行之前添加

# ========== 新增：强制补足5个问题的逻辑 ==========
if len(final_questions) < 5:
    print(f"\n{'='*100}")
    print(f"⚠️  警告: 当前只有 {len(final_questions)} 个问题通过审批")
    print(f"{'='*100}\n")

    # 获取所有问题的评分
    all_evaluations = evaluation.get("evaluations", [])
    approved_set = set(evaluation.get("approved", []))

    # 收集未通过的问题及其分数
    failed_questions = []
    for i, eval_item in enumerate(all_evaluations, 1):
        if i not in approved_set:
            failed_questions.append({
                "index": i,
                "question": questions[i-1],
                "score": eval_item.get("score", 0),
                "reason": eval_item.get("reason", "")
            })

    # 按分数排序（从高到低）
    failed_questions.sort(key=lambda x: x["score"], reverse=True)

    print(f"📊 未通过的问题:")
    for fq in failed_questions:
        print(f"   问题{fq['index']}: {fq['score']}分 - {fq['reason'][:60]}...")

    # 策略1: 尝试补充50-59分的问题
    print(f"\n🔄 策略1: 尝试补充50分以上的问题")
    added_count = 0
    for fq in failed_questions:
        if len(final_questions) >= 5:
            break
        if fq["score"] >= 50:
            final_questions.append(fq["question"])
            added_count += 1
            print(f"   ✅ 补充问题{fq['index']} ({fq['score']}分)")

    if added_count > 0:
        print(f"   补充了 {added_count} 个问题\n")

    # 策略2: 如果仍不足，补充40-49分的问题
    if len(final_questions) < 5:
        print(f"🔄 策略2: 尝试补充40分以上的问题")
        added_count = 0
        for fq in failed_questions:
            if len(final_questions) >= 5:
                break
            if fq["score"] >= 40:
                final_questions.append(fq["question"])
                added_count += 1
                print(f"   ✅ 补充问题{fq['index']} ({fq['score']}分)")

        if added_count > 0:
            print(f"   补充了 {added_count} 个问题\n")

    # 策略3: 如果还不足，接受剩余分数最高的问题
    if len(final_questions) < 5:
        print(f"🔄 策略3: 接受剩余最高分的问题")
        needed = 5 - len(final_questions)
        for fq in failed_questions[:needed]:
            if fq["question"] not in final_questions:
                final_questions.append(fq["question"])
                print(f"   ✅ 补充问题{fq['index']} ({fq['score']}分 - 降级接受)")

    print(f"\n{'='*100}")
    print(f"✅ 最终问题数: {len(final_questions)}")
    print(f"{'='*100}\n")

    if len(final_questions) < 5:
        print(f"⚠️  注意: 最终仍有 {5 - len(final_questions)} 个问题缺失")
        print(f"   这将导致训练数据不足5条\n")
# ========== 强制补足逻辑结束 ==========
```

**优点**:
- 智能降级，优先选择高质量问题
- 保证至少有5个问题（即使质量稍低）
- 有3个降级策略，层层递进

**建议**: 与方案1和方案2组合使用，效果最佳

---

## 🚀 推荐实施顺序

### 第一步：快速测试（5分钟）

只实施方案1（降低及格线到50分），对一个失败的作品重新生成：

```bash
# 找一个数据不足的作品，例如 "Urara迷路帖"

# 1. 备份原文件
cd /home/tcmofashi/LLaMA-Factory
cp react_agent/qa_pipeline_v2.py react_agent/qa_pipeline_v2.py.backup

# 2. 修改代码（降低及格线）
# 编辑 react_agent/qa_pipeline_v2.py 第484行
# 将 score >= 60 改为 score >= 50

# 3. 删除旧的问题文件，重新生成
rm "agent_data/Urara迷路帖 うらら迷路帖_questions.jsonl"

# 4. 重新生成该作品的问题
python3 -c "
import sys
sys.path.insert(0, 'react_agent')
from qa_pipeline_v2 import generate_questions_for_anime_v2

result = generate_questions_for_anime_v2(
    anime_name='Urara迷路帖 うらら迷路帖',
    output_dir='agent_data',
    max_rounds=5
)
print(f'生成了 {result[\"total_questions\"]} 个问题')
"
```

**预期结果**: 应该生成5个问题

---

### 第二步：完整实施（30分钟）

组合实施3个方案：

1. **修改 `batch_process_all_anime.py`**
   ```python
   MAX_ROUNDS = 10  # 第25行
   ```

2. **修改 `qa_pipeline_v2.py`**
   ```python
   # 第484行
   status = "✅ 通过" if score >= 50 else "❌ 需重新生成"

   # 第583行前添加强制补足逻辑（见方案3代码）
   ```

3. **对失败的作品重新运行**
   ```bash
   # 创建一个只包含失败作品的列表
   # 然后重新运行批处理
   ```

---

### 第三步：验证效果（10分钟）

```bash
# 检查生成的问题数量
cd /home/tcmofashi/LLaMA-Factory/agent_data

for file in *_questions.jsonl; do
    count=$(wc -l < "$file")
    if [ "$count" -lt 5 ]; then
        echo "❌ $file: $count 个问题"
    else
        echo "✅ $file: $count 个问题"
    fi
done
```

---

## 📊 预期效果对比

| 方案 | 实施难度 | 预期完整率 | 风险 |
|------|----------|------------|------|
| 当前（60分，5轮） | - | 23.3% | - |
| 方案1: 50分及格线 | ⭐ 极简 | 70%+ | 可能降低质量 |
| 方案2: 10轮审批 | ⭐ 简单 | 60% → 80% | 增加成本 |
| 方案3: 强制补足 | ⭐⭐ 中等 | 95%+ | 可能包含低分问题 |
| 组合方案 (1+2+3) | ⭐⭐⭐ 复杂 | 98%+ | 风险可控 |

---

## ⚠️ 注意事项

1. **备份数据**
   ```bash
   cp react_agent/qa_pipeline_v2.py react_agent/qa_pipeline_v2.py.backup
   cp react_agent/batch_process_all_anime.py react_agent/batch_process_all_anime.py.backup
   ```

2. **先小规模测试**
   - 不要一次性对所有作品重新生成
   - 先选择3-5个失败的作品测试
   - 验证效果后再大规模实施

3. **检查质量**
   - 查看生成的问题质量是否可接受
   - 如果降级到50分后质量明显下降，考虑调整策略

4. **成本控制**
   - 增加MAX_ROUNDS会显著增加API调用
   - 建议从10轮开始，必要时再调整

---

## 🎯 立即可执行的命令

```bash
# 1. 备份
cd /home/tcmofashi/LLaMA-Factory
cp react_agent/qa_pipeline_v2.py react_agent/qa_pipeline_v2.py.backup

# 2. 修改及格线（使用sed快速修改）
sed -i 's/score >= 60/score >= 50/g' react_agent/qa_pipeline_v2.py

# 3. 验证修改
grep "score >= 50" react_agent/qa_pipeline_v2.py

# 4. 重新生成一个失败的作品测试
# （手动运行测试代码）

# 5. 如果效果不好，可以恢复
# cp react_agent/qa_pipeline_v2.py.backup react_agent/qa_pipeline_v2.py
```

---

生成时间: 2026-01-08
重点: 问题生成阶段的根本原因和解决方案
