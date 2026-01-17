# 自动重试失败动画功能说明

**功能**: 自动识别未成功生成问题或训练数据的动画，并重新生成

---

## 功能概述

`retry_failed_anime.py` 脚本可以：

1. **自动识别失败的动画**
   - 缺少问题文件（`_questions.jsonl`）
   - 问题数量不足（少于5个）
   - 缺少训练数据文件（`_train.json`）
   - 训练数据不足（少于5条）
   - 从 `batch_summary.json` 读取失败记录

2. **自动重新生成**
   - 仅重新生成问题
   - 仅重新生成训练数据
   - 重新生成完整流程（问题 + 训练数据）

3. **保存重试结果**
   - 保存到 `retry_result.json`
   - 详细记录每个动画的重试状态

---

## 使用方法

### 模式1: 自动模式（推荐）⭐

自动识别所有失败的动画并尝试修复：

```bash
cd /home/tcmofashi/LLaMA-Factory
python3 react_agent/retry_failed_anime.py --mode auto
```

**功能**:
- ✅ 自动识别缺少问题的动画 → 重新生成问题
- ✅ 自动识别缺少训练数据的动画 → 重新生成训练数据
- ✅ 从 `batch_summary.json` 读取失败记录

---

### 模式2: 仅重新生成问题

只处理缺少问题的动画：

```bash
python3 react_agent/retry_failed_anime.py --mode questions
```

**功能**:
- 识别缺少 `_questions.jsonl` 的动画
- 识别问题数量不足的动画
- 读取 `batch_summary.json` 中问题生成失败的记录
- 重新生成问题（5轮重试）

---

### 模式3: 仅重新生成训练数据

只处理缺少训练数据的动画：

```bash
python3 react_agent/retry_failed_anime.py --mode train
```

**功能**:
- 识别缺少 `_train.json` 的动画
- 识别训练数据不足的动画
- 读取 `batch_summary.json` 中训练数据生成失败的记录
- 重新生成训练数据（5个并发worker）

---

### 模式4: 完整流程

重新生成完整的 QA + 训练数据流程：

```bash
python3 react_agent/retry_failed_anime.py --mode full
```

**功能**:
- 识别缺少问题的动画
- 完整重新生成（问题 + 训练数据）
- 适合需要彻底重试的场景

---

## 高级选项

### 指定数据目录

默认使用 `agent_data`，可以指定其他目录：

```bash
python3 react_agent/retry_failed_anime.py \
  --mode auto \
  --data-dir /path/to/data
```

### 调整重试参数

```bash
python3 react_agent/retry_failed_anime.py \
  --mode auto \
  --max-rounds 7 \
  --max-workers 3
```

**参数说明**:
- `--max-rounds`: QA生成的最大审批轮数（默认: 5）
- `--max-workers`: 训练数据生成的最大并发数（默认: 5）

---

## 工作流程

### 自动模式的工作流程

```
1. 识别缺少问题的动画
   ├─ 扫描所有 _questions.jsonl 文件
   ├─ 检查文件是否存在
   └─ 检查问题数量是否≥5

2. 识别缺少训练数据的动画
   ├─ 扫描所有 _train.json 文件
   ├─ 检查文件是否存在
   └─ 检查训练数据数量是否≥5

3. 读取 batch_summary.json
   ├─ 获取问题生成失败的列表
   └─ 获取训练数据生成失败的列表

4. 合并结果
   ├─ 去重
   └─ 分类：需要重新生成问题 / 训练数据

5. 执行重试
   ├─ 重新生成问题（针对问题失败的动画）
   └─ 重新生成训练数据（针对训练数据失败的动画）

6. 保存结果
   └─ 保存到 retry_result.json
```

---

## 输出示例

### 识别阶段输出

```
====================================================================================================
🔍 识别缺少问题文件的动画
====================================================================================================

  ✅ 莉兹与青鸟 リズと青い鳥
  ❌ 轻音少女 剧场版 映画けいおん！
  ⚠️  向阳素描×蜂窝 (问题数量不足: 3/5)
  ✅ 孤独摇滚！
  ...

📊 统计：
  总动画数: 73
  缺少问题: 5 个
```

### 重试阶段输出

```
====================================================================================================
🔄 重新生成问题
====================================================================================================

[1/5] 处理: 轻音少女 剧场版
----------------------------------------------------------------------------------------------------

====================================================================================================
#                                        QA生成系统: 轻音少女 剧场版
...

✅ 成功: 轻音少女 剧场版

[2/5] 处理: 向阳素描×蜂窝
...

✅ 成功: 向阳素描×蜂窝
```

### 结果文件

**文件位置**: `agent_data/retry_result.json`

```json
{
  "mode": "auto",
  "qa_retry": {
    "total": 5,
    "success": 4,
    "failed": 1,
    "details": [
      {
        "anime": "轻音少女 剧场版 映画けいおん！",
        "status": "success",
        "questions": 5
      },
      {
        "anime": "向阳素描×蜂窝",
        "status": "success",
        "questions": 5
      },
      {
        "anime": "某个失败的动画",
        "status": "failed",
        "error": "GLM Agent连续3轮无法解析输出"
      }
    ]
  },
  "train_retry": {
    "total": 3,
    "success": 3,
    "failed": 0,
    "details": [...]
  }
}
```

---

## 使用场景

### 场景1: 首次运行后自动修复

**首次运行全流程**:
```bash
python3 react_agent/batch_process_all_anime.py
```

**自动重试失败的动画**:
```bash
python3 react_agent/retry_failed_anime.py --mode auto
```

---

### 场景2: 定期检查和维护

**定期运行**，确保所有动画都有完整的数据：

```bash
# 每天运行一次
python3 react_agent/retry_failed_anime.py --mode auto
```

---

### 场景3: 只修复问题生成

如果发现问题生成有问题：

```bash
python3 react_agent/retry_failed_anime.py --mode questions
```

---

### 场景4: 只修复训练数据生成

如果发现训练数据生成有问题：

```bash
python3 react_agent/retry_failed_anime.py --mode train
```

---

## 优势

### 1. 自动化 ⚡

- 无需手动检查每个文件
- 自动识别所有失败
- 一键重试所有失败动画

### 2. 智能识别 🧠

- 从多个维度识别失败（文件检查 + 摘要文件）
- 去重合并
- 避免重复处理

### 3. 详细记录 📊

- 每个动画的重试状态
- 成功/失败原因
- 保存到JSON便于分析

### 4. 灵活模式 🎯

- 自动模式：一键修复所有
- 单独模式：只修复问题或训练数据
- 完整模式：彻底重试

---

## 与主流程的配合

### 完整工作流

```
1. 运行主流程
   python3 react_agent/batch_process_all_anime.py
   ↓
2. 检查失败
   python3 react_agent/retry_failed_anime.py --mode auto
   ↓
3. 查看结果
   cat agent_data/retry_result.json
   ↓
4. 如果仍有失败，再次重试
   python3 react_agent/retry_failed_anime.py --mode auto
```

### 批处理脚本

创建一个批处理脚本 `run_all.sh`:

```bash
#!/bin/bash

# 第1次运行
echo "第1次运行全流程..."
python3 react_agent/batch_process_all_anime.py

# 第1次重试
echo "第1次重试失败动画..."
python3 react_agent/retry_failed_anime.py --mode auto

# 第2次重试（使用更多轮数）
echo "第2次重试失败动画..."
python3 react_agent/retry_failed_anime.py \
  --mode auto \
  --max-rounds 7

echo "完成！"
```

---

## 注意事项

### ⚠️ 不要在主流程运行时使用

- 主流程运行时，不要运行重试脚本
- 可能会导致文件冲突

### ⚠️ 数据备份

- 首次运行前，建议备份 `agent_data` 目录
- 避免数据丢失

### ⚠️ API配额

- 重试会消耗额外的API调用
- 注意API配额限制

---

## 故障排除

### 问题1: 找不到失败的动画

**原因**: 所有动画都已成功

**解决**: 检查 `batch_summary.json` 确认

---

### 问题2: 重试仍然失败

**原因**:
- 萌娘百科数据不足
- API调用失败
- GLM Agent输出不稳定

**解决**:
- 增加 `--max-rounds` 到7或10
- 检查网络连接
- 检查API配置

---

### 问题3: 文件已存在但仍然被识别为失败

**原因**: 文件内容为空或数据不足

**解决**:
- 删除损坏的文件
- 重新运行重试脚本

---

## 总结

### ✅ 功能完备

- 自动识别所有失败类型
- 灵活的重试模式
- 详细的结果记录

### 🎯 使用简单

- 一键修复所有失败
- 自动化程度高
- 无需手动干预

### 📊 可靠性强

- 多维度识别失败
- 去重合并
- 避免重复处理

---

**文档位置**: `/home/tcmofashi/LLaMA-Factory/RETRY_FAILED_ANIME_GUIDE.md`
**脚本位置**: `/home/tcmofashi/LLaMA-Factory/react_agent/retry_failed_anime.py`
