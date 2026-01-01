# 全流程脚本运行指南

## 快速开始

### 最简单的运行方式

```bash
# 在项目根目录执行
cd /home/tcmofashi/LLaMA-Factory

# 运行全流程脚本
python3 react_agent/batch_process_all_anime.py
```

就这么简单！脚本会自动处理 `agent_data/anime.txt` 中的所有动画。

---

## 详细运行步骤

### 步骤1：准备动画列表

编辑动画列表文件：

```bash
vim /home/tcmofashi/LLaMA-Factory/agent_data/anime.txt
```

**格式要求：**
- 每行一个动画名称
- 支持中文和日文标题
- 空行会被自动忽略

**示例：**
```
Re：从零开始的异世界生活 Re:ゼロから始める異世界生活
电影 摇曳露营△ 映画 ゆるキャン△
向阳素描×365 ひだまりスケッチ×365
孤独摇滚！ ぼっち・ざ・ろっく！
```

### 步骤2：确认环境变量

确保设置了萌娘百科API环境变量：

```bash
# 方式1：临时设置
export USE_MOEGIRL_API=true

# 方式2：在运行时设置
USE_MOEGIRL_API=true python3 react_agent/batch_process_all_anime.py
```

### 步骤3：运行脚本

```bash
# 基本运行
python3 react_agent/batch_process_all_anime.py

# 或者使用环境变量
USE_MOEGIRL_API=true python3 react_agent/batch_process_all_anime.py
```

---

## 运行过程说明

### 脚本会做什么？

对每个动画执行：

**1️⃣ 检查是否已有问题文件**
```
检查: {anime_name}_questions.jsonl
├─ 存在 → 跳过问题生成 ✅
└─ 不存在 → 生成新问题
```

**2️⃣ 生成问题（如果需要）**
```
调用 GLM-4.7 Agent
├─ 使用萌娘百科搜索工具
├─ 生成 5 个问题
└─ 保存到 {anime_name}_questions.jsonl
```

**3️⃣ 生成训练数据**
```
读取问题文件
├─ 调用 GLM-4.7 生成答案
├─ 调用 DeepSeek-V3.2 清洗答案
│   ├─ 去除工具痕迹
│   ├─ 评估答案质量
│   └─ 必要时触发重新生成
├─ 生成多样化 system prompt
└─ 保存到 train_fake.json
```

### 运行时输出示例

```
####################################################################################################
# 全流程批量处理动画
# 数据源: /home/tcmofashi/LLaMA-Factory/agent_data/anime.txt
# 输出目录: /home/tcmofashi/LLaMA-Factory/agent_data
####################################################################################################

📊 共 3 个动画需要处理

====================================================================================================
🎬 处理动画 [1/3]: Re：从零开始的异世界生活 Re:ゼロから始める異世界生活
====================================================================================================

📝 步骤1: 生成QA问题
----------------------------------------------------------------------------------------------------

🤖 GLM4.7 Agent 正在工作...

✅ QA生成完成: 5个问题

📚 步骤2: 生成训练数据
----------------------------------------------------------------------------------------------------

🤖 正在生成答案...

🔄 正在后处理答案并生成system prompt...
  🔄 DeepSeek要求重新生成: 原答案没有回答问题，请重新生成...

✅ 答案 1/5 清洗完成
✅ 答案 2/5 清洗完成
...

✅ 训练数据生成完成

====================================================================================================
🎬 处理动画 [2/3]: 电影 摇曳露营△ 映画 ゆるキャン△
====================================================================================================
...
```

---

## 输出文件说明

### 问题文件

**位置：** `agent_data/{anime_name}_questions.jsonl`

**格式：** JSONL（每行一个JSON）

**示例：**
```json
{"question": "问题1", "answer": "答案1", "type": "factual"}
{"question": "问题2", "answer": "答案2", "type": "factual"}
{"question": "问题3", "answer": "答案3", "type": "factual"}
{"question": "问题4", "answer": "答案4", "type": "summary"}
{"question": "问题5", "answer": "答案5", "type": "analysis"}
```

### 训练数据文件

**位置：** `agent_data/train_fake.json`

**格式：** OpenAI Chat格式

**示例：**
```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "作为一个动画爱好者，我熟悉制作团队、声优和故事背景，随时分享动漫细节。"
      },
      {
        "role": "user",
        "content": "《向阳素描×365》的核心内容和主要情节是什么？"
      },
      {
        "role": "assistant",
        "content": "《向阳素描×365》是《向阳素描》系列动画的第二季..."
      }
    ]
  }
]
```

### 进度文件

**位置：** `agent_data/batch_progress.json`

**内容：** 实时更新的处理进度

**示例：**
```json
{
  "total": 3,
  "qa_success": 2,
  "train_success": 2,
  "total_questions": 10,
  "results": [...]
}
```

### 最终摘要

**位置：** `agent_data/batch_summary.json`

**内容：** 最终处理结果统计

---

## 实际运行示例

### 示例1：处理全新的3个动画

```bash
# 当前 anime.txt 内容：
# Re：从零开始的异世界生活
# 电影 摇曳露营△
# 向阳素描×365

# 运行脚本
python3 react_agent/batch_process_all_anime.py
```

**预期结果：**
- 生成 3 个问题文件（每个动画 5 个问题）
- 生成 1 个训练数据文件（15 个样本）
- 总共生成 15 个问答对

### 示例2：处理已有问题的动画

```bash
# 假设孤独摇滚！已有问题文件

# 运行脚本
python3 react_agent/batch_process_all_anime.py
```

**预期结果：**
```
📝 步骤1: 发现已存在的问题文件
ℹ️  文件路径: .../孤独摇滚！_questions.jsonl
✅ 跳过问题生成，使用已有问题: 5个问题

📚 步骤2: 生成训练数据
...
```

### 示例3：中断后继续

```bash
# 如果脚本中断（比如网络问题）
# 再次运行相同的命令即可
python3 react_agent/batch_process_all_anime.py
```

**预期结果：**
- 已完成的动画会被跳过
- 继续处理未完成的动画

---

## 高级用法

### 只生成问题（不生成训练数据）

直接调用问题生成脚本：

```bash
python3 -c "
import sys
sys.path.insert(0, 'react_agent')
from qa_pipeline_v2 import generate_questions_for_anime_v2

result = generate_questions_for_anime_v2(
    anime_name='孤独摇滚！',
    output_dir='agent_data'
)
"
```

### 只生成训练数据（使用已有问题）

直接调用训练数据生成脚本：

```bash
python3 -c "
import sys
sys.path.insert(0, 'react_agent')
from generate_training_data import generate_training_data_from_questions

generate_training_data_from_questions(
    questions_file='agent_data/孤独摇滚！_questions.jsonl',
    output_dir='agent_data'
)
"
```

### 重新生成某个动画的问题

```bash
# 删除旧的问题文件
rm agent_data/孤独摇滚！_questions.jsonl

# 重新运行全流程
python3 react_agent/batch_process_all_anime.py
```

---

## 故障排查

### 问题1：萌娘百科API不可用

**错误信息：**
```
⚠️  Agent检测到萌娘百科信息不足，跳过该动画
```

**解决方案：**
```bash
# 确认萌娘百科API服务是否启动
# 如果需要启动API服务，请参考相关文档
```

### 问题2：DeepSeek调用失败

**错误信息：**
```
⚠️  DeepSeek尝试 1 失败: ...
```

**解决方案：**
- 脚本会自动使用fallback方法（正则表达式清理）
- 检查 `config.toml` 中的DeepSeek配置
- 确认API密钥有效

### 问题3：进度文件损坏

**解决方案：**
```bash
# 删除进度文件，重新开始
rm agent_data/batch_progress.json
rm agent_data/batch_summary.json
```

---

## 配置文件说明

### config.toml

确保配置了必要的provider：

```toml
[providers.glm_official]
name = "glm"
model = "glm-4-plus"
api_key = "..."
base_url = "https://open.bigmodel.cn/api/paas/v4/"
rpm = 2
timeout = 300

[providers.glm_siliconflow]
name = "siliconflow"
model = "Pro/GLM-4-Plus-AWQ"
api_key = "sk-..."
base_url = "https://api.siliconflow.cn/v1"
rpm = 60
timeout = 300

[providers.deepseek]
name = "siliconflow"
model = "Pro/deepseek-ai/DeepSeek-V3.2"
api_key = "sk-..."
base_url = "https://api.siliconflow.cn/v1"
rpm = 60
timeout = 300
```

---

## 性能参考

### 预计时间消耗

每个动画的处理时间：

- **步骤1（生成问题）**：约 2-5 分钟
  - GLM-4.7 搜索 + 生成 5 个问题

- **步骤2（生成训练数据）**：约 5-10 分钟
  - GLM-4.7 生成 5 个答案
  - DeepSeek 清洗 5 个答案
  - 生成 5 个 system prompt

- **总计**：每个动画约 7-15 分钟

### 批量处理

- 3 个动画：约 21-45 分钟
- 10 个动画：约 70-150 分钟

---

## 总结

### 快速命令

```bash
# 1. 编辑动画列表
vim agent_data/anime.txt

# 2. 运行全流程
python3 react_agent/batch_process_all_anime.py

# 3. 查看结果
cat agent_data/train_fake.json
```

### 核心要点

✅ **简单**：一条命令完成所有操作
✅ **智能**：自动跳过已生成的动画
✅ **可靠**：支持断点续传
✅ **灵活**：可以单独运行各个步骤

---

**最后更新**: 2026-01-01
**文档位置**: `/home/tcmofashi/LLaMA-Factory/PIPELINE_RUN_GUIDE.md`
