# 全流程QA生成系统 - 使用说明

## 系统概述

该系统用于为动画列表自动生成高质量的问答对和训练数据。

**核心功能：**
- 从萌娘百科搜索动画信息
- 生成5个高质量问题（事实性问题 + 概括性问题 + 分析性问题）
- 为每个问题生成详细答案
- 输出多种格式的训练数据

**技术特点：**
- ✅ 支持多provider负载均衡（GLM官方 + 硅基流动）
- ✅ 智能速率限制（自动按RPM分配请求）
- ✅ 自动错误重试和provider切换
- ✅ 智能跳过已存在的问题文件
- ✅ 实时进度保存

## 目录结构

```
LLaMA-Factory/
├── full_pipeline.py          # 🎯 主流程脚本
├── config.toml               # ⚙️ 配置文件
├── react_agent/              # 📦 核心模块
│   ├── agent.py              # ReAct Agent实现
│   ├── tools.py              # 萌娘百科搜索工具
│   ├── qa_pipeline_v2.py     # QA生成pipeline（含prompt）
│   ├── generate_training_data.py  # 训练数据生成（含prompt）
│   ├── batch_process_all_anime.py # 批量处理脚本
│   ├── rate_limiter.py       # 速率限制器
│   └── load_balancer.py      # 负载均衡器
└── agent_data/               # 📁 数据目录
    ├── anime.txt             # 动画列表
    ├── batch_progress.json   # 进度文件
    └── batch_summary.json    # 最终摘要
```

## 快速开始

### 1. 配置API密钥

编辑 `config.toml` 文件：

```toml
[providers.primary]
name = "glm-official"
model = "glm-4.7"
api_key = "your-glm-api-key"
base_url = "https://open.bigmodel.cn/api/paas/v4/"
rpm = 2

[providers.fallback]
name = "siliconflow"
model = "Pro/zai-org/GLM-4.7"
api_key = "your-siliconflow-api-key"
base_url = "https://api.siliconflow.cn/v1"
rpm = 60
```

### 2. 准备动画列表

编辑 `agent_data/anime.txt`，每行一个动画名称：

```
莉兹与青鸟 リズと青い鳥
轻音少女 剧场版 映画けいおん！
孤独摇滚！ ぼっち・ざ・ろっく！
...
```

### 3. 运行全流程

```bash
# 方式1：运行主流程脚本
python3 full_pipeline.py

# 方式2：使用批量处理脚本
python3 -c "
import sys
sys.path.insert(0, 'react_agent')
from batch_process_all_anime import main
main()
"
```

### 4. 查看结果

- **进度文件**: `agent_data/batch_progress.json`
- **最终摘要**: `agent_data/batch_summary.json`
- **训练数据**: `agent_data/train_fake.json`
- **完整格式**: `agent_data/answer_record/`

## Prompt 说明

### QA生成Prompt（qa_pipeline_v2.py）

生成5个问题，严格遵循以下结构：

1. **问题1-3**：事实性问题
   - 必须整合多个信息点
   - 包含：播出信息 + 制作信息 + 角色/声优 + 剧情

2. **问题4**：内容概括
   - 核心内容和主要情节
   - 主题、故事背景、主要冲突

3. **问题5**：主题和艺术分析
   - 主题思想、艺术表现、创作特色
   - 导演风格、画面、音乐、象征隐喻

### 答案生成Prompt（generate_training_data.py）

- **必须先使用搜索工具**查找萌娘百科信息
- 基于**真实、准确**的数据回答
- 详细、完整、有条理
- 不编造信息

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `rpm` | 每分钟请求数 | primary: 2, fallback: 60 |
| `max_retries` | 最大重试次数 | 5 |
| `retry_base_delay` | 重试基础延迟（秒） | 2 |
| `load_balancing` | 负载均衡策略 | weighted（按RPM权重） |

## 负载均衡策略

当前配置下的请求分配：
- **GLM官方**: 3.2% (2/62 RPM)
- **硅基流动**: 96.8% (60/62 RPM)

系统会自动：
- 优先使用高RPM的provider
- 遇到速率限制自动切换
- 连接失败自动重试

## 断点续传

系统支持断点续传：

```bash
# 中断后重新运行，会自动继续
python3 full_pipeline.py
```

进度保存在 `agent_data/batch_progress.json`，已处理的动画会自动跳过。

## 故障排查

### 错误1：找不到萌娘百科数据集

```
FileNotFoundError: 未找到萌娘百科数据集文件
```

**解决方案**：使用API模式而不是本地数据集

### 错误2：API key无效

```
AuthenticationError: Api key is invalid
```

**解决方案**：检查config.toml中的API key配置

### 错误3：速率限制

```
RateLimitError: API并发限制
```

**解决方案**：系统会自动切换provider，无需手动干预

## 性能优化建议

1. **提高并发**：调整generate_training_data.py中的max_workers
2. **增加RPM**：在config.toml中提高rpm值
3. **使用缓存**：已生成的问题文件会自动跳过

## 输出格式

### train_fake.json（OpenAI chat格式）

```json
[
  {
    "messages": [
      {"role": "user", "content": "问题内容"},
      {"role": "assistant", "content": "答案内容"}
    ]
  }
]
```

### questions.json（问题集合）

```json
[
  {
    "question": "问题内容",
    "answer": "答案内容",
    "type": "factual|summary|analysis"
  }
]
```

## 许可证

MIT License
