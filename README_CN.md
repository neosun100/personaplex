[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

# 🎙️ PersonaPlex

[![Docker](https://img.shields.io/badge/Docker-neosun%2Fpersonaplex-blue?logo=docker)](https://hub.docker.com/r/neosun/personaplex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Paper](https://img.shields.io/badge/📄-论文-blue)](https://arxiv.org/abs/2602.06053)
[![Model](https://img.shields.io/badge/🤗-模型-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Demo](https://img.shields.io/badge/🎮-演示-green)](https://research.nvidia.com/labs/adlr/personaplex/)

**实时全双工对话式AI，支持语音和角色控制**

PersonaPlex 是一个语音到语音的对话模型，通过文本角色提示和音频语音条件实现人格控制。它能产生自然、低延迟的语音交互，并保持一致的人格特征。

![架构图](assets/architecture_diagram.png)

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 🎯 **全双工** | 实时双向对话 |
| 🎭 **人格控制** | 文本提示定义AI性格 |
| 🗣️ **语音选择** | 18种预训练语音选项 |
| 🌐 **多语言界面** | 英文、简体中文、繁体中文、日语 |
| 🐳 **一体化Docker** | 单容器部署 |
| 📡 **REST API** | OpenAPI/Swagger文档 |
| 🔌 **MCP支持** | 模型上下文协议集成 |
| 🖥️ **GPU管理** | 自动选择和内存卸载 |

## 🚀 快速开始

### Docker（推荐）

```bash
# 拉取并运行
docker run -d --gpus all \
  -p 8998:8998 \
  -e HF_TOKEN=your_token \
  --name personaplex \
  neosun/personaplex:latest

# 访问Web界面
open http://localhost:8998
```

### Docker Compose

```yaml
version: '3.8'
services:
  personaplex:
    image: neosun/personaplex:latest
    container_name: personaplex
    ports:
      - "8998:8998"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - /tmp/personaplex:/tmp/personaplex
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

```bash
# 设置token并启动
export HF_TOKEN=your_huggingface_token
docker-compose up -d
```

### 一键启动

```bash
# 克隆仓库
git clone https://github.com/neosun100/personaplex.git
cd personaplex

# 设置HF token
export HF_TOKEN=your_huggingface_token

# 启动（自动选择显存占用最少的GPU）
./start.sh
```

## ⚙️ 配置说明

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `HF_TOKEN` | - | **必需**：HuggingFace令牌 |
| `PORT` | `8998` | Web界面端口 |
| `DEVICE` | `cuda` | 设备：cuda, cpu |
| `CPU_OFFLOAD` | `false` | GPU内存不足时卸载到CPU |
| `GPU_IDLE_TIMEOUT` | `300` | 空闲后自动卸载（秒） |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 使用的GPU ID |

### GPU选择

```bash
# 使用指定GPU
export NVIDIA_VISIBLE_DEVICES=2
docker-compose up -d

# 或在docker-compose.yml中配置
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['2']
          capabilities: [gpu]
```

## 🗣️ 语音选项

| 类别 | ID | 描述 |
|------|-----|------|
| 自然女声 | NATF0-3 | 自然、对话式 |
| 自然男声 | NATM0-3 | 自然、对话式 |
| 多样女声 | VARF0-4 | 多样风格 |
| 多样男声 | VARM0-4 | 多样风格 |

## 📝 提示词示例

### 助手（默认）
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

### 客服
```
You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity.
```

### 闲聊
```
You enjoy having a good conversation.
```

## 📡 API参考

### 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web界面 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger API文档 |
| `/api/gpu/status` | GET | GPU状态 |
| `/api/gpu/offload` | POST | 释放GPU内存 |
| `/api/voices` | GET | 语音列表 |
| `/api/offline` | POST | 离线推理 |
| `/api/chat` | WebSocket | 实时对话 |

### 离线推理

```bash
curl -X POST http://localhost:8998/api/offline \
  -F "file=@input.wav" \
  -F "voice_prompt=NATF2.pt" \
  -F "text_prompt=You are a helpful assistant." \
  -o output.wav
```

## 🔌 MCP集成

详见 [MCP_GUIDE.md](MCP_GUIDE.md) 了解模型上下文协议集成。

```json
{
  "mcpServers": {
    "personaplex": {
      "command": "docker",
      "args": ["exec", "-i", "personaplex", "python", "-m", "app.mcp_server"]
    }
  }
}
```

## 🏗️ 项目结构

```
personaplex/
├── app/
│   ├── server.py          # FastAPI服务器
│   ├── mcp_server.py      # MCP服务器
│   └── templates/         # Web界面
├── moshi/                  # 核心模型包
├── client/                 # 原始React客户端
├── assets/                 # 测试文件
├── Dockerfile             # 一体化镜像
├── docker-compose.yml     # Compose配置
├── start.sh               # 一键启动脚本
└── MCP_GUIDE.md           # MCP文档
```

## 🛠️ 技术栈

- **模型**: [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) 基于Moshi
- **后端**: FastAPI + Uvicorn
- **前端**: Jinja2 + 原生JS
- **容器**: NVIDIA CUDA 12.4 + cuDNN
- **协议**: WebSocket + REST + MCP

## 📋 更新日志

### v1.2.0 (2026-02-16)
- 🔧 修复CUDA OOM：推理循环添加 `torch.no_grad()`（支持无限时长对话）
- 🔒 修复GPU锁：超时返回"服务器忙"而非无限等待
- 🎤 完整的浏览器语音对话（通过WebSocket进行Opus编解码）
- 💬 流式AI文字显示，按时间自动分段
- 🏓 模型加载期间WebSocket保活（修复Cloudflare 502）
- 🐛 修复 `tensor.detach().numpy()` RuntimeError

### v1.0.0 (2026-02-16)
- 🐳 一体化Docker部署
- 🌐 多语言Web界面（英/中/繁/日）
- 📡 REST API + Swagger文档
- 🔌 MCP服务器集成
- 🖥️ 自动GPU选择
- 🗑️ GPU内存卸载

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing`)
5. 创建Pull Request

## 📄 许可证

- 代码：MIT许可证
- 模型权重：[NVIDIA开放模型许可证](https://huggingface.co/nvidia/personaplex-7b-v1)

## 🙏 致谢

- [NVIDIA PersonaPlex](https://arxiv.org/abs/2602.06053) - 原始研究
- [Kyutai Moshi](https://arxiv.org/abs/2410.00037) - 基础架构
- [Helium LLM](https://kyutai.org/blog/2025-04-30-helium) - 语言模型骨干

---

## ⭐ Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/personaplex&type=Date)](https://star-history.com/#neosun100/personaplex)

## 📱 关注我们

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
