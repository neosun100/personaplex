[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

# 🎙️ PersonaPlex

[![Docker](https://img.shields.io/badge/Docker-neosun%2Fpersonaplex-blue?logo=docker)](https://hub.docker.com/r/neosun/personaplex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2602.06053)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)

**Real-time Full-Duplex Conversational AI with Voice and Role Control**

PersonaPlex is a speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. It produces natural, low-latency spoken interactions with consistent personas.

![Screenshot](assets/architecture_diagram.png)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Full-Duplex** | Real-time bidirectional conversation |
| 🎭 **Persona Control** | Text prompts define AI personality |
| 🗣️ **Voice Selection** | 18 pre-trained voice options |
| 🌐 **Multi-language UI** | English, 中文, 繁體, 日本語 |
| 🐳 **All-in-One Docker** | Single container deployment |
| 📡 **REST API** | OpenAPI/Swagger documented |
| 🔌 **MCP Support** | Model Context Protocol integration |
| 🖥️ **GPU Management** | Auto-select & memory offload |

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Pull and run
docker run -d --gpus all \
  -p 8998:8998 \
  -e HF_TOKEN=your_token \
  --name personaplex \
  neosun/personaplex:latest

# Access Web UI
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
# Set token and start
export HF_TOKEN=your_huggingface_token
docker-compose up -d
```

### One-Click Start

```bash
# Clone repository
git clone https://github.com/neosun100/personaplex.git
cd personaplex

# Set HF token
export HF_TOKEN=your_huggingface_token

# Start (auto-selects GPU with lowest memory usage)
./start.sh
```

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | **Required**: HuggingFace token |
| `PORT` | `8998` | Web UI port |
| `DEVICE` | `cuda` | Device: cuda, cpu |
| `CPU_OFFLOAD` | `false` | Offload to CPU if GPU OOM |
| `GPU_IDLE_TIMEOUT` | `300` | Auto-unload after idle (seconds) |
| `NVIDIA_VISIBLE_DEVICES` | `0` | GPU ID to use |

### GPU Selection

```bash
# Use specific GPU
export NVIDIA_VISIBLE_DEVICES=2
docker-compose up -d

# Or in docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['2']
          capabilities: [gpu]
```

## 🗣️ Voice Options

| Category | IDs | Description |
|----------|-----|-------------|
| Natural Female | NATF0-3 | Natural, conversational |
| Natural Male | NATM0-3 | Natural, conversational |
| Variety Female | VARF0-4 | Diverse styles |
| Variety Male | VARM0-4 | Diverse styles |

## 📝 Prompt Examples

### Assistant (Default)
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

### Customer Service
```
You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity.
```

### Casual Conversation
```
You enjoy having a good conversation.
```

## 📡 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger API docs |
| `/api/gpu/status` | GET | GPU status |
| `/api/gpu/offload` | POST | Release GPU memory |
| `/api/voices` | GET | List voices |
| `/api/offline` | POST | Offline inference |
| `/api/chat` | WebSocket | Real-time conversation |

### Offline Inference

```bash
curl -X POST http://localhost:8998/api/offline \
  -F "file=@input.wav" \
  -F "voice_prompt=NATF2.pt" \
  -F "text_prompt=You are a helpful assistant." \
  -o output.wav
```

## 🔌 MCP Integration

See [MCP_GUIDE.md](MCP_GUIDE.md) for Model Context Protocol integration.

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

## 🏗️ Project Structure

```
personaplex/
├── app/
│   ├── server.py          # FastAPI server
│   ├── mcp_server.py      # MCP server
│   └── templates/         # Web UI
├── moshi/                  # Core model package
├── client/                 # Original React client
├── assets/                 # Test files
├── Dockerfile             # All-in-One image
├── docker-compose.yml     # Compose config
├── start.sh               # One-click start
└── MCP_GUIDE.md           # MCP documentation
```

## 🛠️ Tech Stack

- **Model**: [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) based on Moshi
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Jinja2 + Vanilla JS
- **Container**: NVIDIA CUDA 12.4 + cuDNN
- **Protocol**: WebSocket + REST + MCP

## 📋 Changelog

### v1.0.0 (2026-02-16)
- 🐳 All-in-One Docker deployment
- 🌐 Multi-language Web UI (EN/中文/繁體/日本語)
- 📡 REST API with Swagger docs
- 🔌 MCP server integration
- 🖥️ Auto GPU selection
- 🗑️ GPU memory offload

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

- Code: MIT License
- Model Weights: [NVIDIA Open Model License](https://huggingface.co/nvidia/personaplex-7b-v1)

## 🙏 Acknowledgments

- [NVIDIA PersonaPlex](https://arxiv.org/abs/2602.06053) - Original research
- [Kyutai Moshi](https://arxiv.org/abs/2410.00037) - Base architecture
- [Helium LLM](https://kyutai.org/blog/2025-04-30-helium) - Language model backbone

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/personaplex&type=Date)](https://star-history.com/#neosun100/personaplex)

## 📱 Follow Us

![WeChat](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
