[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

# 🎙️ PersonaPlex

[![Docker](https://img.shields.io/badge/Docker-neosun%2Fpersonaplex-blue?logo=docker)](https://hub.docker.com/r/neosun/personaplex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Paper](https://img.shields.io/badge/📄-論文-blue)](https://arxiv.org/abs/2602.06053)
[![Model](https://img.shields.io/badge/🤗-模型-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Demo](https://img.shields.io/badge/🎮-演示-green)](https://research.nvidia.com/labs/adlr/personaplex/)

**即時全雙工對話式AI，支援語音和角色控制**

PersonaPlex 是一個語音到語音的對話模型，透過文字角色提示和音訊語音條件實現人格控制。它能產生自然、低延遲的語音互動，並保持一致的人格特徵。

![架構圖](assets/architecture_diagram.png)

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 🎯 **全雙工** | 即時雙向對話 |
| 🎭 **人格控制** | 文字提示定義AI性格 |
| 🗣️ **語音選擇** | 18種預訓練語音選項 |
| 🌐 **多語言介面** | 英文、簡體中文、繁體中文、日語 |
| 🐳 **一體化Docker** | 單容器部署 |
| 📡 **REST API** | OpenAPI/Swagger文件 |
| 🔌 **MCP支援** | 模型上下文協議整合 |
| 🖥️ **GPU管理** | 自動選擇和記憶體卸載 |

## 🚀 快速開始

### Docker（推薦）

```bash
# 拉取並執行
docker run -d --gpus all \
  -p 8998:8998 \
  -e HF_TOKEN=your_token \
  --name personaplex \
  neosun/personaplex:latest

# 存取Web介面
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
# 設定token並啟動
export HF_TOKEN=your_huggingface_token
docker-compose up -d
```

### 一鍵啟動

```bash
# 複製儲存庫
git clone https://github.com/neosun100/personaplex.git
cd personaplex

# 設定HF token
export HF_TOKEN=your_huggingface_token

# 啟動（自動選擇顯存佔用最少的GPU）
./start.sh
```

## ⚙️ 配置說明

| 變數 | 預設值 | 描述 |
|------|--------|------|
| `HF_TOKEN` | - | **必需**：HuggingFace令牌 |
| `PORT` | `8998` | Web介面連接埠 |
| `DEVICE` | `cuda` | 裝置：cuda, cpu |
| `CPU_OFFLOAD` | `false` | GPU記憶體不足時卸載到CPU |
| `GPU_IDLE_TIMEOUT` | `300` | 閒置後自動卸載（秒） |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 使用的GPU ID |

## 🗣️ 語音選項

| 類別 | ID | 描述 |
|------|-----|------|
| 自然女聲 | NATF0-3 | 自然、對話式 |
| 自然男聲 | NATM0-3 | 自然、對話式 |
| 多樣女聲 | VARF0-4 | 多樣風格 |
| 多樣男聲 | VARM0-4 | 多樣風格 |

## 📡 API參考

### 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web介面 |
| `/health` | GET | 健康檢查 |
| `/docs` | GET | Swagger API文件 |
| `/api/gpu/status` | GET | GPU狀態 |
| `/api/gpu/offload` | POST | 釋放GPU記憶體 |
| `/api/voices` | GET | 語音列表 |
| `/api/offline` | POST | 離線推理 |
| `/api/chat` | WebSocket | 即時對話 |

## 🔌 MCP整合

詳見 [MCP_GUIDE.md](MCP_GUIDE.md) 了解模型上下文協議整合。

## 🛠️ 技術棧

- **模型**: [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) 基於Moshi
- **後端**: FastAPI + Uvicorn
- **前端**: Jinja2 + 原生JS
- **容器**: NVIDIA CUDA 12.4 + cuDNN
- **協議**: WebSocket + REST + MCP

## 📋 更新日誌

### v1.3.0 (2026-02-17)
- 🗑️ 修復GPU卸載：透過 `gc.collect()` + `torch.cuda.ipc_collect()` 真正釋放顯存（18GB → 0.6GB）
- ⏱️ 閒置自動卸載GPU（`GPU_IDLE_TIMEOUT`，預設300秒）
- 📊 GPU狀態新增活躍連線數和閒置計時器
- 🔄 卸載後下次連線自動重新載入模型

### v1.2.0 (2026-02-16)
- 🔧 修復CUDA OOM：推理迴圈添加 `torch.no_grad()`（支援無限時長對話）
- 🔒 修復GPU鎖：逾時返回「伺服器忙碌」而非無限等待
- 🎤 完整的瀏覽器語音對話（透過WebSocket進行Opus編解碼）
- 💬 串流AI文字顯示，按時間自動分段
- 🏓 模型載入期間WebSocket保活（修復Cloudflare 502）
- 🐛 修復 `tensor.detach().numpy()` RuntimeError

### v1.0.0 (2026-02-16)
- 🐳 一體化Docker部署
- 🌐 多語言Web介面（英/中/繁/日）
- 📡 REST API + Swagger文件
- 🔌 MCP伺服器整合
- 🖥️ 自動GPU選擇
- 🗑️ GPU記憶體卸載

## 📄 授權條款

- 程式碼：MIT授權條款
- 模型權重：[NVIDIA開放模型授權條款](https://huggingface.co/nvidia/personaplex-7b-v1)

---

## ⭐ Star歷史

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/personaplex&type=Date)](https://star-history.com/#neosun100/personaplex)

## 📱 關注我們

![公眾號](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
