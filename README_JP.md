[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

# 🎙️ PersonaPlex

[![Docker](https://img.shields.io/badge/Docker-neosun%2Fpersonaplex-blue?logo=docker)](https://hub.docker.com/r/neosun/personaplex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Paper](https://img.shields.io/badge/📄-論文-blue)](https://arxiv.org/abs/2602.06053)
[![Model](https://img.shields.io/badge/🤗-モデル-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Demo](https://img.shields.io/badge/🎮-デモ-green)](https://research.nvidia.com/labs/adlr/personaplex/)

**リアルタイム全二重会話AI - 音声とロール制御対応**

PersonaPlexは、テキストベースのロールプロンプトとオーディオベースの音声条件付けによってペルソナ制御を可能にする音声対音声の会話モデルです。一貫したペルソナで自然で低遅延の音声インタラクションを実現します。

![アーキテクチャ図](assets/architecture_diagram.png)

---

## ✨ 機能

| 機能 | 説明 |
|------|------|
| 🎯 **全二重** | リアルタイム双方向会話 |
| 🎭 **ペルソナ制御** | テキストプロンプトでAIの性格を定義 |
| 🗣️ **音声選択** | 18種類の事前学習済み音声オプション |
| 🌐 **多言語UI** | 英語、中国語（簡体/繁体）、日本語 |
| 🐳 **オールインワンDocker** | 単一コンテナデプロイ |
| 📡 **REST API** | OpenAPI/Swaggerドキュメント |
| 🔌 **MCPサポート** | モデルコンテキストプロトコル統合 |
| 🖥️ **GPU管理** | 自動選択とメモリオフロード |

## 🚀 クイックスタート

### Docker（推奨）

```bash
# プルして実行
docker run -d --gpus all \
  -p 8998:8998 \
  -e HF_TOKEN=your_token \
  --name personaplex \
  neosun/personaplex:latest

# Web UIにアクセス
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
# トークンを設定して起動
export HF_TOKEN=your_huggingface_token
docker-compose up -d
```

### ワンクリック起動

```bash
# リポジトリをクローン
git clone https://github.com/neosun100/personaplex.git
cd personaplex

# HFトークンを設定
export HF_TOKEN=your_huggingface_token

# 起動（メモリ使用量が最も少ないGPUを自動選択）
./start.sh
```

## ⚙️ 設定

| 変数 | デフォルト | 説明 |
|------|------------|------|
| `HF_TOKEN` | - | **必須**: HuggingFaceトークン |
| `PORT` | `8998` | Web UIポート |
| `DEVICE` | `cuda` | デバイス: cuda, cpu |
| `CPU_OFFLOAD` | `false` | GPUメモリ不足時にCPUにオフロード |
| `GPU_IDLE_TIMEOUT` | `300` | アイドル後の自動アンロード（秒） |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 使用するGPU ID |

## 🗣️ 音声オプション

| カテゴリ | ID | 説明 |
|----------|-----|------|
| ナチュラル女性 | NATF0-3 | 自然な会話調 |
| ナチュラル男性 | NATM0-3 | 自然な会話調 |
| バラエティ女性 | VARF0-4 | 多様なスタイル |
| バラエティ男性 | VARM0-4 | 多様なスタイル |

## 📡 APIリファレンス

### エンドポイント

| エンドポイント | メソッド | 説明 |
|----------------|----------|------|
| `/` | GET | Web UI |
| `/health` | GET | ヘルスチェック |
| `/docs` | GET | Swagger APIドキュメント |
| `/api/gpu/status` | GET | GPU状態 |
| `/api/gpu/offload` | POST | GPUメモリ解放 |
| `/api/voices` | GET | 音声リスト |
| `/api/offline` | POST | オフライン推論 |
| `/api/chat` | WebSocket | リアルタイム会話 |

## 🔌 MCP統合

モデルコンテキストプロトコル統合については [MCP_GUIDE.md](MCP_GUIDE.md) を参照してください。

## 🛠️ 技術スタック

- **モデル**: [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) Moshiベース
- **バックエンド**: FastAPI + Uvicorn
- **フロントエンド**: Jinja2 + バニラJS
- **コンテナ**: NVIDIA CUDA 12.4 + cuDNN
- **プロトコル**: WebSocket + REST + MCP

## 📋 変更履歴

### v1.2.0 (2026-02-16)
- 🔧 CUDA OOM修正：推論ループに `torch.no_grad()` 追加（無制限の会話が可能に）
- 🔒 GPUロック修正：無限待機の代わりに「サーバービジー」タイムアウト
- 🎤 ブラウザベースの完全な音声チャット（WebSocket経由のOpusエンコード/デコード）
- 💬 ストリーミングAIテキスト表示、時間ベースのメッセージバブル
- 🏓 モデルロード中のWebSocketキープアライブ（Cloudflare 502修正）
- 🐛 `tensor.detach().numpy()` RuntimeError修正

### v1.0.0 (2026-02-16)
- 🐳 オールインワンDockerデプロイ
- 🌐 多言語Web UI（英/中/繁/日）
- 📡 REST API + Swaggerドキュメント
- 🔌 MCPサーバー統合
- 🖥️ 自動GPU選択
- 🗑️ GPUメモリオフロード

## 📄 ライセンス

- コード: MITライセンス
- モデル重み: [NVIDIAオープンモデルライセンス](https://huggingface.co/nvidia/personaplex-7b-v1)

---

## ⭐ Star履歴

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/personaplex&type=Date)](https://star-history.com/#neosun100/personaplex)

## 📱 フォローする

![WeChat](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)
