"""
PersonaPlex All-in-One Server
Provides: Web UI + REST API + WebSocket + MCP support
"""
import os
import sys
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# GPU Manager for shared resource management
class GPUManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.mimi = None
        self.other_mimi = None
        self.text_tokenizer = None
        self.lm_gen = None
        self.device = None
        self.lock = asyncio.Lock()
        self.last_used = None
        self.idle_timeout = int(os.getenv("GPU_IDLE_TIMEOUT", "300"))
    
    def get_status(self) -> dict:
        status = {
            "model_loaded": self.model is not None,
            "device": str(self.device) if self.device else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }
        if torch.cuda.is_available():
            try:
                gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
                status["gpu"] = {
                    "id": gpu_id,
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory // (1024**2),
                    "memory_allocated": torch.cuda.memory_allocated(0) // (1024**2),
                    "memory_reserved": torch.cuda.memory_reserved(0) // (1024**2),
                }
            except Exception as e:
                status["gpu_error"] = str(e)
        return status
    
    def offload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.mimi is not None:
            del self.mimi
            self.mimi = None
        if self.other_mimi is not None:
            del self.other_mimi
            self.other_mimi = None
        if self.lm_gen is not None:
            del self.lm_gen
            self.lm_gen = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "offloaded"}

gpu_manager = GPUManager()

# FastAPI App
app = FastAPI(
    title="PersonaPlex API",
    description="Real-time Full-Duplex Conversational AI with Voice and Role Control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Translations
TRANSLATIONS = {
    "en": {
        "title": "PersonaPlex",
        "subtitle": "Real-time Full-Duplex Conversational AI",
        "text_prompt": "Text Prompt",
        "voice": "Voice",
        "connect": "Connect",
        "disconnect": "Disconnect",
        "settings": "Settings",
        "gpu_status": "GPU Status",
        "offload_gpu": "Release GPU",
        "examples": "Examples",
        "assistant": "Assistant",
        "service": "Customer Service",
        "casual": "Casual Chat",
        "natural_female": "Natural Female",
        "natural_male": "Natural Male",
        "variety_female": "Variety Female",
        "variety_male": "Variety Male",
        "api_docs": "API Docs",
        "health": "Health",
        "loading": "Loading...",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "error": "Error",
    },
    "zh-CN": {
        "title": "PersonaPlex",
        "subtitle": "实时全双工对话式AI",
        "text_prompt": "文本提示词",
        "voice": "语音",
        "connect": "连接",
        "disconnect": "断开",
        "settings": "设置",
        "gpu_status": "GPU状态",
        "offload_gpu": "释放显存",
        "examples": "示例",
        "assistant": "助手",
        "service": "客服",
        "casual": "闲聊",
        "natural_female": "自然女声",
        "natural_male": "自然男声",
        "variety_female": "多样女声",
        "variety_male": "多样男声",
        "api_docs": "API文档",
        "health": "健康检查",
        "loading": "加载中...",
        "connected": "已连接",
        "disconnected": "已断开",
        "error": "错误",
    },
    "zh-TW": {
        "title": "PersonaPlex",
        "subtitle": "即時全雙工對話式AI",
        "text_prompt": "文字提示詞",
        "voice": "語音",
        "connect": "連接",
        "disconnect": "斷開",
        "settings": "設定",
        "gpu_status": "GPU狀態",
        "offload_gpu": "釋放顯存",
        "examples": "範例",
        "assistant": "助手",
        "service": "客服",
        "casual": "閒聊",
        "natural_female": "自然女聲",
        "natural_male": "自然男聲",
        "variety_female": "多樣女聲",
        "variety_male": "多樣男聲",
        "api_docs": "API文檔",
        "health": "健康檢查",
        "loading": "載入中...",
        "connected": "已連接",
        "disconnected": "已斷開",
        "error": "錯誤",
    },
    "ja": {
        "title": "PersonaPlex",
        "subtitle": "リアルタイム全二重会話AI",
        "text_prompt": "テキストプロンプト",
        "voice": "音声",
        "connect": "接続",
        "disconnect": "切断",
        "settings": "設定",
        "gpu_status": "GPU状態",
        "offload_gpu": "GPU解放",
        "examples": "例",
        "assistant": "アシスタント",
        "service": "カスタマーサービス",
        "casual": "カジュアル",
        "natural_female": "ナチュラル女性",
        "natural_male": "ナチュラル男性",
        "variety_female": "バラエティ女性",
        "variety_male": "バラエティ男性",
        "api_docs": "APIドキュメント",
        "health": "ヘルスチェック",
        "loading": "読み込み中...",
        "connected": "接続済み",
        "disconnected": "切断済み",
        "error": "エラー",
    }
}

VOICE_OPTIONS = [
    {"id": "NATF0.pt", "name": "Natural Female 0", "category": "natural_female"},
    {"id": "NATF1.pt", "name": "Natural Female 1", "category": "natural_female"},
    {"id": "NATF2.pt", "name": "Natural Female 2", "category": "natural_female"},
    {"id": "NATF3.pt", "name": "Natural Female 3", "category": "natural_female"},
    {"id": "NATM0.pt", "name": "Natural Male 0", "category": "natural_male"},
    {"id": "NATM1.pt", "name": "Natural Male 1", "category": "natural_male"},
    {"id": "NATM2.pt", "name": "Natural Male 2", "category": "natural_male"},
    {"id": "NATM3.pt", "name": "Natural Male 3", "category": "natural_male"},
    {"id": "VARF0.pt", "name": "Variety Female 0", "category": "variety_female"},
    {"id": "VARF1.pt", "name": "Variety Female 1", "category": "variety_female"},
    {"id": "VARF2.pt", "name": "Variety Female 2", "category": "variety_female"},
    {"id": "VARF3.pt", "name": "Variety Female 3", "category": "variety_female"},
    {"id": "VARF4.pt", "name": "Variety Female 4", "category": "variety_female"},
    {"id": "VARM0.pt", "name": "Variety Male 0", "category": "variety_male"},
    {"id": "VARM1.pt", "name": "Variety Male 1", "category": "variety_male"},
    {"id": "VARM2.pt", "name": "Variety Male 2", "category": "variety_male"},
    {"id": "VARM3.pt", "name": "Variety Male 3", "category": "variety_male"},
    {"id": "VARM4.pt", "name": "Variety Male 4", "category": "variety_male"},
]

PROMPT_EXAMPLES = [
    {
        "id": "assistant",
        "name": "Assistant",
        "text": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
    },
    {
        "id": "service",
        "name": "Customer Service",
        "text": "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity."
    },
    {
        "id": "casual",
        "name": "Casual Chat",
        "text": "You enjoy having a good conversation."
    },
    {
        "id": "astronaut",
        "name": "Astronaut",
        "text": "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex."
    }
]

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, lang: str = "en"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return templates.TemplateResponse("index.html", {
        "request": request,
        "t": t,
        "lang": lang,
        "voices": VOICE_OPTIONS,
        "examples": PROMPT_EXAMPLES,
        "languages": ["en", "zh-CN", "zh-TW", "ja"]
    })

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": gpu_manager.model is not None
    }

@app.get("/api/gpu/status")
async def gpu_status():
    return gpu_manager.get_status()

@app.post("/api/gpu/offload")
async def gpu_offload():
    return gpu_manager.offload()

@app.get("/api/voices")
async def list_voices():
    return {"voices": VOICE_OPTIONS}

@app.get("/api/prompts")
async def list_prompts():
    return {"prompts": PROMPT_EXAMPLES}

@app.get("/api/translations/{lang}")
async def get_translations(lang: str):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"])

@app.post("/api/offline")
async def offline_inference(
    file: UploadFile = File(...),
    voice_prompt: str = Form("NATF2.pt"),
    text_prompt: str = Form("You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."),
    seed: int = Form(42424242),
):
    """Offline inference endpoint - process audio file and return response"""
    try:
        # Save uploaded file
        input_path = Path(tempfile.mktemp(suffix=".wav"))
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        output_text = Path(tempfile.mktemp(suffix=".json"))
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run offline inference
        cmd = [
            sys.executable, "-m", "moshi.offline",
            "--voice-prompt", voice_prompt,
            "--text-prompt", text_prompt,
            "--input-wav", str(input_path),
            "--output-wav", str(output_path),
            "--output-text", str(output_text),
            "--seed", str(seed),
        ]
        
        if os.getenv("CPU_OFFLOAD", "false").lower() == "true":
            cmd.append("--cpu-offload")
        
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        
        # Read results
        with open(output_text) as f:
            text_output = json.load(f)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="output.wav",
            headers={"X-Text-Output": json.dumps(text_output)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for p in [input_path, output_text]:
            if p.exists():
                p.unlink()

# Main entry point
def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8998"))
    
    # Check for SSL
    ssl_dir = os.getenv("SSL_DIR")
    ssl_keyfile = None
    ssl_certfile = None
    
    if ssl_dir and Path(ssl_dir).exists():
        key_path = Path(ssl_dir) / "key.pem"
        cert_path = Path(ssl_dir) / "cert.pem"
        if key_path.exists() and cert_path.exists():
            ssl_keyfile = str(key_path)
            ssl_certfile = str(cert_path)
    
    print(f"Starting PersonaPlex server on {host}:{port}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        reload=False,
    )

if __name__ == "__main__":
    main()
