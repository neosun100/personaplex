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
import random
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import gc

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from moshi.models import loaders, MimiModel, LMModel, LMGen


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


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
        self.voice_prompt_dir = None
        self.lock = asyncio.Lock()
        self.last_used = None
        self.idle_timeout = int(os.getenv("GPU_IDLE_TIMEOUT", "300"))
        self.active_connections = 0

    def get_status(self) -> dict:
        status = {
            "model_loaded": self.mimi is not None,
            "device": str(self.device) if self.device else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "active_connections": self.active_connections,
            "idle_timeout": self.idle_timeout,
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
        """Thoroughly release all GPU memory."""
        # Reset streaming states first (clears CUDAGraph caches)
        if self.lm_gen is not None:
            try:
                self.lm_gen.reset_streaming()
            except Exception:
                pass
            if hasattr(self.lm_gen, 'lm_model'):
                del self.lm_gen.lm_model
            del self.lm_gen
            self.lm_gen = None
        if self.mimi is not None:
            try:
                self.mimi.reset_streaming()
            except Exception:
                pass
            del self.mimi
            self.mimi = None
        if self.other_mimi is not None:
            try:
                self.other_mimi.reset_streaming()
            except Exception:
                pass
            del self.other_mimi
            self.other_mimi = None
        if self.model is not None:
            del self.model
            self.model = None
        self.text_tokenizer = None
        # Force garbage collection to break circular refs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return {"status": "offloaded", "memory_allocated": torch.cuda.memory_allocated(0) // (1024**2) if torch.cuda.is_available() else 0}

    def load_model(self):
        """Load model if not already loaded."""
        if self.mimi is not None:
            return

        device_str = os.getenv("DEVICE", "cuda")
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        hf_repo = loaders.DEFAULT_REPO
        cpu_offload = os.getenv("CPU_OFFLOAD", "false").lower() == "true"

        print("Loading mimi...")
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
        print("Mimi loaded.")

        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        print("Loading moshi LM...")
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=cpu_offload)
        lm.eval()
        print("Moshi LM loaded.")

        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )

        # Get voice prompt dir
        import tarfile
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        voices_tgz = Path(voices_tgz)
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            print(f"Extracting {voices_tgz}...")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        self.voice_prompt_dir = str(voices_dir)

        # Start streaming
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Warmup
        print("Warming up...")
        frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("Model ready.")
        self.last_used = datetime.utcnow()


gpu_manager = GPUManager()

# Background idle check task
async def idle_check_loop():
    """Periodically check if GPU should be offloaded due to inactivity."""
    while True:
        await asyncio.sleep(60)  # check every minute
        gm = gpu_manager
        if gm.mimi is None:
            continue
        if gm.active_connections > 0:
            continue
        if gm.last_used is None:
            continue
        elapsed = (datetime.utcnow() - gm.last_used).total_seconds()
        if elapsed >= gm.idle_timeout:
            print(f"[GPU] Idle for {elapsed:.0f}s (timeout={gm.idle_timeout}s), offloading...")
            gm.offload()
            print(f"[GPU] Offloaded. Memory allocated: {torch.cuda.memory_allocated(0)//(1024**2)}MB")

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
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(idle_check_loop())

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
        "model_loaded": gpu_manager.mimi is not None
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
        input_path = Path(tempfile.mktemp(suffix=".wav"))
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        output_text = Path(tempfile.mktemp(suffix=".json"))

        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

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
        for p in [input_path, output_text]:
            if p.exists():
                p.unlink()


@app.websocket("/api/chat")
async def websocket_chat(ws: WebSocket):
    """Real-time full-duplex conversation via WebSocket.

    Protocol (binary frames):
      Client -> Server: 0x01 + opus audio bytes
      Server -> Client: 0x00 (handshake)
      Server -> Client: 0x01 + opus audio bytes
      Server -> Client: 0x02 + utf-8 text token
    """
    await ws.accept()

    # Parse query params
    text_prompt = ws.query_params.get("text_prompt", "")
    voice_prompt_name = ws.query_params.get("voice_prompt", "NATF2.pt")
    seed = ws.query_params.get("seed", None)

    try:
        # Ensure model is loaded
        gpu_manager.load_model()
    except Exception as e:
        await ws.send_bytes(b"\x03" + str(e).encode("utf-8"))
        await ws.close(code=1011, reason="Model load failed")
        return

    gm = gpu_manager
    frame_size = int(gm.mimi.sample_rate / gm.mimi.frame_rate)

    # Resolve voice prompt path
    voice_prompt_path = os.path.join(gm.voice_prompt_dir, voice_prompt_name)
    if not os.path.exists(voice_prompt_path):
        await ws.close(code=1008, reason=f"Voice prompt not found: {voice_prompt_name}")
        return

    close = False

    try:
        await asyncio.wait_for(gm.lock.acquire(), timeout=1.0)
    except asyncio.TimeoutError:
        await ws.send_bytes(b"\x03" + "Server busy, another session is active".encode("utf-8"))
        await ws.close(code=1013, reason="Server busy")
        return

    try:
        gm.last_used = datetime.utcnow()
        gm.active_connections += 1

        if seed is not None and seed != "-1":
            seed_all(int(seed))
        else:
            seed_all(42424242)

        # Load voice prompt
        if gm.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                gm.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                gm.lm_gen.load_voice_prompt(voice_prompt_path)

        # Set text prompt
        if text_prompt:
            gm.lm_gen.text_prompt_tokens = gm.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
        else:
            gm.lm_gen.text_prompt_tokens = None

        opus_writer = sphn.OpusStreamWriter(gm.mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(gm.mimi.sample_rate)

        gm.mimi.reset_streaming()
        gm.other_mimi.reset_streaming()
        gm.lm_gen.reset_streaming()

        # Process system prompts — send keepalive pings to prevent Cloudflare timeout
        async def is_alive():
            if close:
                return False
            try:
                await ws.send_bytes(b"\x04")  # keepalive ping
            except Exception:
                return False
            return True

        try:
            await gm.lm_gen.step_system_prompts_async(gm.mimi, is_alive=is_alive)
        except Exception as e:
            await ws.close(code=1011, reason=f"System prompt error: {e}")
            return

        gm.mimi.reset_streaming()

        # Send handshake
        await ws.send_bytes(b"\x00")

        async def recv_loop():
            nonlocal close
            try:
                while not close:
                    data = await ws.receive_bytes()
                    if len(data) == 0:
                        continue
                    kind = data[0]
                    if kind == 1:  # audio
                        opus_reader.append_bytes(data[1:])
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                close = True

        async def opus_loop():
          with torch.no_grad():
            all_pcm_data = None
            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data is not None and all_pcm_data.shape[-1] >= frame_size:
                    chunk = all_pcm_data[:frame_size]
                    all_pcm_data = all_pcm_data[frame_size:]
                    if all_pcm_data.shape[-1] == 0:
                        all_pcm_data = None
                    chunk = torch.from_numpy(chunk).to(device=gm.device)[None, None]
                    codes = gm.mimi.encode(chunk)
                    _ = gm.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = gm.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        main_pcm = gm.mimi.decode(tokens[:, 1:9])
                        _ = gm.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].detach().numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = gm.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            try:
                                await ws.send_bytes(b"\x02" + _text.encode("utf-8"))
                            except Exception:
                                return

        async def send_loop():
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    try:
                        await ws.send_bytes(b"\x01" + msg)
                    except Exception:
                        return

        tasks = [
            asyncio.create_task(recv_loop()),
            asyncio.create_task(opus_loop()),
            asyncio.create_task(send_loop()),
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        close = True
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            await ws.close()
        except Exception:
            pass

        gm.last_used = datetime.utcnow()
    finally:
        gm.active_connections = max(0, gm.active_connections - 1)
        gm.lock.release()


# Main entry point
def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8998"))

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
