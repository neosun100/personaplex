"""
PersonaPlex MCP Server
Model Context Protocol interface — delegates to the REST API server.
"""
import os
import sys
import json
import subprocess
from typing import Optional

import requests
from fastmcp import FastMCP

mcp = FastMCP("personaplex")

API_BASE = os.getenv("PERSONAPLEX_API_URL", "http://localhost:8998")

VOICE_OPTIONS = [
    "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
    "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
    "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
    "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
]


@mcp.tool()
def health_check() -> dict:
    """
    Check the health status of the PersonaPlex service.

    Returns:
        dict: Health status including GPU availability and model state
    """
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_gpu_status() -> dict:
    """
    Get current GPU status and memory usage.

    Returns:
        dict: GPU status including name, memory usage, model load state,
              active connections, and idle timeout
    """
    try:
        r = requests.get(f"{API_BASE}/api/gpu/status", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def offload_gpu() -> dict:
    """
    Release GPU memory by unloading the model.
    Uses gc.collect + torch.cuda.ipc_collect for thorough VRAM release.

    Returns:
        dict: Status with remaining memory_allocated in MB
    """
    try:
        r = requests.post(f"{API_BASE}/api/gpu/offload", timeout=30)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def list_voices() -> dict:
    """
    List all 18 available voice options for PersonaPlex.

    Returns:
        dict: List of voice objects with id, name, and category
    """
    try:
        r = requests.get(f"{API_BASE}/api/voices", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_prompt_examples() -> dict:
    """
    Get example text prompts for different use cases.

    Returns:
        dict: List of example prompts with id, name, and text
    """
    try:
        r = requests.get(f"{API_BASE}/api/prompts", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def process_audio(
    input_file: str,
    output_file: str,
    voice_prompt: str = "NATF2.pt",
    text_prompt: str = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    seed: int = 42424242,
) -> dict:
    """
    Process an audio file with PersonaPlex for offline inference.

    Args:
        input_file: Path to input WAV file (user audio)
        output_file: Path to output WAV file (AI response)
        voice_prompt: Voice ID to use (e.g., "NATF2.pt", "NATM1.pt")
        text_prompt: Text prompt defining the AI's persona/role
        seed: Random seed for reproducibility (-1 for random)

    Returns:
        dict: Processing result with output file path and generated text
    """
    try:
        if not os.path.exists(input_file):
            return {"status": "error", "error": f"Input file not found: {input_file}"}

        if voice_prompt not in VOICE_OPTIONS:
            return {"status": "error", "error": f"Invalid voice. Options: {VOICE_OPTIONS}"}

        with open(input_file, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/offline",
                files={"file": (os.path.basename(input_file), f, "audio/wav")},
                data={
                    "voice_prompt": voice_prompt,
                    "text_prompt": text_prompt,
                    "seed": str(seed),
                },
                timeout=300,
            )

        if r.status_code != 200:
            return {"status": "error", "error": f"HTTP {r.status_code}: {r.text}"}

        with open(output_file, "wb") as f:
            f.write(r.content)

        text_output = r.headers.get("X-Text-Output", "[]")
        try:
            generated = "".join(json.loads(text_output))
        except Exception:
            generated = ""

        return {
            "status": "success",
            "output_audio": output_file,
            "generated_text": generated,
            "voice_used": voice_prompt,
            "text_prompt_used": text_prompt,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    mcp.run()
