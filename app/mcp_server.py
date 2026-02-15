"""
PersonaPlex MCP Server
Model Context Protocol interface for programmatic access
"""
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("personaplex")

# GPU Manager (shared with main server)
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
        self.model_loaded = False
    
    def get_status(self) -> dict:
        status = {"model_loaded": self.model_loaded}
        if torch.cuda.is_available():
            try:
                status["gpu"] = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2),
                    "memory_allocated_mb": torch.cuda.memory_allocated(0) // (1024**2),
                    "memory_reserved_mb": torch.cuda.memory_reserved(0) // (1024**2),
                }
            except Exception as e:
                status["gpu_error"] = str(e)
        else:
            status["gpu"] = None
        return status
    
    def offload(self) -> dict:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_loaded = False
        return {"status": "offloaded", "message": "GPU memory released"}

gpu_manager = GPUManager()

VOICE_OPTIONS = [
    "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
    "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
    "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
    "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
]

@mcp.tool()
def get_gpu_status() -> dict:
    """
    Get current GPU status and memory usage.
    
    Returns:
        dict: GPU status including name, memory usage, and model load state
    """
    return gpu_manager.get_status()

@mcp.tool()
def offload_gpu() -> dict:
    """
    Release GPU memory by unloading the model.
    
    Returns:
        dict: Status of the offload operation
    """
    return gpu_manager.offload()

@mcp.tool()
def list_voices() -> dict:
    """
    List all available voice options for PersonaPlex.
    
    Returns:
        dict: List of voice IDs with descriptions
    """
    voices = []
    for v in VOICE_OPTIONS:
        name = v.replace('.pt', '')
        category = "Natural" if name.startswith("NAT") else "Variety"
        gender = "Female" if "F" in name else "Male"
        voices.append({
            "id": v,
            "name": f"{category} {gender} {name[-1]}",
            "category": category.lower(),
            "gender": gender.lower()
        })
    return {"voices": voices}

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
        
        output_text = output_file.replace('.wav', '.json')
        
        cmd = [
            sys.executable, "-m", "moshi.offline",
            "--voice-prompt", voice_prompt,
            "--text-prompt", text_prompt,
            "--input-wav", input_file,
            "--output-wav", output_file,
            "--output-text", output_text,
            "--seed", str(seed),
        ]
        
        if os.getenv("CPU_OFFLOAD", "false").lower() == "true":
            cmd.append("--cpu-offload")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        
        if result.returncode != 0:
            return {"status": "error", "error": result.stderr}
        
        # Read generated text
        text_output = []
        if os.path.exists(output_text):
            with open(output_text) as f:
                text_output = json.load(f)
        
        gpu_manager.offload()
        
        return {
            "status": "success",
            "output_audio": output_file,
            "output_text": output_text,
            "generated_text": "".join(text_output),
            "voice_used": voice_prompt,
            "text_prompt_used": text_prompt,
        }
        
    except Exception as e:
        gpu_manager.offload()
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_prompt_examples() -> dict:
    """
    Get example text prompts for different use cases.
    
    Returns:
        dict: List of example prompts with descriptions
    """
    return {
        "examples": [
            {
                "name": "Assistant",
                "description": "General-purpose helpful assistant",
                "prompt": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
            },
            {
                "name": "Customer Service - Bank",
                "description": "Bank customer service representative",
                "prompt": "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity."
            },
            {
                "name": "Customer Service - Restaurant",
                "description": "Restaurant order taker",
                "prompt": "You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25)."
            },
            {
                "name": "Casual Conversation",
                "description": "Open-ended casual chat",
                "prompt": "You enjoy having a good conversation."
            },
            {
                "name": "Astronaut Scenario",
                "description": "Fun roleplay scenario",
                "prompt": "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex."
            }
        ]
    }

@mcp.tool()
def health_check() -> dict:
    """
    Check the health status of the PersonaPlex service.
    
    Returns:
        dict: Health status including GPU availability and model state
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": gpu_manager.model_loaded,
        "python_version": sys.version,
    }

if __name__ == "__main__":
    mcp.run()
