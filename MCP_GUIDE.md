# PersonaPlex MCP Integration Guide

[English](MCP_GUIDE.md) | [简体中文](MCP_GUIDE_CN.md)

## Overview

PersonaPlex provides a Model Context Protocol (MCP) server for programmatic access to its full-duplex conversational AI capabilities. The MCP server delegates to the REST API, ensuring consistent behavior and proper GPU management (including v1.3.0 thorough VRAM release).

## Quick Start

### 1. Docker-based MCP (Recommended)

```json
{
  "mcpServers": {
    "personaplex": {
      "command": "docker",
      "args": [
        "exec", "-i", "personaplex",
        "python", "-m", "app.mcp_server"
      ]
    }
  }
}
```

### 2. Local MCP

```json
{
  "mcpServers": {
    "personaplex": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/personaplex",
      "env": {
        "PERSONAPLEX_API_URL": "http://localhost:8998"
      }
    }
  }
}
```

## Available Tools

### `health_check`
Check service health and GPU availability.

```python
result = await mcp.call_tool("health_check", {})
# Returns: {"status": "healthy", "gpu_available": true, "model_loaded": false, ...}
```

### `get_gpu_status`
Get detailed GPU memory, active connections, and idle timeout info.

```python
result = await mcp.call_tool("get_gpu_status", {})
# Returns: {
#   "model_loaded": true,
#   "active_connections": 0,
#   "idle_timeout": 300,
#   "gpu": {"name": "NVIDIA L40S", "memory_allocated": 18334, ...}
# }
```

### `offload_gpu`
Thoroughly release GPU memory (gc.collect + torch.cuda.ipc_collect).

```python
result = await mcp.call_tool("offload_gpu", {})
# Returns: {"status": "offloaded", "memory_allocated": 16}
```

### `list_voices`
Get all 18 available voice options.

```python
result = await mcp.call_tool("list_voices", {})
# Returns: {"voices": [{"id": "NATF0.pt", "name": "Natural Female 0", ...}, ...]}
```

### `get_prompt_examples`
Get example text prompts for different scenarios.

```python
result = await mcp.call_tool("get_prompt_examples", {})
# Returns: {"prompts": [{"id": "assistant", "name": "Assistant", "text": "...", ...}, ...]}
```

### `process_audio`
Process an audio file with PersonaPlex (main inference tool).

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_file` | string | ✅ | - | Path to input WAV file |
| `output_file` | string | ✅ | - | Path for output WAV file |
| `voice_prompt` | string | ❌ | "NATF2.pt" | Voice ID to use |
| `text_prompt` | string | ❌ | (assistant) | Persona/role prompt |
| `seed` | int | ❌ | 42424242 | Random seed (-1 for random) |

**Example:**
```python
result = await mcp.call_tool("process_audio", {
    "input_file": "/tmp/personaplex/input.wav",
    "output_file": "/tmp/personaplex/output.wav",
    "voice_prompt": "NATM1.pt",
    "text_prompt": "You work for a bank and your name is Alex.",
    "seed": 12345
})
# Returns: {
#   "status": "success",
#   "output_audio": "/tmp/personaplex/output.wav",
#   "generated_text": "Hello, how can I help you today?",
#   ...
# }
```

## Voice Options

| Category | IDs | Description |
|----------|-----|-------------|
| Natural Female | NATF0-3.pt | Natural, conversational female voices |
| Natural Male | NATM0-3.pt | Natural, conversational male voices |
| Variety Female | VARF0-4.pt | Diverse female voice styles |
| Variety Male | VARM0-4.pt | Diverse male voice styles |

## Error Handling

All tools return a `status` field on error:

```python
result = await mcp.call_tool("process_audio", {...})
if "error" in result:
    print(f"Error: {result['error']}")
```

## Architecture

```
MCP Client (Claude, etc.)
    ↓ stdio/JSON-RPC
MCP Server (app/mcp_server.py)
    ↓ HTTP requests
REST API Server (app/server.py)
    ↓ GPU inference
PersonaPlex Model
```

The MCP server is a thin client that delegates all operations to the REST API. This ensures:
- Consistent GPU management (offload truly releases VRAM)
- Accurate status reporting (active connections, idle timer)
- No duplicate model loading

## Comparison: MCP vs REST API

| Feature | MCP | REST API |
|---------|-----|----------|
| Use Case | Programmatic/AI integration | Web/HTTP clients |
| Protocol | stdio/JSON-RPC | HTTP/REST |
| Real-time Chat | ❌ | ✅ (WebSocket) |
| Offline Inference | ✅ | ✅ |
| GPU Management | ✅ | ✅ |

## Troubleshooting

### MCP can't connect to API
```bash
# Ensure container is running
docker ps | grep personaplex

# Check API is accessible
docker exec personaplex curl -s http://localhost:8998/health
```

### Model not loading
```bash
# Check HF_TOKEN is set
echo $HF_TOKEN

# Accept model license at:
# https://huggingface.co/nvidia/personaplex-7b-v1
```

### Out of GPU memory
```bash
# Call offload first
docker exec personaplex curl -X POST http://localhost:8998/api/gpu/offload

# Or enable CPU offload
export CPU_OFFLOAD=true
```
