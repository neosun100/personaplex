# PersonaPlex MCP Integration Guide

[English](MCP_GUIDE.md) | [简体中文](MCP_GUIDE_CN.md)

## Overview

PersonaPlex provides a Model Context Protocol (MCP) server for programmatic access to its full-duplex conversational AI capabilities. This allows integration with AI assistants, automation tools, and custom applications.

## Quick Start

### 1. Configure MCP Client

Add to your MCP configuration (e.g., `~/.config/claude/mcp.json`):

```json
{
  "mcpServers": {
    "personaplex": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/personaplex",
      "env": {
        "HF_TOKEN": "your_huggingface_token",
        "GPU_IDLE_TIMEOUT": "300"
      }
    }
  }
}
```

### 2. Docker-based MCP

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

## Available Tools

### `health_check`
Check service health and GPU availability.

```python
result = await mcp.call_tool("health_check", {})
# Returns: {"status": "healthy", "gpu_available": true, ...}
```

### `get_gpu_status`
Get detailed GPU memory and status information.

```python
result = await mcp.call_tool("get_gpu_status", {})
# Returns: {"gpu": {"name": "NVIDIA L40S", "memory_total_mb": 46068, ...}}
```

### `offload_gpu`
Release GPU memory by unloading the model.

```python
result = await mcp.call_tool("offload_gpu", {})
# Returns: {"status": "offloaded"}
```

### `list_voices`
Get all available voice options.

```python
result = await mcp.call_tool("list_voices", {})
# Returns: {"voices": [{"id": "NATF0.pt", "name": "Natural Female 0", ...}, ...]}
```

### `get_prompt_examples`
Get example text prompts for different scenarios.

```python
result = await mcp.call_tool("get_prompt_examples", {})
# Returns: {"examples": [{"name": "Assistant", "prompt": "...", ...}, ...]}
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

## Prompt Examples

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

## Error Handling

All tools return a `status` field:
- `"success"` - Operation completed successfully
- `"error"` - Operation failed, check `error` field for details

```python
result = await mcp.call_tool("process_audio", {...})
if result["status"] == "error":
    print(f"Error: {result['error']}")
```

## Best Practices

1. **GPU Management**: Call `offload_gpu` after processing to free memory
2. **File Paths**: Use `/tmp/personaplex/` for temporary files
3. **Voice Selection**: Use NAT voices for natural conversations
4. **Seed Control**: Use fixed seed for reproducible results

## Comparison: MCP vs REST API

| Feature | MCP | REST API |
|---------|-----|----------|
| Use Case | Programmatic/AI integration | Web/HTTP clients |
| Protocol | stdio/JSON-RPC | HTTP/REST |
| Real-time | ❌ | ✅ (WebSocket) |
| Batch Processing | ✅ | ✅ |
| GPU Management | ✅ | ✅ |

## Troubleshooting

### Model not loading
```bash
# Check HF_TOKEN is set
echo $HF_TOKEN

# Accept model license at:
# https://huggingface.co/nvidia/personaplex-7b-v1
```

### Out of GPU memory
```bash
# Enable CPU offload
export CPU_OFFLOAD=true

# Or call offload_gpu before processing
```

### Connection issues
```bash
# Check if container is running
docker ps | grep personaplex

# Check logs
docker logs personaplex
```
