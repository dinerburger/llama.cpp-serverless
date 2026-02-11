# llama.cpp on RunPod Serverless

> Run any GGUF model as an OpenAI-compatible API on RunPod Serverless GPUs — zero infrastructure to manage.

This project packages [llama.cpp](https://github.com/ggerganov/llama.cpp) as a [RunPod Serverless](https://www.runpod.io/serverless-gpu) worker. A lightweight Python handler boots `llama-server` inside a CUDA-enabled container, proxies inference requests through an OpenAI-compatible interface, and scales to zero when idle.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
- **Any GGUF model** — Load models from Hugging Face Hub or a local/network-volume path
- **Full GPU offload** — Runs on the `llama.cpp` CUDA backend with configurable GPU layers
- **Auto-lifecycle management** — `llama-server` starts on first request, shuts down on container exit
- **Scale to zero** — RunPod Serverless billing only when actively processing requests
- **Flexible payload** — Send a simple `prompt` string or a full OpenAI-style `payload` object

## Architecture

![Architecture Diagram](./docs/diagrams/architecture.svg)

The worker runs inside a RunPod Serverless container. When a job arrives:

1. **RunPod Gateway** dispatches the job to an available worker container.
2. **`handler.py`** ensures `llama-server` is running (starts it on first call).
3. The handler builds an OpenAI-compatible payload and POSTs it to `localhost:8080`.
4. **`llama-server`** runs inference on the GPU and returns the result.
5. The handler wraps the response and returns it through RunPod to the client.

## Data Flow

![Data Flow Diagram](./docs/diagrams/data-flow.svg)

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- A [RunPod](https://www.runpod.io/) account (for deployment)
- A GGUF model (the default pulls one from Hugging Face automatically)

### Build the Image

```bash
docker build -t <your-dockerhub-user>/llama-cpp-runpod:latest .
docker push <your-dockerhub-user>/llama-cpp-runpod:latest
```

### Deploy to RunPod

1. Create a **Serverless Template** in the RunPod console with your image.
2. Choose a GPU type appropriate for your model size (e.g., RTX A5000 for 7B Q4).
3. Set environment variables (see [Configuration](#configuration)).
4. Create an **Endpoint** from the template.
5. Send requests to `https://api.runpod.ai/v2/<endpoint-id>/runsync`.

See [`runpod-endpoint-config.example.json`](runpod-endpoint-config.example.json) for a sample endpoint configuration.

## Request Format

Requests are sent as RunPod jobs via `/run` (async) or `/runsync` (synchronous).

### Simple prompt

```json
{
  "input": {
    "prompt": "Explain why GGUF is used with llama.cpp in 3 bullet points.",
    "temperature": 0.2,
    "max_tokens": 200
  }
}
```

### Full OpenAI-compatible payload

```json
{
  "input": {
    "endpoint": "/v1/chat/completions",
    "payload": {
      "model": "huihui-glm-4.7-flash-abliterated",
      "messages": [
        {"role": "user", "content": "Say hello in one sentence."}
      ],
      "temperature": 0.2,
      "max_tokens": 64,
      "stream": false
    }
  }
}
```

### Response structure

```json
{
  "status_code": 200,
  "ok": true,
  "endpoint": "/v1/chat/completions",
  "result": { "...": "OpenAI-compatible response from llama-server" }
}
```

### Supported optional parameters

When using the simple format (without `payload`), these keys are forwarded:

| Parameter | Description |
|-----------|-------------|
| `messages` | Full chat messages array (overrides `prompt`) |
| `model` | Model alias to use |
| `temperature` | Sampling temperature |
| `top_p` | Nucleus sampling threshold |
| `top_k` | Top-k sampling |
| `max_tokens` | Maximum tokens to generate |
| `presence_penalty` | Presence penalty |
| `frequency_penalty` | Frequency penalty |
| `stop` | Stop sequences |
| `response_format` | Response format (e.g., JSON mode) |
| `tools` | Function/tool definitions |
| `tool_choice` | Tool selection strategy |
| `params` | Dict of additional parameters merged into payload |

## Configuration

All configuration is via environment variables, set either in the Dockerfile or in the RunPod endpoint settings.

### Model configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Local path to a GGUF file (takes priority) | — |
| `MODEL_HF_REPO` | Hugging Face repo containing GGUF files | `DevQuasar/huihui-ai.Huihui-GLM-4.7-Flash-abliterated-GGUF` |
| `MODEL_HF_FILE` | Specific file within the HF repo | — (auto-detected) |
| `MODEL_ALIAS` | Model name returned in API responses | `huihui-glm-4.7-flash-abliterated` |
| `HF_TOKEN` | Hugging Face token for gated/private repos | — |

> **Note:** `llama.cpp` requires GGUF format. Safetensors models must be converted first.

### Server tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_CTX_SIZE` | Context window size (tokens) | `8192` |
| `LLAMA_N_GPU_LAYERS` | Number of layers offloaded to GPU | `999` (all) |
| `LLAMA_PARALLEL` | Number of parallel inference slots | `1` |
| `LLAMA_THREADS_HTTP` | HTTP server threads | `4` |
| `LLAMA_EXTRA_ARGS` | Additional CLI flags for llama-server (space-separated) | — |
| `LLAMA_API_KEY` | API key for llama-server authentication | — |
| `LLAMA_DISABLE_WEBUI` | Disable the built-in web UI | `1` (disabled) |

### Timeouts

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_START_TIMEOUT_SECONDS` | Max wait for llama-server health check | `900` |
| `SERVER_REQUEST_TIMEOUT_SECONDS` | Max wait for a single inference request | `600` |

### Internal (rarely changed)

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SERVER_BIN` | Path to llama-server binary | `/app/llama-server` |
| `LLAMA_SERVER_HOST` | Bind address for llama-server | `127.0.0.1` |
| `LLAMA_SERVER_PORT` | Port for llama-server | `8080` |

## Project Structure

```
.
├── handler.py                  # RunPod worker: server lifecycle + request proxying
├── Dockerfile                  # CUDA-enabled image based on llama.cpp
├── requirements.txt            # Python dependencies (runpod, requests)
├── request-example.json        # Sample RunPod request body
├── runpod-endpoint-config.example.json  # Sample endpoint configuration
└── docs/
    └── diagrams/
        ├── architecture.drawio # System architecture (editable)
        ├── architecture.svg    # System architecture (rendered)
        ├── data-flow.drawio    # Request data flow (editable)
        └── data-flow.svg       # Request data flow (rendered)
```

## Testing

There is no formal test suite. Use these verification steps:

```bash
# Syntax check
python3 -m py_compile handler.py

# Local container smoke test (requires NVIDIA GPU)
docker run --rm --gpus all \
  -e MODEL_HF_REPO=DevQuasar/huihui-ai.Huihui-GLM-4.7-Flash-abliterated-GGUF \
  <your-image>:latest

# Send a test request to your RunPod endpoint
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d @request-example.json
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes using imperative mood (`fix: handle missing MODEL_HF_REPO`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request with purpose, config changes, and sample request/response

### Code style

- Python 3, PEP 8, 4-space indentation
- `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for environment-driven constants
- Keep handler logic explicit and defensive

## License

See [LICENSE](LICENSE) for details.
