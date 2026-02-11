# llama.cpp on Runpod Serverless

This project packages `llama.cpp` as a Runpod Serverless worker.

Target model family:
- `huihui-ai/Huihui-GLM-4.7-Flash-abliterated` (original repo, safetensors)

Important:
- `llama.cpp` serves GGUF models. The original Flash repo is safetensors, so it cannot be loaded directly by `llama-server`.
- Use a GGUF repo/file for runtime (default is set to a GGUF mirror for the Flash variant).
- No localhost dependency for clients. You call your Runpod endpoint only.

## Files

- `Dockerfile`: Worker image based on `ghcr.io/ggml-org/llama.cpp:server-cuda`
- `handler.py`: Runpod handler that boots `llama-server` once and proxies inference jobs
- `requirements.txt`: Python deps (`runpod`, `requests`)

## Runtime model configuration

Set one of these in Runpod endpoint environment variables:

1. `MODEL_PATH` (local GGUF path in container/network volume)
2. `MODEL_HF_REPO` (+ optional `MODEL_HF_FILE`) for Hugging Face GGUF repository download

Default in `Dockerfile`:
- `MODEL_HF_REPO=DevQuasar/huihui-ai.Huihui-GLM-4.7-Flash-abliterated-GGUF`

Optional auth for gated/private repos:
- `HF_TOKEN=<token>`

## Other useful environment variables

- `MODEL_ALIAS=huihui-glm-4.7-flash-abliterated`
- `LLAMA_CTX_SIZE=8192`
- `LLAMA_N_GPU_LAYERS=999`
- `LLAMA_PARALLEL=1`
- `LLAMA_THREADS_HTTP=4`
- `LLAMA_EXTRA_ARGS=--flash-attn on`
- `SERVER_START_TIMEOUT_SECONDS=900`
- `SERVER_REQUEST_TIMEOUT_SECONDS=600`
- `LLAMA_API_KEY=<optional-api-key>`

## Build and push image

```bash
docker build -t <dockerhub-user>/llama-cpp-runpod:latest .
docker push <dockerhub-user>/llama-cpp-runpod:latest
```

## Configure Runpod Serverless

1. Create a Runpod Serverless template with container image `<dockerhub-user>/llama-cpp-runpod:latest`.
2. Set GPU type/count for your expected model quantization size.
3. Configure autoscaling/timeouts in endpoint settings.
4. Add the environment variables above in the endpoint.
5. Deploy endpoint.

Example endpoint body for `POST https://rest.runpod.io/v1/endpoints`:
- `runpod-endpoint-config.example.json`

Reference docs used for setup:
- Runpod worker pattern: `runpod.serverless.start({"handler": handler})`
- Runpod endpoint config supports workers min/max, idle timeout, execution timeout

## Request format (Runpod `/run` or `/runsync`)

### Simplest request

```json
{
  "input": {
    "prompt": "Write a concise summary of GGUF.",
    "max_tokens": 256,
    "temperature": 0.2
  }
}
```

### Explicit OpenAI-compatible payload passthrough

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

`handler.py` returns:
- `status_code`
- `ok`
- `endpoint`
- `result` (JSON from `llama-server`)

## Notes on model choice

Because the requested source model repo is not GGUF, you must either:
- provide a GGUF Flash repo/file in `MODEL_HF_REPO` / `MODEL_HF_FILE`, or
- mount a pre-converted GGUF via `MODEL_PATH`.
