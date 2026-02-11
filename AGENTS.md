# Repository Guidelines

## Project Structure & Module Organization
This repository packages `llama.cpp` as a Runpod Serverless worker.

- `handler.py`: main worker entrypoint; starts `llama-server`, builds request payloads, and proxies inference calls.
- `Dockerfile`: runtime image based on `ghcr.io/ggml-org/llama.cpp:server-cuda`.
- `requirements.txt`: Python runtime dependencies (`runpod`, `requests`).
- `request-example.json`: minimal `/runsync` input example.
- `runpod-endpoint-config.example.json`: example endpoint configuration payload.
- `README.md`: deployment and environment variable reference.

## Build, Test, and Development Commands
- `docker build -t <user>/llama-cpp-runpod:latest .`
  Builds the worker image.
- `docker run --rm -p 8000:8000 --env MODEL_HF_REPO=<gguf-repo> <user>/llama-cpp-runpod:latest`
  Starts the container locally (adjust env vars as needed).
- `pip3 install -r requirements.txt`
  Installs Python dependencies for local script checks.
- `python3 -m py_compile handler.py`
  Fast syntax check before building/pushing.

## Coding Style & Naming Conventions
- Follow Python 3 style with 4-space indentation and PEP 8 naming.
- Use `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for env-driven constants.
- Keep handler logic explicit and defensive: validate env vars, return structured errors, and avoid hidden side effects.
- Prefer small helper functions (`_build_server_command`, `_wait_for_server_ready`) over large inline blocks.

## Testing Guidelines
There is no formal test suite in this snapshot. Use focused verification:

- Syntax check: `python3 -m py_compile handler.py`
- Container smoke test: start image and confirm server health path responds internally.
- Endpoint behavior test: send `request-example.json` through your Runpod endpoint and verify `status_code`, `ok`, and `result`.

When adding tests, place them in `tests/` and use `test_<feature>.py` naming.

## Commit & Pull Request Guidelines
Git history is not available in this workspace snapshot, so use a clear default:

- Commit messages in imperative mood, e.g. `fix: handle missing MODEL_HF_REPO`.
- Keep commits scoped to one change area (startup, payload mapping, timeout handling, etc.).
- PRs should include: purpose, config/env changes, validation steps, and sample request/response snippets for behavior changes.

## Security & Configuration Tips
- Never hardcode secrets; provide `HF_TOKEN` and `LLAMA_API_KEY` via environment variables.
- Prefer `MODEL_PATH` or approved GGUF repos only; `llama.cpp` cannot load safetensors directly.
