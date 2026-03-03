import asyncio
import atexit
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List

import aiohttp
import runpod
from runpod import RunPodLogger

from engine import LlamaCppEngine
from utils import JobInput

log = RunPodLogger()
logger = logging.getLogger(__name__)

# Configuration
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "/app/llama-server")
LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

SERVER_START_TIMEOUT_SECONDS = int(os.getenv("SERVER_START_TIMEOUT_SECONDS", "900"))
SERVER_REQUEST_TIMEOUT_SECONDS = int(os.getenv("SERVER_REQUEST_TIMEOUT_SECONDS", "600"))

# Global state
_server_process: subprocess.Popen = None
_engine: LlamaCppEngine | None = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_server_command() -> List[str]:
    cmd = [
        LLAMA_SERVER_BIN,
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
        "--ctx-size",
        os.getenv("LLAMA_CTX_SIZE", "0"),
        "--n-gpu-layers",
        os.getenv("LLAMA_N_GPU_LAYERS", "999"),
        "--threads-http",
        os.getenv("LLAMA_THREADS_HTTP", "4"),
        "--parallel",
        os.getenv("LLAMA_PARALLEL", "1"),
    ]

    if _env_flag("LLAMA_DISABLE_WEBUI", True):
        cmd.append("--no-webui")

    model_path = os.getenv("MODEL_PATH", "").strip()
    hf_repo = os.getenv("MODEL_HF_REPO", "").strip()
    hf_file = os.getenv("MODEL_HF_FILE", "").strip()

    if model_path:
        cmd.extend(["--model", model_path])
    elif hf_repo:
        cmd.extend(["--hf-repo", hf_repo])
        if hf_file:
            cmd.extend(["--hf-file", hf_file])
    else:
        raise RuntimeError(
            "No model configured. Set MODEL_PATH or MODEL_HF_REPO (and optionally MODEL_HF_FILE)."
        )

    # Multimodal projector (mmproj) support for vision models
    mmproj_path = os.getenv("MMPROJ_PATH", "").strip()
    mmproj_url = os.getenv("MMPROJ_URL", "").strip()

    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])
    elif mmproj_url:
        cmd.extend(["--mmproj-url", mmproj_url])

    alias = os.getenv("MODEL_ALIAS", "huihui-glm-4.7-flash-abliterated").strip()
    if alias:
        cmd.extend(["--alias", alias])

    api_key = os.getenv("LLAMA_API_KEY", "").strip()
    if api_key:
        cmd.extend(["--api-key", api_key])

    extra_args = os.getenv("LLAMA_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(extra_args.split())

    return cmd


def _wait_for_server_ready() -> None:
    deadline = time.time() + SERVER_START_TIMEOUT_SECONDS
    health_url = f"{LLAMA_SERVER_URL}/health"

    while time.time() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            output = _server_process.stdout.read().decode(errors="replace")[-2000:]
            raise RuntimeError(
                f"llama-server exited with code {_server_process.returncode}: {output}"
            )

        try:
            import requests
            response = requests.get(health_url, timeout=10)
            if response.ok:
                return
        except Exception:
            pass

        time.sleep(2)

    raise TimeoutError(
        f"llama-server did not become ready within {SERVER_START_TIMEOUT_SECONDS} seconds"
    )


def _ensure_server_running() -> None:
    global _server_process

    if _server_process is not None and _server_process.poll() is None:
        return

    command = _build_server_command()
    _server_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )


def _stop_server() -> None:
    global _server_process

    if _server_process is None:
        return

    if _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _server_process.kill()

    _server_process = None


async def handler(job: Dict[str, Any]):
    """Async handler for RunPod serverless."""
    try:
        # log.info(f"Received job: {job}")
        job_input = JobInput(job)
        # log.info(f"JobInput: route={job_input.openai_route}, stream={job_input.stream}")
        
        async for batch in _engine.generate(job_input):
            batch_preview = str(batch)[:200] if batch else 'empty'
            # log.info(f"Yielding batch: {batch_preview}")
            yield batch
    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")

        # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str}


# Only run in main process to prevent re-initialization
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    import asyncio

    try:
        # Initialize the engine
        _engine = LlamaCppEngine()
        log.info("LlamaCppEngine initialized successfully")

        # Ensure server is running
        _ensure_server_running()
        _wait_for_server_ready()

    except Exception as e:
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )


atexit.register(_stop_server)