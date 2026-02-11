FROM ghcr.io/ggml-org/llama.cpp:full-cuda

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Work around broken/missing shared-library symlinks in some upstream llama.cpp images.
# Use Python here to avoid shell/coreutils differences across base image variants.
RUN python3 - <<'PY'
import glob
import os
import re

dirs = [
    "/app/lib",
    "/app/build/lib",
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
]

for d in dirs:
    if not os.path.isdir(d):
        continue
    for f in glob.glob(os.path.join(d, "*.so.[0-9]*")):
        base = os.path.basename(f)
        m = re.match(r"^(.*\.so)\.([0-9]+).*$", base)
        if not m:
            continue
        stem, major = m.group(1), m.group(2)
        for link_name in (stem, f"{stem}.{major}"):
            link_path = os.path.join(d, link_name)
            try:
                if os.path.lexists(link_path):
                    os.remove(link_path)
                os.symlink(base, link_path)
            except OSError:
                pass
PY

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py

ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/app/lib:/app/build/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib:${LD_LIBRARY_PATH}
ENV LLAMA_SERVER_BIN=/app/llama-server
ENV LLAMA_SERVER_HOST=127.0.0.1
ENV LLAMA_SERVER_PORT=8080
ENV LLAMA_CTX_SIZE=8192
ENV LLAMA_N_GPU_LAYERS=999
ENV LLAMA_THREADS_HTTP=4
ENV LLAMA_DISABLE_WEBUI=1

# GGUF default for the requested Flash model family.
# Override with MODEL_PATH or MODEL_HF_REPO/MODEL_HF_FILE as needed.
ENV MODEL_HF_REPO=DevQuasar/huihui-ai.Huihui-GLM-4.7-Flash-abliterated-GGUF

CMD ["python3", "-u", "/handler.py"]
