FROM ghcr.io/ggml-org/llama.cpp:server-cuda

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Work around broken/missing shared-library symlinks in some upstream llama.cpp images.
# Ensures both libfoo.so and libfoo.so.<major> point to the versioned .so file.
RUN set -eux; \
    if [ -d /app/lib ]; then \
        for f in /app/lib/*.so.[0-9]*; do \
            [ -e "$f" ] || continue; \
            base="$(basename "$f")"; \
            stem="${base%%.so.*}.so"; \
            major="$(printf '%s' "$base" | sed -E 's/^.*\.so\.([0-9]+).*$/\1/')"; \
            ln -sf "$base" "/app/lib/${stem}"; \
            ln -sf "$base" "/app/lib/${stem}.${major}"; \
        done; \
    fi

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py

ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}
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
