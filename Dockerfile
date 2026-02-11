FROM ghcr.io/ggml-org/llama.cpp:full-cuda

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Work around broken/missing shared-library symlinks in some upstream llama.cpp images.
# Ensures both libfoo.so and libfoo.so.<major> point to the versioned .so file.
RUN set -eux; \
    for d in /app/lib /app/build/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu; do \
        [ -d "$d" ] || continue; \
        for f in "$d"/*.so.[0-9]*; do \
            [ -e "$f" ] || continue; \
            base="${f##*/}"; \
            stem="${base%%.so.*}.so"; \
            major="${base#*.so.}"; \
            major="${major%%.*}"; \
            ln -sf "$base" "$d/${stem}"; \
            ln -sf "$base" "$d/${stem}.${major}"; \
        done; \
    done

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
