FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS aptdeps

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

FROM aptdeps AS withmodel

WORKDIR /
COPY model/Qwen3.5-0.8B-UD-Q6_K_XL.gguf /model/model.gguf
COPY model/Qwen3.5-0.8B-mmproj.gguf /model/mmproj.gguf

FROM withmodel AS final

WORKDIR /

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py
COPY engine.py /engine.py
COPY utils.py /utils.py

ENV PYTHONUNBUFFERED=1

# Discover every directory under /app that contains shared libraries
# and register them with the dynamic linker so llama-server can find
# libmtmd, libllama, libggml, etc. regardless of where the base image
# places them.
RUN find /app -name "*.so*" -exec dirname {} \; | sort -u \
      > /etc/ld.so.conf.d/llama.conf && ldconfig
ENV LLAMA_SERVER_BIN=/app/llama-server
ENV LLAMA_SERVER_HOST=127.0.0.1
ENV LLAMA_SERVER_PORT=8080
ENV LLAMA_CTX_SIZE=0
ENV LLAMA_N_GPU_LAYERS=999
ENV LLAMA_THREADS_HTTP=4
ENV LLAMA_DISABLE_WEBUI=1
ENV LLAMA_LOG_FILE=/llama.log
ENV LLAMA_LOG_VERBOSITY=4
ENV LLAMA_ARG_FLASH_ATTN=1

# GGUF default for the requested Flash model family.
# Override with MODEL_PATH
ENV MODEL_PATH=/model/model.gguf
ENV MMPROJ_PATH=/model/mmproj.gguf
ENV MODEL_ALIAS="Qwen3.5-27B"

# The base image sets an entrypoint dispatcher that only recognises
# llama.cpp sub-commands (--server, --run, etc.).  Reset it so the
# container executes our Python handler directly.
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
