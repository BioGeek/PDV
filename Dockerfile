FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INOVO_ROOT=/opt/instanovo-versions \
    INOVO_VENV=/opt/instanovo-env

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      tini \
      unzip \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages "${INOVO_VENV}" \
    && "${INOVO_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

RUN mkdir -p "${INOVO_ROOT}" \
    && for tag in 1.0.0 1.1.0 1.1.3 1.2.2; do \
         git clone --depth 1 --branch "${tag}" https://github.com/instadeepai/InstaNovo.git "${INOVO_ROOT}/${tag}"; \
       done

RUN "${INOVO_VENV}/bin/python" -m pip install -e "${INOVO_ROOT}/1.2.2" \
    && "${INOVO_VENV}/bin/python" -m pip install \
         lightning \
         pytorch-lightning \
         pyopenms \
         s3fs \
    && "${INOVO_VENV}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

WORKDIR /workspace
COPY aichor/generate_instanovo_predictions.py /workspace/aichor/generate_instanovo_predictions.py

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/instanovo-env/bin/python", "-u", "/workspace/aichor/generate_instanovo_predictions.py"]
