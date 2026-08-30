ARG UNSLOTH_IMAGE=unsloth/unsloth:latest
FROM ${UNSLOTH_IMAGE}

ARG VCS_REF
ARG UNSLOTH_VERSION=2026.8.22
ARG UNSLOTH_ZOO_VERSION=2026.8.16
LABEL org.opencontainers.image.source="https://github.com/mikaelkanzaki-git/tech-fine-tuning"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONPATH=/workspace/tech-fine-tuning/src \
    HF_HOME=/workspace/cache/huggingface \
    TECH_FINE_TUNING_REVISION=${VCS_REF}

WORKDIR /workspace/tech-fine-tuning
RUN python -m pip install --no-cache-dir --no-deps \
    "unsloth==${UNSLOTH_VERSION}" \
    "unsloth-zoo==${UNSLOTH_ZOO_VERSION}"
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY schemas ./schemas
COPY src ./src

CMD ["python", "-m", "tech_fine_tuning", "--help"]
