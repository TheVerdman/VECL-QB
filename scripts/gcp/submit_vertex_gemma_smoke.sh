#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set BUCKET to an existing gs:// bucket}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-google/gemma-4-E4B-it}"
GEMMA_MAX_NEW_TOKENS="${GEMMA_MAX_NEW_TOKENS:-64}"
GEMMA_ALLOWED_CUDA_DEVICES="${GEMMA_ALLOWED_CUDA_DEVICES:-A100}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the Vertex smoke job."
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecl-qb-job.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
PKG_DIR="$WORK_DIR/package"
PACKAGE_TGZ="$WORK_DIR/vecl-qb-smoke-${JOB_TS}.tar.gz"
CONFIG_YAML="$WORK_DIR/job.yaml"
DISPLAY_NAME="vecl-qb-gemma-e4b-a100-smoke-${JOB_TS}"

mkdir -p "$PKG_DIR/vecl"
cp scripts/smoke_gemma.py "$PKG_DIR/smoke_gemma.py"
cp vecl/__init__.py "$PKG_DIR/vecl/__init__.py"
cp vecl/_gemma.py "$PKG_DIR/vecl/_gemma.py"

cat > "$PKG_DIR/setup.py" <<'PY'
from setuptools import setup

setup(
    name="vecl-qb-smoke",
    version="0.0.0",
    packages=["vecl"],
    py_modules=["smoke_gemma"],
    install_requires=[
        "pydantic>=2.7",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=5.0",
        "accelerate>=1.0",
        "sentencepiece>=0.2",
        "huggingface_hub>=0.27",
        "pillow>=10",
        "safetensors>=0.4",
        "python-json-logger>=3.2",
    ],
)
PY

tar -C "$PKG_DIR" -czf "$PACKAGE_TGZ" setup.py smoke_gemma.py vecl
gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null
gcloud storage cp "$PACKAGE_TGZ" "$BUCKET/packages/" --project="$PROJECT_ID"

PACKAGE_URI="$BUCKET/packages/vecl-qb-smoke-${JOB_TS}.tar.gz"

cat > "$CONFIG_YAML" <<EOF
workerPoolSpecs:
- machineSpec:
    machineType: a2-ultragpu-1g
    acceleratorType: NVIDIA_A100_80GB
    acceleratorCount: 1
  replicaCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: 250
  pythonPackageSpec:
    executorImageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
    packageUris:
    - ${PACKAGE_URI}
    pythonModule: smoke_gemma
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: GEMMA_MODEL_ID
      value: "${GEMMA_MODEL_ID}"
    - name: GEMMA_ALLOWED_CUDA_DEVICES
      value: "${GEMMA_ALLOWED_CUDA_DEVICES}"
    - name: GEMMA_MAX_NEW_TOKENS
      value: "${GEMMA_MAX_NEW_TOKENS}"
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: PIP_ROOT_USER_ACTION
      value: "ignore"
scheduling:
  timeout: 3600s
  disableRetries: true
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs
EOF

chmod 600 "$CONFIG_YAML"

print "Submitting ${DISPLAY_NAME}"
gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="$DISPLAY_NAME" \
  --config="$CONFIG_YAML"

rm -f -- "$CONFIG_YAML"

JOB_NAME="$(gcloud ai custom-jobs list \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --filter="displayName=${DISPLAY_NAME}" \
  --sort-by=~createTime \
  --limit=1 \
  --format='value(name)')"
JOB_ID="${JOB_NAME##*/}"

print "Vertex custom job id: ${JOB_ID}"
print "Temporary credential-bearing config removed after submission."
print "Unset HF_TOKEN when no longer needed."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
