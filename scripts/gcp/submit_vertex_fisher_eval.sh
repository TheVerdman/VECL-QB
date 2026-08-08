#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set BUCKET to an existing gs:// bucket}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
VECL_FISHER_MODEL_ID="${VECL_FISHER_MODEL_ID:-google/gemma-4-31B-it}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the Vertex Fisher eval."
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecl-qb-job.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
PKG_DIR="$WORK_DIR/package"
PACKAGE_TGZ="$WORK_DIR/vecl-qb-fisher-eval-${JOB_TS}.tar.gz"
CONFIG_YAML="$WORK_DIR/job.yaml"
DISPLAY_NAME="vecl-qb-gemma-fisher-eval-${JOB_TS}"
GCS_OUTPUT_URI="${BUCKET}/fisher-eval/${JOB_TS}"
FISHER_BASELINE_SNAPSHOT_URI="${VECL_FISHER_BASELINE_SNAPSHOT_URI:-}"
FISHER_BASELINE_SNAPSHOT_HASH="${VECL_FISHER_BASELINE_SNAPSHOT_HASH:-}"
OPTIONAL_BASELINE_ENV=""
if [[ -n "$FISHER_BASELINE_SNAPSHOT_URI" ]]; then
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}    - name: VECL_FISHER_BASELINE_SNAPSHOT_URI"$'\n'
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}      value: \"${FISHER_BASELINE_SNAPSHOT_URI}\""$'\n'
fi
if [[ -n "$FISHER_BASELINE_SNAPSHOT_HASH" ]]; then
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}    - name: VECL_FISHER_BASELINE_SNAPSHOT_HASH"$'\n'
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}      value: \"${FISHER_BASELINE_SNAPSHOT_HASH}\""$'\n'
fi

mkdir -p "$PKG_DIR/fisher_eval_job"
cp -R vecl "$PKG_DIR/vecl"
cp scripts/fisher_eval_gemma.py "$PKG_DIR/fisher_eval_gemma.py"
touch "$PKG_DIR/fisher_eval_job/__init__.py"

cat > "$PKG_DIR/fisher_eval_job/__main__.py" <<'PY'
from __future__ import annotations

from fisher_eval_gemma import main

raise SystemExit(main())
PY

cat > "$PKG_DIR/sitecustomize.py" <<'PY'
import datetime
import enum

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return self.value

    enum.StrEnum = StrEnum
PY

cat > "$PKG_DIR/setup.py" <<'PY'
from setuptools import find_packages, setup

setup(
    name="vecl-qb-fisher-eval",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["sitecustomize", "fisher_eval_gemma"],
    install_requires=[
        "networkx>=3.3",
        "numpy>=1.26.4,<2",
        "pydantic>=2.7",
        "PyYAML>=6.0",
        "sympy>=1.13",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=5.0",
        "accelerate>=1.0",
        "peft>=0.14",
        "sentencepiece>=0.2",
        "huggingface_hub>=0.27",
        "pillow>=10",
        "safetensors>=0.5.3",
        "python-json-logger>=3.2",
    ],
)
PY

tar -C "$PKG_DIR" -czf "$PACKAGE_TGZ" .
gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null
gcloud storage cp "$PACKAGE_TGZ" "$BUCKET/packages/" --project="$PROJECT_ID"

PACKAGE_URI="$BUCKET/packages/vecl-qb-fisher-eval-${JOB_TS}.tar.gz"

cat > "$CONFIG_YAML" <<EOF
workerPoolSpecs:
- machineSpec:
    machineType: a2-ultragpu-1g
    acceleratorType: NVIDIA_A100_80GB
    acceleratorCount: 1
  replicaCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: 500
  pythonPackageSpec:
    executorImageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
    packageUris:
    - ${PACKAGE_URI}
    pythonModule: fisher_eval_job
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: VECL_FISHER_MODEL_ID
      value: "${VECL_FISHER_MODEL_ID}"
    - name: GEMMA_DTYPE
      value: "${GEMMA_DTYPE:-bfloat16}"
    - name: VECL_FISHER_TORCH_SEED
      value: "${VECL_FISHER_TORCH_SEED:-1234}"
    - name: VECL_FISHER_LORA_RANK
      value: "${VECL_FISHER_LORA_RANK:-8}"
    - name: VECL_FISHER_LAST_N_LAYERS
      value: "${VECL_FISHER_LAST_N_LAYERS:-8}"
    - name: VECL_FISHER_DRIFT_THRESHOLD
      value: "${VECL_FISHER_DRIFT_THRESHOLD:-1.0}"
${OPTIONAL_BASELINE_ENV}    - name: VECL_FISHER_GCS_OUTPUT_URI
      value: "${GCS_OUTPUT_URI}"
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: PIP_ROOT_USER_ACTION
      value: "ignore"
scheduling:
  timeout: 10800s
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
print "Fisher outputs: ${GCS_OUTPUT_URI}"
print "Temporary credential-bearing config removed after submission."
print "Unset HF_TOKEN when no longer needed."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
