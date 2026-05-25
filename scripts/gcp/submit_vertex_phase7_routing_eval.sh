#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-49b1b523-d248-434f-bd4}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-vecl-qb-artifacts}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
VECL_ROUTING_MODEL_ID="${VECL_ROUTING_MODEL_ID:-google/gemma-4-31B-it}"
VECL_ROUTING_MAX_NEW_TOKENS="${VECL_ROUTING_MAX_NEW_TOKENS:-128}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the Phase 7 Vertex routing eval."
  exit 2
fi

PKG_DIR="/tmp/vecl-qb-phase7-routing-eval-${JOB_TS}"
PACKAGE_TGZ="/tmp/vecl-qb-phase7-routing-eval-${JOB_TS}.tar.gz"
CONFIG_YAML="/tmp/vecl-qb-phase7-routing-eval-${JOB_TS}.yaml"
DISPLAY_NAME="vecl-qb-gemma-phase7-routing-eval-${JOB_TS}"

mkdir -p "$PKG_DIR/phase7_routing_eval_job"
cp -R vecl "$PKG_DIR/vecl"
cp scripts/routing_eval_gemma_phase7.py "$PKG_DIR/phase7_routing_eval_job/__main__.py"
touch "$PKG_DIR/phase7_routing_eval_job/__init__.py"
cp tests/qb/fixtures/phase7_routing_eval_v0.json "$PKG_DIR/phase7_routing_eval_job/phase7_routing_eval_v0.json"

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
    name="vecl-qb-phase7-routing-eval",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["sitecustomize"],
    package_data={"phase7_routing_eval_job": ["phase7_routing_eval_v0.json"]},
    install_requires=[
        "networkx>=3.3",
        "numpy>=1.25,<1.28",
        "pydantic>=2.7",
        "PyYAML>=6.0",
        "sympy>=1.13",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=5.0",
        "accelerate>=1.0",
        "sentencepiece>=0.2",
        "huggingface_hub>=0.27",
        "pillow>=10",
        "safetensors>=0.4",
        "python-json-logger>=3.2",
        "tomli>=2.0; python_version < '3.11'",
    ],
)
PY

tar -C "$PKG_DIR" -czf "$PACKAGE_TGZ" .
gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null
gcloud storage cp "$PACKAGE_TGZ" "$BUCKET/packages/" --project="$PROJECT_ID"

PACKAGE_URI="$BUCKET/packages/vecl-qb-phase7-routing-eval-${JOB_TS}.tar.gz"

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
    pythonModule: phase7_routing_eval_job
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: VECL_ROUTING_MODEL_ID
      value: "${VECL_ROUTING_MODEL_ID}"
    - name: VECL_ROUTING_MAX_NEW_TOKENS
      value: "${VECL_ROUTING_MAX_NEW_TOKENS}"
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: PIP_ROOT_USER_ACTION
      value: "ignore"
scheduling:
  timeout: 7200s
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

JOB_NAME="$(gcloud ai custom-jobs list \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --filter="displayName=${DISPLAY_NAME}" \
  --sort-by=~createTime \
  --limit=1 \
  --format='value(name)')"
JOB_ID="${JOB_NAME##*/}"

print "Vertex custom job id: ${JOB_ID}"
print "Temporary config with HF_TOKEN: ${CONFIG_YAML}"
print "After the job starts, delete the temporary config and unset HF_TOKEN."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
