#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set BUCKET to an existing gs:// bucket}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
VECL_ROUTING_MODEL_ID="${VECL_ROUTING_MODEL_ID:-google/gemma-4-31B-it}"
VECL_ROUTING_MAX_NEW_TOKENS="${VECL_ROUTING_MAX_NEW_TOKENS:-512}"
VECL_TERRAFORM_VERSION="${VECL_TERRAFORM_VERSION:-1.15.4}"
VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES="${VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES:-6}"
TIMESFM_MODEL_ID="${TIMESFM_MODEL_ID:-google/timesfm-2.5-200m-pytorch}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the Vertex release eval."
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecl-qb-job.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
PKG_DIR="$WORK_DIR/package"
PACKAGE_TGZ="$WORK_DIR/vecl-qb-release-eval-${JOB_TS}.tar.gz"
CONFIG_YAML="$WORK_DIR/job.yaml"
DISPLAY_NAME="vecl-qb-gemma-release-eval-${JOB_TS}"

mkdir -p "$PKG_DIR/release_eval_job"
cp -R vecl "$PKG_DIR/vecl"
cp configs/release/phase9a-vertex-gemma.json "$PKG_DIR/release_eval_job/phase9a-vertex-gemma.json"
cp tests/qb/fixtures/model_driver_eval_v0.json "$PKG_DIR/release_eval_job/model_driver_eval_v0.json"
cp tests/qb/fixtures/tool_call_payload_eval_v0.json "$PKG_DIR/release_eval_job/tool_call_payload_eval_v0.json"
cp tests/qb/fixtures/phase7_routing_eval_v0.json "$PKG_DIR/release_eval_job/phase7_routing_eval_v0.json"
cp tests/qb/fixtures/terraform_stockfish_eval_v0.json "$PKG_DIR/release_eval_job/terraform_stockfish_eval_v0.json"
cp tests/qb/fixtures/timesfm_demand_eval_v0.json "$PKG_DIR/release_eval_job/timesfm_demand_eval_v0.json"
cp scripts/model_driver_eval.py "$PKG_DIR/model_driver_eval.py"
cp scripts/tool_call_payload_eval.py "$PKG_DIR/tool_call_payload_eval.py"
cp scripts/routing_eval_gemma_phase7.py "$PKG_DIR/routing_eval_gemma_phase7.py"
cp scripts/ethics_eval_gemma.py "$PKG_DIR/ethics_eval_gemma.py"
cp scripts/terraform_eval_gemma.py "$PKG_DIR/terraform_eval_gemma.py"
cp scripts/terraform_stockfish_eval_gemma.py "$PKG_DIR/terraform_stockfish_eval_gemma.py"
cp scripts/routing_eval_gemma_stockfish.py "$PKG_DIR/routing_eval_gemma_stockfish.py"
cp scripts/timesfm_demand_eval_gemma.py "$PKG_DIR/timesfm_demand_eval_gemma.py"
cp scripts/release_eval_gemma_aggregate.py "$PKG_DIR/release_eval_gemma_aggregate.py"
touch "$PKG_DIR/release_eval_job/__init__.py"

cat > "$PKG_DIR/release_eval_job/__main__.py" <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from release_eval_gemma_aggregate import main
from vecl._paths import environment_directory

base = Path(__file__).resolve().parent
artifact_root = environment_directory(
    "VECL_RELEASE_EVAL_ARTIFACT_ROOT", prefix="vecl-release-eval-artifacts-"
)

os.environ.setdefault("VECL_MODEL_DRIVER", "gemma")

raise SystemExit(main(base_dir=base, artifact_root=artifact_root))
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
    name="vecl-qb-release-eval",
    version="0.0.0",
    packages=find_packages(),
    py_modules=[
        "sitecustomize",
        "model_driver_eval",
        "tool_call_payload_eval",
        "routing_eval_gemma_phase7",
        "ethics_eval_gemma",
        "terraform_eval_gemma",
        "terraform_stockfish_eval_gemma",
        "routing_eval_gemma_stockfish",
        "timesfm_demand_eval_gemma",
        "release_eval_gemma_aggregate",
    ],
    package_data={
        "release_eval_job": [
            "phase9a-vertex-gemma.json",
            "model_driver_eval_v0.json",
            "tool_call_payload_eval_v0.json",
            "phase7_routing_eval_v0.json",
            "terraform_stockfish_eval_v0.json",
            "timesfm_demand_eval_v0.json",
        ]
    },
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
        "sentencepiece>=0.2",
        "huggingface_hub>=0.27",
        "pillow>=10",
        "safetensors>=0.5.3",
        "python-json-logger>=3.2",
        "timesfm[torch] @ https://github.com/google-research/timesfm/archive/refs/heads/master.zip",
        "tomli>=2.0; python_version < '3.11'",
    ],
)
PY

tar -C "$PKG_DIR" -czf "$PACKAGE_TGZ" .
gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null
gcloud storage cp "$PACKAGE_TGZ" "$BUCKET/packages/" --project="$PROJECT_ID"

PACKAGE_URI="$BUCKET/packages/vecl-qb-release-eval-${JOB_TS}.tar.gz"

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
    pythonModule: release_eval_job
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: VECL_MODEL_DRIVER
      value: "gemma"
    - name: VECL_ROUTING_MODEL_ID
      value: "${VECL_ROUTING_MODEL_ID}"
    - name: VECL_ETHICS_EVAL_MODEL_ID
      value: "${VECL_ROUTING_MODEL_ID}"
    - name: VECL_ROUTING_MAX_NEW_TOKENS
      value: "${VECL_ROUTING_MAX_NEW_TOKENS}"
    - name: VECL_TERRAFORM_VERSION
      value: "${VECL_TERRAFORM_VERSION}"
    - name: VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES
      value: "${VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES}"
    - name: TIMESFM_MODEL_ID
      value: "${TIMESFM_MODEL_ID}"
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
print "Temporary credential-bearing config removed after submission."
print "Unset HF_TOKEN when no longer needed."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
