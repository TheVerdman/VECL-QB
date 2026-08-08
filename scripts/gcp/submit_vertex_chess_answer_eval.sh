#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set BUCKET to an existing gs:// bucket}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
VECL_ROUTING_MODEL_ID="${VECL_ROUTING_MODEL_ID:-google/gemma-4-31B-it}"
VECL_ROUTING_MAX_NEW_TOKENS="${VECL_ROUTING_MAX_NEW_TOKENS:-256}"
VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES="${VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES:-16}"
VECL_CHESS_ANSWER_MIN_RATE="${VECL_CHESS_ANSWER_MIN_RATE:-0.8}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the Vertex chess answer eval."
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecl-qb-job.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
PKG_DIR="$WORK_DIR/package"
PACKAGE_TGZ="$WORK_DIR/vecl-qb-chess-answer-eval-${JOB_TS}.tar.gz"
CONFIG_YAML="$WORK_DIR/job.yaml"
DISPLAY_NAME="vecl-qb-gemma-chess-answer-eval-${JOB_TS}"

mkdir -p "$PKG_DIR/chess_answer_eval_job" "$PKG_DIR/scripts"
cp -R vecl "$PKG_DIR/vecl"
cp scripts/smoke_gemma.py "$PKG_DIR/scripts/smoke_gemma.py"
cp scripts/routing_eval_gemma_stockfish.py "$PKG_DIR/scripts/routing_eval_gemma_stockfish.py"
cp scripts/chain_eval_gemma_stockfish.py "$PKG_DIR/scripts/chain_eval_gemma_stockfish.py"
cp scripts/chess_answer_eval_gemma_stockfish.py "$PKG_DIR/chess_answer_eval_job/__main__.py"
touch "$PKG_DIR/scripts/__init__.py"
touch "$PKG_DIR/chess_answer_eval_job/__init__.py"
cp tests/qb/fixtures/chess_answer_eval_v0.json "$PKG_DIR/chess_answer_eval_job/chess_answer_eval_v0.json"

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
    name="vecl-qb-chess-answer-eval",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["sitecustomize"],
    package_data={"chess_answer_eval_job": ["chess_answer_eval_v0.json"]},
    install_requires=[
        "networkx>=3.3",
        "numpy>=1.25,<1.28",
        "pydantic>=2.7",
        "PyYAML>=6.0",
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

PACKAGE_URI="$BUCKET/packages/vecl-qb-chess-answer-eval-${JOB_TS}.tar.gz"

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
    pythonModule: chess_answer_eval_job
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: VECL_ROUTING_MODEL_ID
      value: "${VECL_ROUTING_MODEL_ID}"
    - name: VECL_ROUTING_MAX_NEW_TOKENS
      value: "${VECL_ROUTING_MAX_NEW_TOKENS}"
    - name: VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES
      value: "${VECL_CHESS_ANSWER_EVAL_MAX_ENTRIES}"
    - name: VECL_CHESS_ANSWER_MIN_RATE
      value: "${VECL_CHESS_ANSWER_MIN_RATE}"
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
