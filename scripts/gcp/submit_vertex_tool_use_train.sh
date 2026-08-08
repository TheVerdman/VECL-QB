#!/usr/bin/env zsh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set BUCKET to an existing gs:// bucket}"
JOB_TS="$(date +%Y%m%d-%H%M%S)"
VECL_TRAIN_MODEL_ID="${VECL_TRAIN_MODEL_ID:-google/gemma-4-31B-it}"
STREAM_LOGS="${STREAM_LOGS:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  print -u2 "HF_TOKEN is required. Export it in your shell before launching the tool-use training eval."
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecl-qb-job.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
PKG_DIR="$WORK_DIR/package"
PACKAGE_TGZ="$WORK_DIR/vecl-qb-tool-use-train-${JOB_TS}.tar.gz"
CONFIG_YAML="$WORK_DIR/job.yaml"
DISPLAY_NAME="vecl-qb-gemma-tool-use-train-${JOB_TS}"
GCS_OUTPUT_URI="${BUCKET}/tool-use-training/${JOB_TS}"
LOCAL_CORPUS_EXPORT_DIR="${VECL_TRAIN_CORPUS_EXPORT_DIR:-data/synthetic/v1-hard/exports}"
GCS_CORPUS_EXPORT_URI="${BUCKET}/tool-use-training-corpora/${JOB_TS}/exports"
TRAIN_DOMAIN_MIX="${VECL_TRAIN_DOMAIN_MIX:-__all__}"
BASELINE_SNAPSHOT_URI="${VECL_TRAIN_BASELINE_SNAPSHOT_URI:-}"
BASELINE_SNAPSHOT_HASH="${VECL_TRAIN_BASELINE_SNAPSHOT_HASH:-}"
OPTIONAL_BASELINE_ENV=""
if [[ -n "$BASELINE_SNAPSHOT_URI" ]]; then
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}    - name: VECL_TRAIN_BASELINE_SNAPSHOT_URI"$'\n'
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}      value: \"${BASELINE_SNAPSHOT_URI}\""$'\n'
fi
if [[ -n "$BASELINE_SNAPSHOT_HASH" ]]; then
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}    - name: VECL_TRAIN_BASELINE_SNAPSHOT_HASH"$'\n'
  OPTIONAL_BASELINE_ENV="${OPTIONAL_BASELINE_ENV}      value: \"${BASELINE_SNAPSHOT_HASH}\""$'\n'
fi
EWC_APPROVED_SNAPSHOT_URI="${VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI:-}"
EWC_APPROVED_SNAPSHOT_HASH="${VECL_TRAIN_EWC_APPROVED_SNAPSHOT_HASH:-}"
OPTIONAL_EWC_APPROVED_ENV=""
if [[ -n "$EWC_APPROVED_SNAPSHOT_URI" ]]; then
  OPTIONAL_EWC_APPROVED_ENV="${OPTIONAL_EWC_APPROVED_ENV}    - name: VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI"$'\n'
  OPTIONAL_EWC_APPROVED_ENV="${OPTIONAL_EWC_APPROVED_ENV}      value: \"${EWC_APPROVED_SNAPSHOT_URI}\""$'\n'
fi
if [[ -n "$EWC_APPROVED_SNAPSHOT_HASH" ]]; then
  OPTIONAL_EWC_APPROVED_ENV="${OPTIONAL_EWC_APPROVED_ENV}    - name: VECL_TRAIN_EWC_APPROVED_SNAPSHOT_HASH"$'\n'
  OPTIONAL_EWC_APPROVED_ENV="${OPTIONAL_EWC_APPROVED_ENV}      value: \"${EWC_APPROVED_SNAPSHOT_HASH}\""$'\n'
fi
if [[ -n "${VECL_TRAIN_EXPECT_DOMAINS:-}" ]]; then
  TRAIN_EXPECT_DOMAINS="${VECL_TRAIN_EXPECT_DOMAINS}"
elif [[ "$TRAIN_DOMAIN_MIX" == "__all__" || "$TRAIN_DOMAIN_MIX" == "all" || "$TRAIN_DOMAIN_MIX" == "*" ]]; then
  TRAIN_EXPECT_DOMAINS="blast,cross,eda,stockfish,sympy,terraform,timesfm"
else
  TRAIN_EXPECT_DOMAINS="$TRAIN_DOMAIN_MIX"
fi

if [[ -d "$LOCAL_CORPUS_EXPORT_DIR" ]]; then
  gcloud storage rsync "$LOCAL_CORPUS_EXPORT_DIR" "$GCS_CORPUS_EXPORT_URI" --project="$PROJECT_ID" >/dev/null
else
  print -u2 "Corpus export dir not found: ${LOCAL_CORPUS_EXPORT_DIR}."
  if [[ "${VECL_TRAIN_REQUIRE_CORPUS:-1}" == "1" ]]; then
    print -u2 "Set VECL_TRAIN_REQUIRE_CORPUS=0 to allow Stockfish fallback examples."
    exit 2
  fi
  print -u2 "Vertex job will use Stockfish fallback examples."
  GCS_CORPUS_EXPORT_URI=""
fi

mkdir -p "$PKG_DIR/tool_use_train_job"
cp -R vecl "$PKG_DIR/vecl"
cp scripts/tool_use_train_gemma.py "$PKG_DIR/tool_use_train_gemma.py"
touch "$PKG_DIR/tool_use_train_job/__init__.py"

cat > "$PKG_DIR/tool_use_train_job/__main__.py" <<'PY'
from __future__ import annotations

from tool_use_train_gemma import main

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
    name="vecl-qb-tool-use-train",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["sitecustomize", "tool_use_train_gemma"],
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

PACKAGE_URI="$BUCKET/packages/vecl-qb-tool-use-train-${JOB_TS}.tar.gz"

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
    pythonModule: tool_use_train_job
    env:
    - name: HF_TOKEN
      value: "${HF_TOKEN}"
    - name: VECL_TRAIN_MODEL_ID
      value: "${VECL_TRAIN_MODEL_ID}"
    - name: GEMMA_DTYPE
      value: "${GEMMA_DTYPE:-bfloat16}"
    - name: VECL_TRAIN_LORA_RANK
      value: "${VECL_TRAIN_LORA_RANK:-8}"
    - name: VECL_TRAIN_LAST_N_LAYERS
      value: "${VECL_TRAIN_LAST_N_LAYERS:-8}"
    - name: VECL_TRAIN_MAX_SLOTS
      value: "${VECL_TRAIN_MAX_SLOTS:-8}"
    - name: VECL_TRAIN_LEARNING_RATE
      value: "${VECL_TRAIN_LEARNING_RATE:-0.001}"
    - name: VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS
      value: "${VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS:-8}"
    - name: VECL_TRAIN_PROMPT_MODE
      value: "${VECL_TRAIN_PROMPT_MODE:-tool_call_author}"
    - name: VECL_TRAIN_DIAGNOSTIC_MODE
      value: "${VECL_TRAIN_DIAGNOSTIC_MODE:-}"
    - name: VECL_TRAIN_OVERFIT_SAMPLE_COUNT
      value: "${VECL_TRAIN_OVERFIT_SAMPLE_COUNT:-8}"
    - name: VECL_TRAIN_OVERFIT_TASK_KIND
      value: "${VECL_TRAIN_OVERFIT_TASK_KIND:-tool_call_json}"
    - name: VECL_TRAIN_OVERFIT_STEPS
      value: "${VECL_TRAIN_OVERFIT_STEPS:-25}"
    - name: VECL_TRAIN_OVERFIT_LEARNING_RATE
      value: "${VECL_TRAIN_OVERFIT_LEARNING_RATE:-0.01}"
    - name: VECL_TRAIN_OVERFIT_MIN_CE_DROP
      value: "${VECL_TRAIN_OVERFIT_MIN_CE_DROP:-0.05}"
    - name: VECL_TRAIN_OVERFIT_GENERATION_PROBE_COUNT
      value: "${VECL_TRAIN_OVERFIT_GENERATION_PROBE_COUNT:-0}"
    - name: VECL_TRAIN_SPARSE_OVERFIT_CYCLES
      value: "${VECL_TRAIN_SPARSE_OVERFIT_CYCLES:-25}"
    - name: VECL_TRAIN_SPARSE_OVERFIT_LEARNING_RATE
      value: "${VECL_TRAIN_SPARSE_OVERFIT_LEARNING_RATE:-0.01}"
    - name: VECL_TRAIN_SPARSE_OVERFIT_MAX_SLOTS
      value: "${VECL_TRAIN_SPARSE_OVERFIT_MAX_SLOTS:-${VECL_TRAIN_MAX_SLOTS:-8}}"
    - name: VECL_TRAIN_SPARSE_OVERFIT_GRADIENT_ACCUMULATION_STEPS
      value: "${VECL_TRAIN_SPARSE_OVERFIT_GRADIENT_ACCUMULATION_STEPS:-${VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS:-8}}"
    - name: VECL_TRAIN_SPARSE_OVERFIT_MIN_CE_DROP
      value: "${VECL_TRAIN_SPARSE_OVERFIT_MIN_CE_DROP:-0.05}"
    - name: VECL_TRAIN_EWC_LAMBDA
      value: "${VECL_TRAIN_EWC_LAMBDA:-0.0}"
    - name: VECL_TRAIN_EWC_DRIFT_THRESHOLD
      value: "${VECL_TRAIN_EWC_DRIFT_THRESHOLD:-1.0}"
    - name: VECL_TRAIN_BASELINE_ID
      value: "${VECL_TRAIN_BASELINE_ID:-tool_use_v0}"
${OPTIONAL_BASELINE_ENV}${OPTIONAL_EWC_APPROVED_ENV}    - name: VECL_TRAIN_CORPUS_EXPORT_URI
      value: "${GCS_CORPUS_EXPORT_URI}"
    - name: VECL_TRAIN_SAMPLE_COUNT
      value: "${VECL_TRAIN_SAMPLE_COUNT:-1024}"
    - name: VECL_TRAIN_SAMPLE_SEED
      value: "${VECL_TRAIN_SAMPLE_SEED:-1107}"
    - name: VECL_TRAIN_TORCH_SEED
      value: "${VECL_TRAIN_TORCH_SEED:-1234}"
    - name: VECL_TRAIN_SMOKE_PROBE_COUNT
      value: "${VECL_TRAIN_SMOKE_PROBE_COUNT:-8}"
    - name: VECL_TRAIN_HELDOUT_PROBE_COUNT
      value: "${VECL_TRAIN_HELDOUT_PROBE_COUNT:-48}"
    - name: VECL_TRAIN_HARD_PROBE_COUNT
      value: "${VECL_TRAIN_HARD_PROBE_COUNT:-48}"
    - name: VECL_TRAIN_DOMAIN_MIX
      value: "${TRAIN_DOMAIN_MIX}"
    - name: VECL_TRAIN_EXPECT_DOMAINS
      value: "${TRAIN_EXPECT_DOMAINS}"
    - name: VECL_TRAIN_INTERFERENCE_THRESHOLD
      value: "${VECL_TRAIN_INTERFERENCE_THRESHOLD:-0.05}"
    - name: VECL_TRAIN_EXPECT_REJECTION
      value: "${VECL_TRAIN_EXPECT_REJECTION:-0}"
    - name: VECL_TRAIN_REQUIRE_INLINE_DRIFT
      value: "${VECL_TRAIN_REQUIRE_INLINE_DRIFT:-0}"
    - name: VECL_TRAIN_TOOL_CALL_GENERATION_PROBE_COUNT
      value: "${VECL_TRAIN_TOOL_CALL_GENERATION_PROBE_COUNT:-0}"
    - name: VECL_TRAIN_TOOL_CALL_GENERATION_MAX_NEW_TOKENS
      value: "${VECL_TRAIN_TOOL_CALL_GENERATION_MAX_NEW_TOKENS:-256}"
    - name: VECL_TRAIN_GCS_OUTPUT_URI
      value: "${GCS_OUTPUT_URI}"
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: PIP_ROOT_USER_ACTION
      value: "ignore"
    - name: PYTORCH_CUDA_ALLOC_CONF
      value: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
print "Training outputs: ${GCS_OUTPUT_URI}"
print "Temporary credential-bearing config removed after submission."
print "Unset HF_TOKEN when no longer needed."

if [[ "$STREAM_LOGS" == "true" ]]; then
  gcloud ai custom-jobs stream-logs "$JOB_ID" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --polling-interval=15
fi
