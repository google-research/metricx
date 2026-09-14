#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/llm-align/liuchonghan/hf_cache}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-$(find "$HF_CACHE_ROOT/models--google--metricx-24-hybrid-xxl-v2p6/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:-$(find "$HF_CACHE_ROOT/models--google--mt5-xl/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)}"

OUTPUT_DIR="${OUTPUT_DIR:-/llm-align/liuchonghan/metricx_result}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-1024}"
NUM_GPUS="${NUM_GPUS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_BF16="${USE_BF16:-1}"

INPUT_DIR_1="${INPUT_DIR_1:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_1}"
INPUT_DIR_2="${INPUT_DIR_2:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_2_t0.9}"
INPUT_DIR_3="${INPUT_DIR_3:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_3_t0.9}"
INPUT_DIR_4="${INPUT_DIR_4:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_4_t0.9}"
INPUT_DIR_5="${INPUT_DIR_5:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_5_t0.9}"

if [[ -z "$MODEL_SNAPSHOT" || ! -d "$MODEL_SNAPSHOT" ]]; then
  echo "MetricX model snapshot not found under $HF_CACHE_ROOT" >&2
  exit 1
fi

if [[ -z "$TOKENIZER_SNAPSHOT" || ! -d "$TOKENIZER_SNAPSHOT" ]]; then
  echo "mT5 tokenizer snapshot not found under $HF_CACHE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

cd "$REPO_DIR"

echo "repo_dir=$REPO_DIR"
echo "model_snapshot=$MODEL_SNAPSHOT"
echo "tokenizer_snapshot=$TOKENIZER_SNAPSHOT"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_input_length=$MAX_INPUT_LENGTH"
echo "num_gpus=$NUM_GPUS"
echo "preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS"
echo "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
echo "use_bf16=$USE_BF16"
echo "input_dirs:"
echo "  $INPUT_DIR_1"
echo "  $INPUT_DIR_2"
echo "  $INPUT_DIR_3"
echo "  $INPUT_DIR_4"
echo "  $INPUT_DIR_5"

EXTRA_ARGS=()
if [[ "$USE_BF16" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi

PIDS=()
for ((SHARD_INDEX=0; SHARD_INDEX<NUM_GPUS; SHARD_INDEX++)); do
  LOG_FILE="$OUTPUT_DIR/logs/shard_${SHARD_INDEX}.log"
  echo "starting shard=$SHARD_INDEX log_file=$LOG_FILE"
  CUDA_VISIBLE_DEVICES="$SHARD_INDEX" python3 -m metricx24.score_dialog_outputs \
    --tokenizer "$TOKENIZER_SNAPSHOT" \
    --model_name_or_path "$MODEL_SNAPSHOT" \
    --input_dirs "$INPUT_DIR_1" "$INPUT_DIR_2" "$INPUT_DIR_3" "$INPUT_DIR_4" "$INPUT_DIR_5" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_input_length "$MAX_INPUT_LENGTH" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --num_shards "$NUM_GPUS" \
    --shard_index "$SHARD_INDEX" \
    "${EXTRA_ARGS[@]}" \
    --qe > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "$PID"
done

python3 -m metricx24.score_dialog_outputs \
  --output_dir "$OUTPUT_DIR" \
  --num_shards "$NUM_GPUS" \
  --merge_only

echo "summary_file=$OUTPUT_DIR/summary_by_dir.json"
