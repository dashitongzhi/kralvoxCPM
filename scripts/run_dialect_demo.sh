#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8808}"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif [[ -x "/root/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/root/miniconda3/bin/python"
elif [[ -x "/opt/conda/bin/python" ]]; then
  PYTHON_BIN="/opt/conda/bin/python"
else
  PYTHON_BIN="python"
fi

die() {
  printf '\n[run_dialect_demo] ERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[run_dialect_demo] %s\n' "$*"
}

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    info "Loading environment from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  else
    info "No .env found at $ENV_FILE; using current shell environment"
  fi
}

configure_data_paths() {
  if [[ -z "${DATA_ROOT:-}" ]]; then
    return
  fi

  DATA_ROOT="${DATA_ROOT%/}"
  [[ -n "$DATA_ROOT" ]] || die "DATA_ROOT is empty after normalization. Set it to the remote data disk mount point, for example /root/autodl-tmp."
  export DATA_ROOT

  if [[ ! -d "$DATA_ROOT" ]]; then
    die "DATA_ROOT mount directory does not exist: $DATA_ROOT. The remote cloud/data disk is not mounted, or DATA_ROOT points to the wrong mount path."
  fi

  local missing_subdirs=()
  local required_subdirs=(
    "models"
    "cache"
  )

  for subdir in "${required_subdirs[@]}"; do
    [[ -d "$DATA_ROOT/$subdir" ]] || missing_subdirs+=("$DATA_ROOT/$subdir")
  done

  if ((${#missing_subdirs[@]} > 0)); then
    printf '\n[run_dialect_demo] ERROR: DATA_ROOT is reachable, but its required demo subdirectories are missing: %s\n' "$DATA_ROOT" >&2
    printf '[run_dialect_demo] Missing directories:\n' >&2
    printf '  - %s\n' "${missing_subdirs[@]}" >&2
    printf '[run_dialect_demo] This usually means the cloud disk is mounted but not prepared for the 8808 demo. Create the directories on the mounted disk, or fix DATA_ROOT if this is not the expected disk.\n' >&2
    exit 1
  fi

  : "${VOXCPM_MODEL_PATH:=$DATA_ROOT/models/VoxCPM2}"
  : "${HF_HOME:=$DATA_ROOT/cache/hf}"
  : "${MODELSCOPE_CACHE:=$DATA_ROOT/cache/modelscope}"
  : "${TORCH_HOME:=$DATA_ROOT/cache/torch}"

  export VOXCPM_MODEL_PATH HF_HOME MODELSCOPE_CACHE TORCH_HOME
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

check_port() {
  if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    die "PORT must be a number, got: $PORT"
  fi
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "Port $PORT is already in use. Stop the existing process or start with PORT=<other-port>."
  fi
}

check_required_env() {
  [[ -n "${KRALAPI_API_KEY:-}" ]] || die "KRALAPI_API_KEY is required for dialect rewrite. Set it in .env or export it before running."
  [[ -n "${KRALAPI_BASE_URL:-}" ]] || die "KRALAPI_BASE_URL is required. Example: KRALAPI_BASE_URL=https://kralapi.kralai.tech"
  [[ -n "${KRALAPI_MODEL:-}" ]] || die "KRALAPI_MODEL is required. Example: KRALAPI_MODEL=gpt-5.5"
  [[ -n "${VOXCPM_MODEL_PATH:-}" ]] || die "VOXCPM_MODEL_PATH is required. For remote data disks, set DATA_ROOT=/path/to/data-disk so the script can use DATA_ROOT/models/VoxCPM2."
}

check_model_path() {
  local model_path="$VOXCPM_MODEL_PATH"
  if [[ ! -d "$model_path" ]]; then
    if [[ -n "${DATA_ROOT:-}" && "$model_path" == "$DATA_ROOT"/models/* ]]; then
      die "Model directory is missing under DATA_ROOT: $model_path. The data disk mount is reachable, but the VoxCPM2 model files are not present. Put the model in $DATA_ROOT/models/VoxCPM2 or set VOXCPM_MODEL_PATH to the actual model directory."
    fi
    die "VOXCPM_MODEL_PATH does not exist or is not a directory: $model_path. If this path is on a cloud/data disk, check that the disk is mounted; otherwise place the VoxCPM2 model files there."
  fi

  local missing=()
  local required_files=(
    "config.json"
    "audiovae.pth"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "tokenization_voxcpm2.py"
  )

  for file in "${required_files[@]}"; do
    [[ -f "$model_path/$file" ]] || missing+=("$file")
  done

  shopt -s nullglob
  local weight_files=("$model_path"/*.safetensors "$model_path"/*.bin)
  shopt -u nullglob
  ((${#weight_files[@]} > 0)) || missing+=("model.safetensors or another *.safetensors/*.bin weight file")

  if ((${#missing[@]} > 0)); then
    printf '\n[run_dialect_demo] ERROR: Model directory is incomplete: %s\n' "$model_path" >&2
    if [[ -n "${DATA_ROOT:-}" && "$model_path" == "$DATA_ROOT"/models/* ]]; then
      printf '[run_dialect_demo] DATA_ROOT is mounted, but the VoxCPM2 model files are incomplete.\n' >&2
    fi
    printf '[run_dialect_demo] Missing:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

check_python_deps() {
  "$PYTHON_BIN" - <<'PY' || die "Python dependencies are not ready. Activate the correct env and run: pip install -e ."
import importlib

for module in ("gradio", "funasr", "voxcpm"):
    importlib.import_module(module)
PY
}

check_device() {
  if [[ "$DEVICE" == cuda* ]]; then
    "$PYTHON_BIN" - <<'PY' || die "DEVICE is set to cuda, but torch cannot see CUDA. Use DEVICE=auto or DEVICE=cpu, or fix the CUDA runtime."
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  fi
}

main() {
  cd "$ROOT_DIR"
  load_env
  configure_data_paths
  require_command "$PYTHON_BIN"
  check_port
  check_required_env
  check_model_path
  check_python_deps
  check_device

  info "Starting VoxCPM dialect TTS demo"
  info "URL: http://127.0.0.1:$PORT"
  info "Bind: $HOST:$PORT"
  info "Model: $VOXCPM_MODEL_PATH"
  [[ -z "${DATA_ROOT:-}" ]] || info "Data root: $DATA_ROOT"

  exec "$PYTHON_BIN" "$ROOT_DIR/app.py" \
    --model-id "$VOXCPM_MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"
}

main "$@"
