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
  [[ -n "${VOXCPM_MODEL_PATH:-}" ]] || die "VOXCPM_MODEL_PATH is required. Point it to the local VoxCPM2 model directory."
}

check_model_path() {
  local model_path="$VOXCPM_MODEL_PATH"
  [[ -d "$model_path" ]] || die "VOXCPM_MODEL_PATH does not exist or is not a directory: $model_path"

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

  exec "$PYTHON_BIN" "$ROOT_DIR/app.py" \
    --model-id "$VOXCPM_MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"
}

main "$@"
