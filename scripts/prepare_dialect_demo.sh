#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_EXAMPLE="${ENV_EXAMPLE:-$ROOT_DIR/.env.example}"

die() {
  printf '\n[prepare_dialect_demo] ERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[prepare_dialect_demo] %s\n' "$*"
}

warn() {
  printf '[prepare_dialect_demo] WARN: %s\n' "$*" >&2
}

ensure_env_file() {
  [[ -f "$ENV_EXAMPLE" ]] || die "Missing env template: $ENV_EXAMPLE"

  if [[ -f "$ENV_FILE" ]]; then
    info "Using existing env file: $ENV_FILE"
    return
  fi

  mkdir -p "$(dirname "$ENV_FILE")"
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  info "Created editable env file from .env.example: $ENV_FILE"
}

load_env() {
  local data_root_override="${DATA_ROOT:-}"

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  if [[ -n "$data_root_override" ]]; then
    DATA_ROOT="$data_root_override"
  fi

  DATA_ROOT="${DATA_ROOT:-}"
  DATA_ROOT="${DATA_ROOT%/}"
  [[ -n "$DATA_ROOT" ]] || die "DATA_ROOT is empty. Set it in $ENV_FILE or run: DATA_ROOT=/root/autodl-tmp scripts/prepare_dialect_demo.sh"
  export DATA_ROOT
}

prepare_data_root() {
  local required_dirs=(
    "models"
    "models/VoxCPM2"
    "cache"
    "cache/hf"
    "cache/modelscope"
    "cache/torch"
  )

  [[ -d "$DATA_ROOT" ]] || die "DATA_ROOT does not exist: $DATA_ROOT. Mount the remote data disk first, or set DATA_ROOT to the mounted path."

  local rel
  for rel in "${required_dirs[@]}"; do
    mkdir -p "$DATA_ROOT/$rel" || die "Cannot create directory: $DATA_ROOT/$rel"
  done

  info "Prepared DATA_ROOT directory structure under: $DATA_ROOT"
  for rel in "${required_dirs[@]}"; do
    printf '  - %s\n' "$DATA_ROOT/$rel"
  done
}

check_local_env() {
  if [[ -z "${KRALAPI_API_KEY:-}" ]]; then
    warn "KRALAPI_API_KEY is empty. Edit $ENV_FILE before starting the demo."
  fi

  if [[ -z "${KRALAPI_BASE_URL:-}" ]]; then
    warn "KRALAPI_BASE_URL is empty. Keep the .env.example default or set the actual API base URL."
  fi

  if [[ -z "${KRALAPI_MODEL:-}" ]]; then
    warn "KRALAPI_MODEL is empty. Keep the .env.example default or set the model name."
  fi
}

check_model_hint() {
  local model_path="${VOXCPM_MODEL_PATH:-$DATA_ROOT/models/VoxCPM2}"
  local required_files=(
    "config.json"
    "audiovae.pth"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "tokenization_voxcpm2.py"
  )
  local missing=()
  local file

  for file in "${required_files[@]}"; do
    [[ -f "$model_path/$file" ]] || missing+=("$file")
  done

  shopt -s nullglob
  local weight_files=("$model_path"/*.safetensors "$model_path"/*.bin)
  shopt -u nullglob
  ((${#weight_files[@]} > 0)) || missing+=("*.safetensors or *.bin weight file")

  if ((${#missing[@]} > 0)); then
    warn "VoxCPM2 model files are not complete yet: $model_path"
    printf '[prepare_dialect_demo] Missing model files:\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
  else
    info "Model directory looks ready: $model_path"
  fi
}

main() {
  cd "$ROOT_DIR"
  ensure_env_file
  load_env
  prepare_data_root
  check_local_env
  check_model_hint

  info "Preparation complete. Edit $ENV_FILE if needed, then run: scripts/run_dialect_demo.sh"
}

main "$@"
