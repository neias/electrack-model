#!/usr/bin/env bash
#
# electrack — PCB komponent tespiti hattı için yardımcı çalıştırıcı.
#
# Kullanım:
#   ./run.sh setup                 # venv oluştur + bağımlılıkları kur
#   ./run.sh test                  # testleri çalıştır (pytest)
#   ./run.sh lint                  # ruff + black --check
#   ./run.sh prepare-data          # veri kümesini doğrula/hazırla
#   ./run.sh train [--epochs N]    # modeli eğit
#   ./run.sh export --weights P    # Core ML'e aktar
#   ./run.sh infer --model M --image I [--json out.json]
#   ./run.sh evaluate --model M [--dataset D] [--measure-latency]
#   ./run.sh validate-output FILE  # tespit çıktısını sözleşmeye göre doğrula
#   ./run.sh cli ...               # electrack.cli'ye ham argüman geçir
#   ./run.sh all                   # setup + lint + test (CI benzeri)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
# Python yorumlayıcı seçimi (venv yoksa): coremltools 3.13 desteklemez → 3.9-3.12 tercih.
PYBIN="${PYBIN:-python3}"

log() { printf '\033[1;34m[run]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[run:hata]\033[0m %s\n' "$*" >&2; }

ensure_venv() {
  if [[ ! -x "$PY" ]]; then
    err "venv yok. Önce: ./run.sh setup"
    exit 1
  fi
}

cmd_setup() {
  local extras="${1:-train,export,dev}"
  if [[ ! -d "$VENV" ]]; then
    log "venv oluşturuluyor ($PYBIN)..."
    "$PYBIN" -m venv "$VENV"
  fi
  log "pip güncelleniyor..."
  "$PY" -m pip install --upgrade pip --quiet
  log "bağımlılıklar kuruluyor: .[$extras] (torch/coremltools büyük olabilir)..."
  "$PY" -m pip install -e ".[$extras]"
  log "kurulum tamam. Sürümler:"
  "$PY" - <<'PYEOF'
import importlib
for m in ("torch", "ultralytics", "coremltools", "numpy", "cv2", "jsonschema"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m}: KURULU DEĞİL ({e})")
try:
    import torch
    print(f"  MPS (Apple Silicon GPU): {torch.backends.mps.is_available()}")
except Exception:
    pass
PYEOF
}

cmd_test() {
  ensure_venv
  log "pytest çalıştırılıyor..."
  "$PY" -m pytest "${@:-}"
}

cmd_lint() {
  ensure_venv
  log "ruff check..."
  "$PY" -m ruff check src tests
  log "black --check..."
  "$PY" -m black --check src tests
}

cmd_cli() {
  ensure_venv
  "$PY" -m electrack.cli "$@"
}

cmd_all() {
  cmd_setup
  cmd_lint
  cmd_test
}

main() {
  local sub="${1:-help}"
  shift || true
  case "$sub" in
    setup)            cmd_setup "$@" ;;
    test)             cmd_test "$@" ;;
    lint)             cmd_lint ;;
    all)              cmd_all ;;
    prepare-data|train|export|infer|evaluate|validate-output)
                      cmd_cli "$sub" "$@" ;;
    cli)              cmd_cli "$@" ;;
    help|-h|--help)   sed -n '3,22p' "$ROOT/run.sh" ;;
    *)                err "bilinmeyen komut: $sub"; sed -n '3,22p' "$ROOT/run.sh"; exit 2 ;;
  esac
}

main "$@"
