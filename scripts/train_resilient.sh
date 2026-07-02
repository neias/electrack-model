#!/bin/bash
# Sağlam eğitim: MPS'in aralıklı çökmesine (ultralytics TaskAlignedAssigner shape
# mismatch) karşı otomatik-resume. Çökerse son kontrol noktasından AYNI LR programıyla
# sürdürür (warm-start'ın aksine ağırlıkları bozmaz). caffeinate ile uykuyu engeller.
#
# Kullanım:  bash scripts/train_resilient.sh [EPOCHS] [MAX_RETRY]
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS="${1:-50}"
MAX_RETRY="${2:-30}"
IMGSZ="${IMGSZ:-640}"   # 6500 görüntü + 16GB için hız dengesi (960 çok yavaş)
BATCH="${BATCH:-8}"
LOG=/tmp/train_full.log
: > "$LOG"

run() { caffeinate -i env PYTHONPATH=src python3 -m electrack.cli train "$@" >>"$LOG" 2>&1; }
done_yet() { grep -q "epochs completed" "$LOG"; }
latest_last() { ls -t runs/detect/models/weights/*/weights/last.pt 2>/dev/null | head -1; }

echo "=== FRESH START ($EPOCHS epoch, imgsz $IMGSZ, batch $BATCH) ===" >>"$LOG"
run --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"

i=0
while ! done_yet && [ "$i" -lt "$MAX_RETRY" ]; do
  i=$((i + 1))
  LAST="$(latest_last)"
  if [ -z "$LAST" ]; then echo "RESUME: last.pt yok, çıkılıyor." >>"$LOG"; break; fi
  echo "=== RESUME #$i from $LAST ===" >>"$LOG"
  run --resume --model "$LAST"
done

if done_yet; then
  echo "=== TAMAMLANDI ($i resume) ===" >>"$LOG"
else
  echo "=== DURDU: tamamlanmadı ($i/$MAX_RETRY resume) ===" >>"$LOG"
fi
