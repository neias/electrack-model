"""Kamera ile canlı PCB komponent tespiti demosu.

Kullanım:
  python scripts/camera_demo.py                       # canlı pencere ('q' ile çık)
  python scripts/camera_demo.py --snapshot out.png    # tek kare yakala + işaretle (GUI'siz)
  python scripts/camera_demo.py --source 1            # farklı kamera indeksi
  python scripts/camera_demo.py --conf 0.4            # güven eşiği

Not: macOS ilk çalıştırmada kamera izni ister; terminal/uygulamaya erişim ver.
Canlı pencere bir ekran (GUI) gerektirir — bu yüzden komutu KENDİ terminalinde
çalıştır: `! python scripts/camera_demo.py`
"""

from __future__ import annotations

import argparse
import time

DEFAULT_MODEL = "models/demo/best.pt"  # repoyla gelir (git-izlemeli); iMac'te de hazır


def main() -> int:
    ap = argparse.ArgumentParser(description="electrack kamera demosu")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default="0", help="kamera indeksi (0/1) veya video yolu")
    ap.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="çıkarım boyutu (model 768'de eğitildi; yoğun kartta minik SMD için 1280+ dene)",
    )
    ap.add_argument(
        "--conf", type=float, default=0.35, help="tespit güven eşiği (0.35 kabul noktası)"
    )
    ap.add_argument("--snapshot", default=None, help="tek kare yakala, işaretle, buraya kaydet")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("HATA: Kamera açılamadı — macOS kamera izni verildi mi? (Ayarlar > Gizlilik)")
        return 1

    if args.snapshot:
        frame = None
        for _ in range(8):  # ısınma: ilk kareler siyah/otomatik-pozlama olabilir
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("HATA: Kameradan kare okunamadı (izin/erişim?).")
            return 1
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        cv2.imwrite(args.snapshot, res.plot())
        print(f"Kaydedildi: {args.snapshot} | tespit sayısı: {len(res.boxes)}")
        return 0

    print("Canlı demo başladı — çıkmak için pencereye tıklayıp 'q'.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        annotated = res.plot()
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(
            annotated,
            f"{fps:4.1f} FPS  |  {len(res.boxes)} komponent",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("electrack — kamera (q ile cik)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
