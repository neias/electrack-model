# Kabul Test Kümesi (Acceptance Set) — Veri Kartı

Sabit, **dondurulmuş** doğrulama/test kümesi. Bir kez oluşturulur; model seçiminden etkilenmez (FR-013, SC-001/SC-002/SC-003). Değerlendirme yalnızca bu küme üzerinde yürür.

## Zorunlu bileşim (kabul metriklerinin anlamlı olması için)

| Gereksinim | Hedef | Neden |
|------------|-------|-------|
| Farklı kart sayısı | **≥ ~20** | Çeşitlilik / istatistiksel anlam (FR-013) |
| Etiketli görüntü | **≥ ~300** | Yeterli örnek |
| Her MVP türünden örnek | yeterli (dengeli) | Per-class recall anlamı |
| **Negatif görüntüler** (boş yüzey / komponent-olmayan) | dahil | FP oranı ölçümü (SC-002, FR-007) |
| **Zorlu koşullar** (zayıf/aşırı ışık, yansıma, çok yakın/uzak, kısmi kapanma, kalabalık) | dahil | FR-008 doğrulaması |
| **Belirsiz nesneler** (opsiyonel `ambiguous/`) | dahil | unknown davranışı (SC-004) |

## Yapı

```text
datasets/acceptance/
├── images/            # test görüntüleri
├── labels/            # YOLO etiketleri (negatifler için boş .txt)
├── ambiguous/         # (ops.) belirsiz-nesne görüntüleri + etiketleri (class_id=-1)
└── DATASET_CARD.md    # bu dosya
```

## Dondurma & sürüm

- Küme `electrack.data.manifest.dataset_id()` ile hash'lenir; rapordaki `dataset_id` bu sürümü sabitler.
- Kümeye örnek eklendiğinde `dataset_id` değişir → eski raporlarla karşılaştırma yaparken dikkat.

> NOT: Gerçek görüntüler `.gitignore` ile izlenmez. Bu kart, kümenin bileşim sözleşmesini tanımlar; görüntüler toplandıkça buraya sayılar/kaynaklar eklenir.
