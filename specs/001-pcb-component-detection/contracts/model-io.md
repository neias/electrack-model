# Contract: Model Girdi/Çıktı (Core ML `.mlpackage`)

Bu sözleşme, dağıtılan modelin (ve referans `inference/detector.py` sarmalayıcısının) girdi/çıktı biçimini tanımlar. Tüketen masaüstü uygulaması bu sözleşmeye bağlıdır.

## Girdi (Input)

| Özellik | Değer |
|---------|-------|
| Tip | RGB görüntü (Core ML `imageType`) |
| Renk | RGB, 8-bit/kanal |
| Kaynak çözünürlük | **Herhangi** — uygulama ham kareyi verir (FR-001a) |
| Ön-işleme | Yeniden boyutlandırma (letterbox, en-boy korunur) + [0,1] normalize **model/sarmalayıcı tarafında** yapılır |
| Model iç boyutu | Sabit (ör. 640×640) — dışa aktarımda gömülü; uygulamadan gizlenir |

**Sözleşme kuralı**: Uygulama çözünürlük/renk dönüşümü yapmak zorunda değildir; herhangi bir RGB kare kabul edilir. Letterbox dolgusu nedeniyle oluşan koordinat dönüşümü sarmalayıcıda geri alınır, böylece döndürülen kutular **orijinal kareye göre normalize** edilir.

## Çıktı (Output)

Ham Core ML çıktısı (gömülü NMS + eşik ile) sarmalayıcıda [`detection-output.schema.json`](./detection-output.schema.json) biçimine dönüştürülür:

- `bbox`: normalize [0-1], eksen-hizalı, köşe `[x_min, y_min, x_max, y_max]` (orijinal kareye göre).
- `class_name`: 6 MVP sınıfı veya `"unknown"`.
- `class_id`: 0..5 veya `null` (unknown).
- `confidence`: [0,1].

## Son-işleme sözleşmesi (model içinde)

Modelde gömülü (FR-002a):
1. **NMS** — çakışan kutular bastırılır (sınıf-içi).
2. **`det_threshold`** — genel tespit güven eşiği; altındaki adaylar elenir (hayali tespit yok — FR-007).
3. **`class_threshold`** — sınıf kesinlik eşiği; `det_threshold` üstünde ama bunun altındaki tespitler `unknown` olarak etiketlenir (FR-004, iki-eşik mantığı: `research.md` R5).

Eşikler dışa-aktarım zamanında sabitlenir; farklı çalışma noktası yeni bir dışa aktarım/paket sürümü demektir.

## Değişmezler (invariants)

- Aynı gerçek nesne için en fazla bir kutu (NMS).
- Boş/komponentsiz kare → boş `detections` (FR-007).
- Kutu koordinatları daima [0,1] içinde ve `x_min<x_max, y_min<y_max`.
- Çıkarım tamamen yerel; hiçbir ağ çağrısı yok (FR-005).

## Genişletme (yeni sınıf)

Yeni sınıf eklendiğinde: `data.yaml` sınıf listesi büyür → yeniden eğitim → yeniden dışa aktarım. Sözleşme yapısı değişmez; yalnızca `class_name` enum kümesi ve olası `class_id` aralığı genişler (FR-009). Tüketen uygulama, bilinmeyen `class_name` değerlerini zarifçe ele almalıdır (ileri uyumluluk).
