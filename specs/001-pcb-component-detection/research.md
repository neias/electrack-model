# Research: PCB Komponent Tespiti — Teknik Kararlar

Bu belge, plan Technical Context'indeki açık kararları ve alternatiflerini gerekçeleriyle kayıt altına alır. Tüm "NEEDS CLARIFICATION" noktaları burada çözülmüştür.

---

## R1. Model mimarisi ve boyutu

**Decision**: Ultralytics **YOLO11** ailesi; MVP için **YOLO11n** (nano) ile başla, ≥15 FPS hedefi tutuyorsa ve doğruluk yetersizse **YOLO11s** (small)'a yüksel. Girdi çözünürlüğü 640×640 (gerekirse 512 ile hız/doğruluk dengesi).

**Rationale**:
- Tek-aşamalı (single-stage) anchor-free detektör; gerçek zamanlı kullanım için tasarlanmış.
- Ultralytics doğrudan Core ML dışa aktarımı ve gömülü NMS destekler (bkz. R4).
- 6 sınıf + ~50 nesne/kare, nano/small kapasitesi için rahat; Apple Neural Engine'de düşük gecikme.
- Genişletilebilirlik: sınıf sayısı `data.yaml` ile yönetilir; yeni sınıf head'i yeniden boyutlandırır ama hat/kod değişmez (FR-009).

**Alternatives considered**:
- **YOLOv8**: Olgun ama YOLO11 daha iyi hız/doğruluk dengesi sunar. Geriye dönük seçenek olarak açık.
- **RT-DETR / DETR türevleri**: Daha ağır, Neural Engine'de gerçek zamanlı garanti zayıf; MVP için gereksiz karmaşıklık (YAGNI).
- **İki aşamalı (detektör + ayrı sınıflandırıcı)**: Daha yüksek doğruluk potansiyeli ama gecikme + karmaşıklık; MVP'de reddedildi.

---

## R2. Hedef donanımda gerçek zamanlı performans (Apple Silicon M4)

**Decision**: Çıkarım **Core ML** üzerinden, hesaplama birimi `all` (Neural Engine + GPU + CPU). Uçtan-uca bütçe ≤66 ms/kare (≥15 FPS), ~50 nesne. Ölçüm önişleme + inference + NMS/eşik dahil.

**Rationale**:
- M4 Neural Engine, nano/small YOLO için tipik olarak 66 ms bütçesinin çok altında tek-kare gecikme sağlar; ~50 nesnede NMS maliyeti düşük kalır.
- Birleşik bellek kopya maliyetini azaltır.
- FPS bütçesi bir kabul metriğidir (SC-003); değerlendirme harness'i hedef donanımda ölçer.

**Alternatives considered**:
- **CPU-only (ONNX Runtime CPU)**: Taşınabilir ama M4'te Neural Engine'i boşa harcar; hız marjı riskli. Reddedildi.
- **Metal/MPS doğrudan**: Core ML zaten MPS/ANE'yi kullanır; ek karmaşıklık gereksiz.

**Not**: Kesin FPS, model boyutu ve çözünürlükle değişir; `evaluate` adımı gerçek ölçümü rapora yazar. 15 FPS tutmazsa sıra: çözünürlük düşür (640→512) → nano'da kal → gerekiyorsa yarı-hassasiyet (fp16) doğrula.

---

## R3. Veri kümesi stratejisi (açık kümeler + kendi verimiz)

**Decision**: İki kaynaklı hibrit veri stratejisi, YOLO formatında birleştirilir:
1. **Açık kümeler** (lisans kısıtı yok — araştırma dahil, spec kararı): Öncelikle **Roboflow Universe** üzerindeki PCB *komponent tespiti* (through-hole/IC/kondansatör vb.) projeleri; akademik kümeler (ör. **FICS-PCB**, **PCB-DSLR** benzeri komponent-etiketli setler) uygunluk/etiket kalitesi açısından değerlendirilir.
2. **Kendi çekimlerimiz**: Açık kümelerde eksik/zayıf temsil edilen 6 MVP türü için kendi PCB görüntülerimizi toplayıp etiketleriz (özellikle TO-92 transistör, delikli direnç, LED gibi türler ve zorlu koşullar).

**Sınıf eşleme**: Her açık kümenin etiket taksonomisi bizim 6-sınıf kümesine eşlenir (`convert.py` içinde eşleme tablosu). Eşlenemeyen/gürültülü sınıflar dışlanır.

**Negatif örnekler**: Boş yüzey / komponent-olmayan nesne (kalem, parmak, arka plan) görüntüleri boş etiket dosyalarıyla eklenir → yanlış-pozitif bastırma (FR-007, SC-002).

**Rationale**:
- Spec, açık kümeleri değerlendir + eksikleri kendi verinle tamamla diyor (FR-010).
- Lisans kısıtı kaldırıldığından (clarify kararı) Roboflow/akademik non-commercial kümeler kullanılabilir → hızlı başlangıç.
- Negatif görüntüler FP oranını doğrudan düşürür.

**Alternatives considered**:
- **Yalnızca sentetik veri**: Domain gap riski (gerçek yansıma/ışık/kalabalık); tek başına reddedildi, augmentasyonla desteklenir.
- **Yalnızca kendi verimiz**: Etiketleme maliyeti yüksek, kapsam dar. Reddedildi.

**Doğrulama görevi (prepare-data)**: Aday açık kümeler indirilip etiket kalitesi ve lisansı `datasets/raw/<source>/DATASET_CARD.md` içine kaydedilir; taksonomi eşlemesi belgelenir.

---

## R4. Core ML dışa aktarımı + gömülü NMS/eşik

**Decision**: Ultralytics `export(format="coreml", nms=True)` ile **`.mlpackage`** üret. NMS ve güven eşiği modele gömülür; çıktı doğrudan kutu + skor + sınıf verir. Girdi: Core ML `imageType` (sabit boyut, ör. 640×640); ön-işleme (yeniden boyutlandırma/normalize) dışa-aktarım hattında kapsüllenir.

**Rationale**:
- FR-002a: model içeride eşik + NMS uygulamalı → `nms=True` bunu karşılar; tüketen uygulama ham aday filtrelemesiyle uğraşmaz.
- `.mlpackage` Neural Engine için tercih edilen paketleme.
- Girdi `imageType` olması, uygulamanın `CVPixelBuffer`/görüntü vermesini kolaylaştırır; ancak spec "herhangi çözünürlük" dediği için: model sabit boyut bekler, **yeniden boyutlandırma referans `inference/detector.py` sarmalayıcısında** (ve uygulama tarafı entegrasyon notunda) tanımlanır — mantıksal olarak "ön-işleme model tarafında" sözleşmesi korunur.

**Çıktı → sözleşme dönüşümü**: Core ML NMS çıktısı (piksel/relatif kutu + sınıf skorları) `contracts/detection-output.schema.json`'daki **normalize [0-1] xyxy** biçimine çevrilir. Bu normalize dönüşüm sarmalayıcıda yapılır ve uygulamaya bu biçimde sunulur (FR-002).

**Alternatives considered**:
- **NMS'i uygulamada yap**: Reddedildi — FR-002a'ya aykırı, entegrasyonu zorlaştırır.
- **ONNX + ort**: Apple Neural Engine'i tam kullanmaz. Reddedildi.
- **TFLite**: Apple ekosisteminde Core ML'e göre dezavantajlı.

**Ayarlanabilir eşik**: Varsayılan güven eşiği (ör. 0.25) dışa-aktarım parametresi olarak saklanır; farklı çalışma noktası gerekiyorsa yeniden aktarımla ayarlanır. (Operasyonel eşik seçimi, SC-001 recall + SC-002 FP kısıtlarını aynı anda sağlayacak biçimde `evaluate` adımında taranır.)

---

## R5. "unknown" / arka plan (background) stratejisi

**Decision**: MVP'de model **6 gerçek sınıf** üzerinde eğitilir; `unknown` ayrı bir eğitilmiş sınıf DEĞİLDİR. `unknown`, çıktı sözleşmesinde şu türetme mantığıyla yüzeye çıkar:
- Detektör bir kutu önerir (yeterli objectness/güven) ama **en yüksek sınıf güveni, sınıf-kabul eşiğinin altındaysa** → tespit `class_name="unknown"`, `class_id=null` olarak döndürülür.
- Bu, "nesne var ama türünü güvenle diyemiyorum" durumunu FR-004'e uygun biçimde açık `unknown` etiketiyle verir.
- **Yanlış-pozitif bastırma** ayrı mekanizmadır: objectness/güven, genel tespit eşiğinin de altındaysa hiç tespit üretilmez (FR-007). Negatif (background) görüntülerle eğitim bunu güçlendirir.

**İki eşik**:
1. `det_threshold` (genel tespit): altında → tespit yok (FP bastırma).
2. `class_threshold` (sınıf kesinliği): `det_threshold` üstünde ama bunun altında → `unknown`.

**Rationale**:
- Ayrı bir "unknown" sınıfını doğrudan eğitmek, tutarlı negatif tanım gerektirir ve gürültülüdür; eşik-türetimli unknown daha basit ve genişletilebilir (YAGNI).
- İki-eşik yaklaşımı SC-004 (belirsizde yüksek-güvenli yanlış sınıflandırma üretme) ile SC-002 (FP < %10) hedeflerini ayrı ayrı kontrol edilebilir kılar.

**Alternatives considered**:
- **Eğitilmiş "unknown/other" sınıfı**: İleride belirsiz komponentler için etiketli veri biriktikçe eklenebilir (genişletilebilirlik); MVP için reddedildi.
- **Open-set / OOD skorlama (enerji, MC-dropout)**: Aşırı karmaşık, cihaz-üstü gerçek zamanı zorlar. Reddedildi.

**Not**: `class_threshold` ve `det_threshold` değerleri `evaluate` adımında, doğrulama kümesi üzerinde SC kısıtlarını sağlayacak biçimde seçilir ve rapora yazılır.

---

## R6. Zorlu koşullara dayanıklılık (augmentasyon)

**Decision**: Eğitimde standart YOLO augmentasyonları + hedefli ekler: parlaklık/kontrast/gamma (zayıf/aşırı ışık), highlight/yansıma simülasyonu, ölçek jitter (çok yakın/uzak kart), mozaik + kırpma (kalabalık/kısmi kapanma), hafif blur/noise. Doğrulama/kabul kümesi bu koşullardan gerçek örnekler içerir.

**Rationale**: Spec Edge Cases (ışık, ölçek, kapanma, kalabalık) doğrudan augmentasyon + gerçek örnek stratejisine eşlenir (FR-008). Güven, bu koşullarda kalibre edilir → yanlış tespit bastırma.

**Alternatives considered**: Test-time augmentation (TTA) — gecikmeyi artırır, gerçek zaman bütçesini bozar; kabul çıkarımında kapalı.

---

## R7. Değerlendirme metriği implementasyonu

**Decision**: Özel harness (`eval/metrics.py`):
- **overall_recall** = mikro-ortalama recall @ IoU 0.5 (tüm görünür komponentler; doğru = IoU≥0.5 ve sınıf doğru) → SC-001.
- **false_positive_rate** = (gerçek komponente eşleşmeyen tespit sayısı) / (toplam üretilen tespit) → SC-002; negatif görüntülerdeki tüm tespitler FP sayılır.
- **per_class** recall/precision/support; **unknown_behavior** (belirsiz örneklerde yüksek-güvenli yanlış sınıflandırma) → SC-004.
- **latency_fps** hedef donanımda ölçülür → SC-003.
- Tekrar üretilebilirlik: sabit kabul kümesi + sabit seed + sürümlü model id (FR-013, SC-006).

**Rationale**: Metrik tanımları clarify'da sabitlendiği için (mikro-recall @ IoU0.5, FP oranı) harness bunları birebir uygular; standart mAP yerine spec'e sadık kalınır (mAP ek bilgi olarak raporlanabilir).

**Alternatives considered**: Yalnızca COCO mAP — kullanıcı ifadesinden farklı; birincil kabul metriği olarak reddedildi, tamamlayıcı olarak tutulabilir.

---

## R8. Tekrar üretilebilirlik ve veri sürümleme

**Decision**: Konfig-odaklı hat (`training/config/*.yaml`), sabit `seed`, deterministik bölünme (`splits.py`, hash-tabanlı). Veri kümeleri git-ignore; içerik bir manifest/`data.yaml` + isteğe bağlı DVC ile izlenir. Model artefaktları `models/` altında sürüm/hash ile.

**Rationale**: FR-013 + SC-006 tekrar üretilebilirliği zorunlu kılar; büyük ikili dosyalar git dışında tutulur.

**Alternatives considered**: Her şeyi git'e koymak — repo şişer; reddedildi.
