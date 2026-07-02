# Yeni Komponent Türü Ekleme (US3 / FR-009 / SC-005)

Tür kümesi **veri-odaklıdır**: yeni bir sınıf eklemek hat/mimari kod değişikliği gerektirmez — yalnızca `data.yaml` + veri + yeniden eğitim.

## Adımlar

1. **`datasets/yolo/data.yaml`'a sınıfı ekle** (sona, yeni id ile — mevcut id'ler değişmez):

   ```yaml
   names:
     0: ic
     1: capacitor
     2: connector
     3: led
     4: resistor
     5: transistor
     6: diode               # <-- yeni
   ```

   İsteğe bağlı: `src/electrack/data/classes.py` içindeki `DISPLAY_NAMES`'e insan-okur ad ekleyin (zorunlu değil; tespit sözleşmesini etkilemez).

2. **Etiketli veri ekle**: Yeni sınıf için görüntü + YOLO etiketlerini `datasets/yolo/images/*` ve `labels/*` altına koy (etiketleme kuralları: `src/electrack/labeling/guidelines.md`). Yeterli örnek + kabul kümesine temsil ekleyin.

3. **Yeniden eğit**:

   ```sh
   python -m electrack.cli prepare-data
   python -m electrack.cli train
   python -m electrack.cli export --weights models/weights/mvp/weights/best.pt
   ```

4. **Değerlendir**: Yeni tür raporlanır; **mevcut türlerin recall'ü kabul eşiğinin altına düşmemeli** (regresyon testi: `tests/integration/test_add_class.py`).

## Neden kod değişmez

- Sınıf listesi tek yerden (`data.yaml`) okunur (`ClassRegistry.from_data_yaml`).
- `train`/`export`/`eval` sınıf sayısını/adlarını buradan alır; sabit-kodlu sınıf yoktur.
- Çıktı sözleşmesi (`detection-output.schema.json`) yapısı sabittir; yalnızca geçerli `class_name` kümesi genişler. Tüketen uygulama bilinmeyen `class_name` değerlerini zarifçe ele almalıdır (ileri uyumluluk — bkz. `contracts/model-io.md`).
