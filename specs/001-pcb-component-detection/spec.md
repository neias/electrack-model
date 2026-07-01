# Feature Specification: PCB Komponent Tespiti (Cihaz-Üstü Nesne Tespit Modeli)

**Feature Branch**: `001-pcb-component-detection`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "Elektronik kart (PCB) fotoğraflarında/karelerinde komponentleri tespit edip sınıflandıran, cihaz üzerinde çevrimdışı çalışan bir nesne tespit modeli ve onu üreten eğitim/değerlendirme hattı geliştirmek."

## Clarifications

### Session 2026-07-01

- Q: Model hangi hesaplama hedefinde gerçek zamanlı çalışmalı? → A: Apple Silicon M4 Mac Mini (birleşik bellek + GPU + Neural Engine mevcut; sadece-CPU değil, cihaz-üstü hızlandırma kullanılabilir).
- Q: "Canlı his" için kabul edilecek gerçek zamanlı performans eşiği ne olmalı (~50 komponentli karede)? → A: ≥15 FPS (≤66 ms/kare).
- Q: SC-001'deki "%80 doğru tür" hedefi tam olarak nasıl ölçülmeli? → A: Tüm görünür komponentler üzerinde toplam (mikro-ortalama) recall ≥ %80, IoU 0.5 eşiğiyle.
- Q: "Bilinmeyen" durumu tüketen uygulamaya nasıl aktarılmalı? → A: Çıktıda ayrı bir "bilinmeyen" (unknown) sınıfı olarak döndürülür.
- Q: Açık veri kümeleri için lisans/kullanım kısıtı ne olmalı? → A: Kısıt yok; araştırma-amaçlı (non-commercial) dahil her türlü açık veri kümesi değerlendirilebilir.
- Q: Tespit çıktısındaki sınırlayıcı kutu koordinatları hangi biçimde dönmeli? → A: Normalize edilmiş [0-1] köşe biçimi (x_min, y_min, x_max, y_max), eksen-hizalı.
- Q: Model, tespitleri döndürmeden önce içeride son-işleme uygulamalı mı? → A: Evet; model ayarlanabilir bir varsayılan güven eşiği ve NMS uygulayıp yalnızca kesinleşmiş tespitleri döndürür.
- Q: Tüketen uygulama modele kareyi hangi biçimde verir? → A: Herhangi bir çözünürlükte RGB kare; yeniden boyutlandırma/normalizasyon model tarafında yapılır.
- Q: Sabit doğrulama/test kümesinin asgari büyüklüğü ne olmalı? → A: En az ~20 farklı kart ve ~300+ etiketli görüntü; her MVP türünden yeterli örnek.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Bir kamera karesinde komponentleri tespit etme (Priority: P1)

Tüketen masaüstü uygulaması modele tek bir kamera karesi (görüntü) verir. Model, kare içinde görünen her elektronik komponent için görüntüdeki konumunu (sınırlayıcı kutu), tür sınıfını ve bir güven değerini döndürür. Model uygulamaya gömülü olarak tamamen çevrimdışı çalışır; hiçbir görüntü harici bir servise gönderilmez. Güvenle sınıflandırılamayan nesneler "bilinmeyen" olarak işaretlenir.

**Why this priority**: Bu, ürünün var oluş sebebidir — çalışan bir tespit yeteneği olmadan projenin hiçbir değeri yoktur. Tek başına bu hikâye teslim edildiğinde uygulama canlı komponent tespiti yapabilir ve MVP değeri oluşur.

**Independent Test**: Sabit bir doğrulama kartı görüntü kümesi modele verilerek, dönen tespitlerin (kutu + sınıf + güven) elle etiketlenmiş gerçek değerlerle (ground truth) karşılaştırılmasıyla bağımsız olarak test edilebilir. Doğru tür oranı ve yanlış-pozitif oranı ölçülür.

**Acceptance Scenarios**:

1. **Given** üzerinde MVP kapsamındaki türlerden komponentler bulunan net bir PCB görüntüsü, **When** görüntü modele verilir, **Then** her görünür komponent için doğru tür sınıfı, görüntü üzerindeki konumunu makul biçimde saran bir sınırlayıcı kutu ve 0–1 aralığında bir güven değeri döndürülür.
2. **Given** üzerinde hiç komponent olmayan boş bir yüzey veya komponent olmayan bir nesne (örn. kalem, parmak) içeren görüntü, **When** görüntü modele verilir, **Then** model hayali tespit üretmez (hiç tespit döndürmez veya çok düşük güvenle bastırır).
3. **Given** modelin daha önce eğitilmediği veya türü belirsiz bir nesne, **When** görüntü modele verilir, **Then** model yanlış bir tür yerine "bilinmeyen" etiketiyle veya düşük güvenle işaretler.
4. **Given** ~50 komponente kadar kalabalık bir kart görüntüsü, **When** görüntü modele verilir, **Then** model çıktıyı canlı his verecek kadar hızlı üretir ve tespitlerin çoğunluğu doğru konumlandırılır.

---

### User Story 2 - Modeli tekrarlanabilir biçimde üretme (Priority: P2)

Model geliştiricisi, çalışan model dosyasını sıfırdan yeniden üretebilmek için gereken tüm adımları izleyebilir: eğitim verisinin toplanması ve etiketlenmesi (açık veri kümelerinin değerlendirilmesi + eksik türler için kendi kart görüntülerinin toplanıp etiketlenmesi), modelin eğitilmesi, cihaz-üstü çalıştırmaya uygun taşınabilir bir formata dışa aktarılması ve doğruluğunun ölçülmesi.

**Why this priority**: Tek seferlik bir model dosyası uzun vadede yetersizdir; verinin genişlemesi, yeni türlerin eklenmesi ve kalite denetimi için hattın tekrarlanabilir olması gerekir. P1 sonrası en yüksek değer buradadır.

**Independent Test**: Belgelenen adımlar takip edilerek, aynı veri kümesi düzeninden başlanıp aynı taşınabilir model çıktısı ve aynı değerlendirme raporunun (kabul metrikleri) üretilebildiği doğrulanır.

**Acceptance Scenarios**:

1. **Given** tanımlı veri kümesi düzeni ve etiketleme kuralları, **When** geliştirici eğitim sürecini çalıştırır, **Then** cihaz-üstü çalıştırmaya uygun taşınabilir bir model çıktısı üretilir.
2. **Given** eğitilmiş bir model ve sabit bir doğrulama/test kartı kümesi, **When** değerlendirme süreci çalıştırılır, **Then** doğru tür oranı ve yanlış-pozitif oranını içeren, kabul eşiklerine göre geç/kal kararı veren bir rapor üretilir.
3. **Given** tekrarlanabilir hat, **When** aynı girdilerle süreç yeniden çalıştırılır, **Then** kabul metrikleri karşılaştırılabilir (tekrar üretilebilir) sonuçlar verir.

---

### User Story 3 - Yeni komponent türü ekleme (Priority: P3)

Model geliştiricisi, tür kümesini (örn. SMD parçalar ve diğerleri) tüm hattı yeniden yazmadan, yalnızca veri ekleyip yeniden eğiterek genişletebilir.

**Why this priority**: Genişletilebilirlik uzun vadeli değer ve bakım maliyeti açısından kritiktir ama ilk teslimat için zorunlu değildir; MVP altı türle çalışabilir.

**Independent Test**: Kapsama yeni bir tür için etiketli veri eklenip yeniden eğitim çalıştırıldığında, hattın kod/yapı değişikliği olmadan yeni türü tanıyan bir model ürettiği doğrulanır.

**Acceptance Scenarios**:

1. **Given** yeni bir tür için etiketlenmiş görüntüler, **When** tür kümesine eklenip yeniden eğitim çalıştırılır, **Then** hat mimarisi yeniden yazılmadan yeni türü de tanıyan bir model üretilir.
2. **Given** genişletilmiş tür kümesi, **When** değerlendirme çalıştırılır, **Then** mevcut türlerin doğruluğu kabul eşiklerinin altına düşmeden yeni tür raporlanır.

---

### Edge Cases

- **Zayıf/aşırı ışık ve yansıma**: Model, aydınlatma bozuk olduğunda güveni düşürerek yanlış tespiti bastırabilmelidir; hayali yüksek güvenli tespit üretmemelidir.
- **Ölçek uçları (çok yakın / çok uzak kart)**: Kart çerçeveye sığmayacak kadar yakın veya komponentler ayırt edilemeyecek kadar uzak olduğunda, model belirsiz durumları düşük güvenle işaretler.
- **Kısmi kapanma / üst üste binme**: Komponentler kısmen kapandığında veya birbirine bindiğinde, model mümkün olan tespitleri üretir ve ayırt edemediklerinde güveni düşürür.
- **Kalabalık kartlar**: Karede ~50 komponente kadar yoğunlukta model çalışmaya devam eder; sayı üst sınırının aşıldığı durumda davranış öngörülebilir olmalıdır (bkz. Assumptions).
- **Türü belirsiz / eğitilmemiş komponentler**: Yanlış tür üretmek yerine "bilinmeyen" etiketi veya düşük güven kullanılır.
- **Komponent olmayan nesneler / boş yüzey**: Hayali tespit üretilmez.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Model, tek bir görüntü (kamera karesi) girdisi alıp o karede tespit edilen her komponent için bir tespit listesi döndürmelidir.
- **FR-002**: Her tespit; görüntü üzerindeki konumu tanımlayan bir sınırlayıcı kutu, bir tür sınıfı (6 MVP türünden biri veya "bilinmeyen") ve 0–1 aralığında bir güven değeri içeren standart, tüketen uygulamanın kolayca kullanabileceği bir tespit yapısı olmalıdır. Sınırlayıcı kutu, normalize edilmiş [0-1] köşe biçiminde (x_min, y_min, x_max, y_max) ve eksen-hizalı olmalıdır.
- **FR-001a**: Model girdisi, herhangi bir çözünürlükteki bir RGB kamera karesi olmalı; modelin beklediği boyuta yeniden boyutlandırma ve normalizasyon model tarafında (uygulamada değil) yapılmalıdır.
- **FR-002a**: Model, tespitleri döndürmeden önce içeride son-işleme uygulamalıdır: ayarlanabilir bir varsayılan güven eşiği ile çakışan kutuları bastırma (NMS) uygulayıp yalnızca kesinleşmiş tespitleri döndürmelidir.
- **FR-003**: MVP sürümü şu 6 komponent türünü tanımalıdır: entegre/IC, elektrolitik kondansatör, konnektör/header, LED, delikli direnç, TO-92 transistör.
- **FR-004**: Model, komponent olduğunu tespit edip güvenle sınıflandıramadığı nesneleri düşük güvenli/yanlış bir tür üretmek yerine çıktıda ayrı bir "bilinmeyen" (unknown) sınıfı olarak işaretleyebilmelidir.
- **FR-005**: Model, uygulamaya gömülü olarak tamamen çevrimdışı çalışmalı ve hiçbir görüntüyü ya da tespiti harici bir servise göndermemelidir.
- **FR-006**: Model, taşınabilir ve cihaz-üstü çalıştırmaya uygun bir formatta paketlenebilmeli, böylece masaüstü uygulaması tarafından yerel olarak tüketilebilmelidir.
- **FR-007**: Model, boş yüzeyde veya komponent olmayan nesnelerde hayali (yanlış-pozitif) tespit üretmemelidir.
- **FR-008**: Model, zorlu koşullarda (zayıf/aşırı ışık, yansıma, ölçek uçları, kısmi kapanma, kalabalık kart, belirsiz tür) güven değerini düşürerek yanlış tespitleri bastırabilmelidir.
- **FR-009**: Tür kümesi, SMD ve diğer parçalarla genişletilebilecek şekilde tasarlanmalı; yeni tür eklemek tüm hattın yeniden yazılmasını gerektirmemeli, yalnızca veri ekleme + yeniden eğitim ile mümkün olmalıdır.
- **FR-010**: Proje kapsamı, eğitim verisinin toplanması ve etiketlenmesi için bir strateji ile bir veri kümesi düzeni tanımlamalıdır (açık veri kümelerinin değerlendirilmesi + eksik türler için kendi kart görüntülerinin toplanıp etiketlenmesi). Açık veri kümelerinde lisans kısıtı uygulanmaz; araştırma-amaçlı (non-commercial) dahil her türlü açık küme değerlendirilebilir.
- **FR-011**: Proje, modelin doğruluğunu ölçen ve kabul eşiklerine göre geç/kal kararı veren bir değerlendirme/kabul süreci içermelidir.
- **FR-012**: Model, Apple Silicon M4 Mac Mini hedef donanımında bir karede ~50 komponente kadar tespiti en az 15 kare/saniye (≤66 ms/kare) hızında üretebilmelidir.
- **FR-013**: Değerlendirme, sabit (değişmeyen) bir doğrulama/test kartı kümesi üzerinden yürütülmeli ve tekrar üretilebilir sonuçlar vermelidir. Bu küme, kabul metriklerinin anlamlı olması için en az ~20 farklı kart ve ~300+ etiketli görüntü içermeli; her MVP türünden yeterli örnek barındırmalıdır.

### Key Entities *(include if feature involves data)*

- **Görüntü / Kamera Karesi**: Modele girdi olarak verilen tek bir PCB görüntüsü. Değişken aydınlatma, ölçek ve kalabalık düzeye sahip olabilir.
- **Tespit (Detection)**: Model çıktısının tek bir birimi. Nitelikleri: sınırlayıcı kutu (normalize [0-1], eksen-hizalı, köşe biçimi x_min/y_min/x_max/y_max), tür sınıfı (6 MVP türünden biri veya "bilinmeyen"), güven değeri (0–1). Model içeride eşikleme + NMS uyguladığından liste yalnızca kesinleşmiş tespitleri içerir.
- **Komponent Türü (Class)**: Tanınabilir komponent kategorisi. MVP'de 6 tür + "bilinmeyen"; kümenin ileride genişletilebilir olması beklenir.
- **Veri Kümesi**: Etiketli PCB görüntüleri koleksiyonu (eğitim/doğrulama/test bölünmeleri). Açık kaynaklı ve kendi toplanan görüntüleri içerir; tanımlı bir düzen ve etiketleme kuralına uyar.
- **Değerlendirme Raporu**: Sabit test kümesi üzerinde ölçülen doğru tür oranı, yanlış-pozitif oranı ve kabul geç/kal kararını içeren çıktı.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Sabit bir doğrulama/test kartı kümesinde, tüm görünür komponentler üzerinde toplam (mikro-ortalama) recall en az %80'dir; bir tespit, gerçek komponentle IoU ≥ 0.5 örtüşme sağladığında ve tür sınıfı doğru olduğunda "doğru" sayılır.
- **SC-002**: Modelin ürettiği tespitlerin yanlış-pozitif oranı %10'un altındadır (boş yüzey veya komponent olmayan nesnede hayali tespit üretmez).
- **SC-003**: Model, Apple Silicon M4 Mac Mini hedef donanımında bir karede ~50 komponente kadar en az 15 kare/saniye (≤66 ms/kare) hızında çıkarım yapar.
- **SC-004**: Türü belirsiz/eğitilmemiş nesneler için model, yanlış tür üretme yerine "bilinmeyen"/düşük güven davranışını gösterir; belirsiz test örneklerinde yüksek güvenli yanlış sınıflandırma üretmez.
- **SC-005**: Yeni bir komponent türü, hat mimarisi yeniden yazılmadan yalnızca veri ekleme + yeniden eğitim ile kullanıma alınabilir.
- **SC-006**: Eğitim/değerlendirme hattı, aynı veri kümesi düzeninden başlanarak taşınabilir model çıktısını ve kabul metriklerini tekrar üretilebilir biçimde üretir.

## Assumptions

- **Referans donanım**: Hedef donanım Apple Silicon M4 Mac Mini'dir (birleşik bellek + GPU + Neural Engine). Cihaz-üstü hızlandırma kullanılabilir; kabul testleri bu donanımda ölçülür.
- **Gerçek zamanlı hedef eşiği**: "Canlı his", hedef donanımda kare başına ~50 komponente kadar en az 15 kare/saniye (≤66 ms/kare) etkileşimli çıkarım olarak sabitlenmiştir.
- **"Doğru tür ile tespit" tanımı**: Bir tespit, gerçek komponentle IoU ≥ 0.5 örtüşme sağladığında ve tür sınıfı doğru olduğunda "doğru" sayılır; kabul metriği tüm görünür komponentler üzerinde mikro-ortalama recall'dir.
- **Yanlış-pozitif tanımı**: Yanlış-pozitif, gerçek bir komponente karşılık gelmeyen (boş yüzey veya komponent-olmayan nesne üzerinde) üretilmiş tespittir; oran değerlendirme kümesindeki üretilen tespitler üzerinden hesaplanır.
- **Kalabalık üst sınırı**: Bir karede ~50 komponent üst tasarım sınırıdır; bunun üzerindeki durumlarda öncelik en güvenli/en belirgin komponentlere verilir ve performans zarif biçimde düşer.
- **Gizlilik**: Tüm görüntü işleme cihaz üzerinde ve çevrimdışı yapılır; hiçbir görüntü kalıcı olarak dışa aktarılmaz veya iletilmez.
- **Veri kaynakları**: Açık veri kümeleri lisans kısıtı olmadan (araştırma-amaçlı dahil) kalite/kapsam açısından değerlendirilecek, eksik türler için kendi kart görüntüleri toplanıp etiketlenecektir.
- **Çıktı sözleşmesi**: Sınırlayıcı kutular normalize [0-1] köşe biçimindedir (x_min, y_min, x_max, y_max), eksen-hizalıdır. Model içeride ayarlanabilir güven eşiği + NMS uygular ve yalnızca kesinleşmiş tespitleri döndürür. Girdi, herhangi bir çözünürlükte RGB karedir; ön-işleme model tarafında yapılır.
- **Kabul test kümesi**: Sabit doğrulama/test kümesi en az ~20 farklı kart ve ~300+ etiketli görüntü içerir; her MVP türünden yeterli örnek barındırır.
- **Tüketen uygulama kapsam dışı**: Masaüstü uygulamasının kendisi (kullanıcı arayüzü, kamera akışı yönetimi) bu özelliğin kapsamı dışındadır; kapsam model + üretim/değerlendirme hattıdır.
