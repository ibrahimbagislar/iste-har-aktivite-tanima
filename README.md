# ISTE-HAR Aktivite Tanıma

Bu proje, ISTE-HAR veri seti üzerinde insan aktivitesi tanıma problemi için hazırlanmış makine öğrenmesi ve derin öğrenme çalışmasıdır. Veri setindeki 20 sayısal özellik kullanılarak iki sınıflı aktivite tahmini yapılmış, farklı MLP mimarileri ve özellik indirgeme yaklaşımları karşılaştırılmıştır.

Çalışmanın ana odağı, ham sayısal özellikleri doğrudan modele vermek yerine ölçekleme, PCA ve farklı model parametreleriyle başarımı karşılaştırmaktır. Bu yüzden proje yalnızca tek bir model eğitimi değil, aynı zamanda model deneme sürecini ve hiperparametre etkisini de göstermektedir.

## Projede Ne Var?

- ISTE-HAR veri setinin okunması ve sınıf dağılımının incelenmesi
- Eğitim ve test verisinin stratified split ile ayrılması
- StandardScaler ile özellik ölçekleme
- PCA ile boyut indirgeme
- MLP tabanlı ikili sınıflandırma modelleri
- Farklı batch size, loss, learning rate, epoch, aktivasyon ve optimizer denemeleri
- Eğitilmiş Keras `.h5` model çıktıları
- Model sonuçlarının CSV olarak saklanması

## Veri Seti

Veri seti 1000 satır ve 21 sütundan oluşmaktadır.

| Alan | Açıklama |
| --- | --- |
| `f1` - `f20` | Modelde kullanılan sayısal özellikler |
| `label` | Hedef sınıf etiketi |

Sınıf dağılımı:

| Sınıf | Kayıt Sayısı |
| --- | ---: |
| 0 | 510 |
| 1 | 490 |

Bu dağılım birbirine yakın olduğu için model eğitiminde sınıf dengesizliği çok baskın değildir. Yine de eğitim ve test ayrımı yapılırken sınıf oranlarını korumak için `stratify` kullanılmıştır.

## Kullanılan Yöntem

Modelleme akışında önce veri eğitim ve test olarak ayrılmıştır. Daha sonra sayısal özellikler `StandardScaler` ile ölçeklendirilmiş, ardından PCA ile daha kompakt bir temsil elde edilmiştir. Son aşamada MLP mimarileri denenmiş ve performans test doğruluğu üzerinden karşılaştırılmıştır.

Denemelerde özellikle şu noktalar karşılaştırılmıştır:

- PCA bileşen sayısı
- MLP katman yapısı
- Aktivasyon fonksiyonu
- Optimizer seçimi
- Learning rate
- Batch size
- Loss fonksiyonu
- Epoch sayısı

## Öne Çıkan Sonuçlar

Notebook denemelerinde PCA tabanlı modellerin doğrudan ham özelliklerle eğitilen modellere göre daha dengeli sonuç verdiği görülmüştür. En iyi gözlenen sonuçlardan biri PCA-15 ve Seed 8 ayarında elde edilmiştir.

| Yaklaşım | Train Accuracy | Test Accuracy |
| --- | ---: | ---: |
| PCA-15, Seed 8 | 68.8% | 65.0% |
| FC(48)-FC(24), PCA-15, SGD | 72.12% | 57.0% |

Sonuçlar, veri setinin sınırlı boyutu ve özelliklerin yapısı nedeniyle modelin çok yüksek doğruluklara kolay ulaşmadığını göstermektedir. Bu yüzden proje, yalnızca en yüksek skoru vermekten çok, farklı modelleme kararlarının sonuca etkisini göstermesi açısından değerlidir.

## Proje Yapısı

```text
ISTE-HAR-Aktivite-Tanima/
├── data/
│   ├── ISTE-HAR_dataset.csv
│   ├── README.md
│   └── processed/
│       └── model_results.csv
├── docs/
│   └── model_sonuclari.md
├── models/
│   ├── mlp_pca7_seed2_test62_5_train61.h5
│   └── mlp_pca8_test60_5_train68.h5
├── notebooks/
│   └── iste_har_aktivite_tanima.ipynb
├── src/
│   └── train_pca_mlp.py
├── .gitignore
├── README.md
└── requirements.txt
```

Ana akış GitHub'da daha temiz görünmesi için `notebooks/iste_har_aktivite_tanima.ipynb` ve `src/train_pca_mlp.py` dosyalarında sadeleştirilmiştir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

Modeli yeniden eğitmek için:

```bash
python src/train_pca_mlp.py
```

Varsayılan ayarlarda script:

- `data/ISTE-HAR_dataset.csv` dosyasını okur
- veriyi eğitim/test olarak ayırır
- StandardScaler ve PCA uygular
- MLP modelini eğitir
- train/test accuracy değerlerini ekrana yazar
- eğitilen modeli `models/iste_har_mlp_pca_model.keras` olarak kaydeder

## Not

Bu çalışma akademik ve portfolyo amaçlıdır. Veri seti küçük olduğu için sonuçlar daha büyük ve gerçek ortam verileriyle yeniden test edilmelidir.
