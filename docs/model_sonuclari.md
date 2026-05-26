# Model Sonuçları

Çalışmada farklı PCA ve MLP kombinasyonları denenmiştir. En iyi gözlenen sonuçlardan biri PCA-15 ve Seed 8 ayarında elde edilmiştir.

| Yöntem | Train Accuracy | Test Accuracy | Not |
| --- | ---: | ---: | --- |
| PCA-15, Seed 8 | 68.8% | 65.0% | PCA denemeleri içinde en iyi gözlenen sonuç |
| FC(48)-FC(24), PCA-15, SGD | 72.12% | 57.0% | 60 parametre denemesi içindeki en iyi kayıt |
| FC(48)-FC(24), PCA-15, SGD | 52.25% | 56.5% | İkinci en iyi parametre denemesi |
| FC(48)-FC(24), PCA-15, SGD | 76.88% | 56.0% | Üçüncü en iyi parametre denemesi |

Model sonuçları `data/processed/model_results.csv` dosyasında daha ayrıntılı biçimde yer almaktadır.

Bu sonuçlar, veri setinin küçük ve özelliklerin sınırlı olması nedeniyle modelin çok yüksek doğruluklara çıkmadığını göstermektedir. Buna rağmen PCA ile boyut indirgeme ve MLP mimari denemeleri, farklı modelleme kararlarının performansa etkisini görmek için faydalı bir karşılaştırma sağlamıştır.
