# Food Classifier

## Masalah apa yang ingin dipecahkan?

Untuk mengidentifikasi berbagai jenis makanan barat.

## Data-nya darimana?

Data yang digunakan diambil dari https://www.kaggle.com/dansbecker/food-101

## Apa saja fitur dari aplikasi ini?

Beberapa hal yang dapat menjadi patokan pengerjaan aplikasi ini:
* Menggunakan `EfficientNetB0` sebagai arsitektur model dan menggunakan *fine tuning* pada 10 layer terakhir pada model pada training kedua kalinya.
* Dapat mengklasifikasi 101 macam makanan.
* Training dataset sekitar ~7000 gambar.
* Test dataset sekitar ~25000 gambar.

## Apa yang dapat ditingkatkan lagi?

* Arsitektur model dapat dibandingkan dengan `MobileNet`, `ResNetV2` dan dll.
* Dataset bisa ditingkatkan lagi hingga ~50000-100000 gambar dengan lebih dari >200 macam makanan.
