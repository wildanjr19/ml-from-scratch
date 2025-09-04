# Random Forest
Random forest adalah perkembangan lebih lanjut dari Decision Tree dimana algoritma ini termasuk ke dalam menggunakan metode voting. 

## Intuisi
Data akan dilewatkan ke setiap pohon dan akan menghasilkan prediksi kelas-kelas yang berbeda, dari sini dilakukan teknik voting dengan mencari kelas terbanyak yang muncul.

Random forest menggunakan teknik bootstrapping untuk membuat subset data yang unik dari dataset asli. Hal ini memungkinkan pohon dalam random forest untuk dilatih pada versi data yang sedikit berbeda

### Cara kerja bootstrapping
1. Pengambilan sampel dengan pengembalian
2. Hasil dari pengambilan sampel ini akan selalu menghasilkan nilai/elemen pengambilan yang sedikit berbeda satu sama lain (menghasilkan subset data baru)
3. Setiap pohon dilatih secara independen menggunakan subset data hasil bootstrapping
