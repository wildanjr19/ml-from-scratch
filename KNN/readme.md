# K-Nearest Neighbors (KNN)

## Apa itu KNN?

K-Nearest Neighbors (KNN) adalah salah satu algoritma machine learning yang paling sederhana dan mudah dipahami. Algoritma ini bekerja berdasarkan prinsip bahwa **"objek yang serupa cenderung berada di lokasi yang berdekatan"**.

### Cara Kerja KNN:

1. **Pilih nilai K** - Tentukan berapa tetangga terdekat yang akan dipertimbangkan
2. **Hitung jarak** - Ukur jarak antara data baru dengan semua data training
3. **Temukan tetangga terdekat** - Pilih K data dengan jarak paling kecil
4. **Voting** - Untuk klasifikasi: pilih kelas yang paling sering muncul di antara K tetangga. Untuk regresi: ambil rata-rata nilai K tetangga

## Rumus Euclidean Distance

Euclidean Distance adalah metrik jarak yang paling umum digunakan dalam KNN. Rumus matematikanya adalah:

### Untuk 2 Dimensi:
```
d = √[(x₂ - x₁)² + (y₂ - y₁)²]
```

### Untuk n Dimensi:
```
d(p,q) = √[∑ᵢ₌₁ⁿ (qᵢ - pᵢ)²]
```

Dimana:
- **d(p,q)** = jarak antara titik p dan titik q
- **pᵢ** = koordinat titik p pada dimensi ke-i
- **qᵢ** = koordinat titik q pada dimensi ke-i
- **n** = jumlah dimensi/fitur