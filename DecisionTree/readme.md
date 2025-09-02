# Decision Tree

## Entropy
$$E = - \sum p(X) \cdot log_2 (P(X))$$

$p(X) = \frac{x}{n}$

## Information Gain (Gini)
$$IG = E(parent) - [weightedaverage] \cdot E(children)$$

## Intuisi

### Train
- Mulai dari node teratas, dan di setiap node memilih split terbaik berdasar information gain
- **Greedy Search** looping ke seluruh fitur dan seluruh thresholds (semua kemungkinan nilai fitur)
- Simpan split fitur terbaik dan split threshold terbaik di setiap node
- Bangun tree secara rekursif (diulang pada setiap cabang hingga kriteria stop terpenuhi.)
- Untuk menghentikan pertumbuhan node, gunakan beberapa kriteria untuk stop (maximum depth, minimum samples di setiap node, distribution, dll)
- Ketika kita punya leaf node, simpan label kelas yang paling sering muncul


### Prediction
- Melintasi model secara rekursif
- Di setiap node, lihat split terbaik dari test feature vector dan menentukan untuk ke node kanan atau node kiri. berdasar pada, `x[feature_idx] <= threshold`
- Ketika sudah sampai di node terakhit (leaf node), kita kembalikan dan simpan label yang paling sering muncul