# PCA (Principal Component Analysis)

PCA berusaha menemukan set baru dimensi dimana semua dimensinya orthogonal (dan linear independent) dan melakukan rangking dari variansinya. 

Menemukan transformasi sedemikian rupa :  

1. Fitur yang ditransformasi adalah linear independen.
2. Dimensi dapat direduksi dengan mengambil hanya dimensi yang memiliki nilai importanca tinggi.
3. Dimensi baru yang ditemukan harus meminimalisasi projection error.
4. Projection poin harus memiliki variansi maksimum.

## Variansi
Seberapa besar sebaran data
$$Var(X) = \frac{1}{n} \sum (X_i - \bar{X})^2$$

## Covariance Matriks
$$Cov(X, Y) = \frac{1}{n}(X_i - \bar{X})(Y_i - \bar{Y})^T$$
$$Cov(X, X) = \frac{1}{n}(X_i - \bar{X})(X_i - \bar{X})^T$$

## Eigenvector dan Eigenvalue


## Intuisi
- Kurangi mean dari X
- Hitung $Cov(X, X)$
- Hitung eigenvalue dan eigenvector dari covariance matrix
- Urutkan eigenvector berdasar eigenvalue-nya dari yang terbesar
- Pilih $k$ eigenvector, dan darisini akan menjadi $k$ dimensi baru
- Tranform yang asli dengan n dimensional menjadi $k$ dimensi