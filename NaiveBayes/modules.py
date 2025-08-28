import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_faetures = X.shape
        # banyak kelas
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # inisialisasi mean, var, prior
        self._mean = np.zeros((n_classes, n_faetures), dtype=np.float64)
        self._var = np.zeros((n_classes, n_faetures), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            # ambil semua baris dari X, yang labelnya sama dengan c
            X_c = X[c == y]
            # cari mean, var, prior dari tiap fitur pada data yang termasuk kelas c (X_c)
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        pass


    # helper -> prediksi untuk satu sampel
    def _predict(self, x):
        pass