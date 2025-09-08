import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, X):
        # mean
        """cari mean dari setiap fitur, kurangi X dengan mean-nya"""
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # urutkan eigenvectors berdasarkan eigenvalues
        eigenvectors = eigenvectors.T
        # kembalikan indeks dari eigevectors yang sudah diurutkan
        idxs = np.argsort(eigenvalues)[::-1] # reverse list jadi descending
        # pasangkan eigenvalues dan eigenvectors dengan idxs (urut)
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs] = eigenvectors[0:self.n_components]

        # simpan n pertama
        self.components = eigenvectors[0:self.n_components] 
    
    def transform(self, X):
        # proyeksikan data
        X = X - self.mean
        return np.dot(X, self.components.T)