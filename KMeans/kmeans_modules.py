import numpy as np

# set random
np.random.seed(42)

# euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K = 5, max_iters = 100, plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list samples untuk setiap cluster
        self.clusters = [[] for _ in range(self.K)]
        # mean fitur untuk setiap cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        """inisialisasi centroid awal secara random"""
        # pilih K random sample sebagai centorid awal -> mengembalikan index data tanpa pengembalian
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        # simpan centroid awal tsbt
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        """optimisasi -> iterasi selama max_itres"""
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            # update centroids

            # cek apakah sudah konvergen

        # kembalikan label cluster


    # helper -> update clusters
        
        