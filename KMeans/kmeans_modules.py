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
        # simpan centroid awal 
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        """optimisasi -> iterasi selama max_itres"""
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # cek apakah sudah konvergen
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # kembalikan label cluster
        return self._get_cluster_labels(self.clusters)

    # helper -> create clusters
    def _create_clusters(self, centroids):
        """mengkluster setiap data ke centroid terdekat"""
        # buat list kosong sebanyak K, yang akan diisi indeks
        clusters = [[] for _ in range(self.K)]
        # looping setiap data pada dataset (X)
        for idx, sample in enumerate(self.X):
            # setiap data akan dimasukkan ke indeks centroid terdekat
            centroid_idx = self._closest_centroid(sample, centroids)
            # idx (indeks data) dimasukkan ke dalam ke list clusters pada indeks centroid terdekatnya
            clusters[centroid_idx].append(idx)
        return clusters
    
    # helper -> closest centroid
    def _closest_centroid(self, sample, centroids):
        """mengembalikan indeks centroid terdekat dari data"""
        # jarak dari sample data ke setiap centroid
        distances = euclidean_distance(sample, centroids)
        # dapatkan indeks centroid dengan jarak terdekat
        closest_index = np.argmin(distances)
        return closest_index
    
    # helper -> get centroids
    def _get_centroids(self, clusters):
        """menghitung centorid baru dari data yang sudah dikluster"""
        # array kosong [k, n_features]
        centroids = np.zeros((self.K, self.n_features))
        # hitung mean fitur di tiap kluster, lalu assign ke centroids
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    # helper -> is converged
    def _is_converged(self, centroids_old, centroids):
        """cek konvergensi. jika jarak antara centroids lama dan baru 0 maka konvergen (True)"""
        # hitung jarak antara centroids lama dan baru pada tiap kluster
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0
    
    # helper -> get cluster labels
    def _get_cluster_labels(self, clusters):
        """setiap sampel data akan memiliki label kluster yang sesuai"""
        # inisialisasi array kosong
        labels = np.empty(self.n_samples)

        # clusters -> list berisi indeks dari data
        
        # looping untuk tiap kluster
        for cluster_idx, cluster in enumerate(clusters):
            # looping untuk tiap indeks data pada kluster
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
