from platform import node
import numpy as np
from collections import Counter

# entropy
def entropy(y):
    # hitung probabilitas awal dari kelas
    hist = np.bincount(y)
    p_y = hist / len(y)
    return -np.sum([p * np.log2(p) for p in p_y if p > 0]) # menghindari log 0

# kelas node
class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature          # index fitur untuk split
        self.threshold = threshold      # nilai threshold untuk split
        self.left = left                # node kiri
        self.right = right              # node kanan
        self.value = value              # nilai kelas untuk leaf node

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Args:
            min_samples_splilt : jumlah minimum sampel yang diperlukan di sebuah node agar node tersebut dapat di split
            max_depth : kedalaman maksimum dari tree -> membatasi pertumbuhan tree
            n_feats : jumlah fitur yang dipilih secara acak untuk dipertimbangkan saat melakukan split
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        """Pertumbuhan pohon / penambahan node"""
        # pastikan fitur
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # grow
        self.root = self._grow_tree(X, y)

    # helper -> growing tree / penambahan node
    def _grow_tree(self, X, y, depth = 0):
        """Fungsi utama untuk training, membuat node dan pertumbuhan pohon"""
        # insialisasi parameter
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)
    

        # ambil fitur indeks dari n_features sebanyak n_feats dan tanpa pengembalian (replace=false)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search (looping) -> mencari fitur dan threshold terbaik
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        # split node kanan dan kiri
        left_idxs, right_idxs = self._split(X[:, best_feat],  best_thresh)
        # rekursif
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    # helper -> mencari kriteria terbaik
    def _best_criteria(self, X, y, feat_idxs):
        # buat nilai patokan nilai gain -1
        best_gain = -1
        split_idx, split_thresh = None, None

        # looping
        for feat_idx in feat_idxs:
            # ambil semua baris pada tiap feat_idxs
            X_column = X[:, feat_idx]
            # dapatkan threshold dengan mencari nilai uniq di tiap kolom/variabel
            thresholds = np.unique(X_column)
            # looping
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                # get
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    # helper -> hitung information gain
    def _information_gain(self, y, X_column, split_thresh):
        """Hitung information gain"""
        # E(parent)
        parent_entropy = entropy(y)
        # buat split -> kiri dan kanan
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # mencegah node kosong
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # [weighted average] * E(children)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        children_entropy = (n_left /n) * entropy_left + (n_right / n) * entropy_right

        # totalkan
        ig = parent_entropy - children_entropy
        return ig

    # helper -> split
    def _split(self, X_column, split_thresh):
        """Split data berdasarkan threshold"""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # helper -> mencari label yang paling sering muncul
    def _most_common_label(self, y):
        """Mencari label yang paling sering muncul"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # melewatkan data di pohon
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # helper -> traverse tree / melewatkan data di pohon
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)