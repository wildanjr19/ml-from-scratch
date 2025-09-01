import numpy as np


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