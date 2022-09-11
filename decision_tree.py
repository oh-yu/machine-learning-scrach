import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.faeture = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
    
    def fit(self, X, y):
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._monst_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_thr = self._best_critera(X, y, feat_idxs)

    def predict(self, X):
        # traverse tree
    
    def _most_common_label(self, y):
        counter = Counter(y)
        # the number of occurence for each y label
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_length = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thr = threshold
        return split_idx, split_thr
    
    def _information_gain(self, y, X_column, split_thr):
        # Parent Enetropy
        parent_entropy = entropy(y)
    
        # Generate Split
        left_idxs, right_idxs = self._split(X_column, split_thr)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # Weighted Average Child Entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thr):
        left_idxs = np.argwhere(X_column <= split_thr).flatten()
        right_idxs = np.argwhere(X_column > split_thr).flatten()
        return left_idxs, right_idxs
