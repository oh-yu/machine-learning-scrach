import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # 1. compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # 2. get k-nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 3. majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Import Module
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Prepare Data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    plt.scatter(X[:, 2], X[:, 3], c=y)
    plt.show()

    # KNN
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    # Accuracy
    acc = np.sum(predict == y_test) / predict.shape[0]
    print(f"acc:{acc}")
