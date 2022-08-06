import numpy as np


class BaseRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Init Params
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)
            # y_predicted shape: (N, 1)
            # X shape: (N, 1)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _approximation(self, X, w, b):
        raise NotImplementedError()
    
    def _predict(self, X, w, b):
        raise NotImplementedError()

class LinearRegression(BaseRegression):
    # super().__init__()

    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    # X shape: (N, 1)
    # y shape: (N, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    plt.scatter(X_test[:, 0], y_test, label="actual")
    plt.plot(X_test[:, 0], y_pred, c="r", label="pred")
    plt.legend()
    plt.show()



