import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # x_i shape: (n_features, )

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)
                delta = self.lr * (y_[idx]-y_pred)
                # delta shape: (1, )

                self.weights += delta * x_i
                self.bias += delta

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2,  cluster_std=1.05, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    perceptron = Perceptron()
    perceptron .fit(X_train, y_train)
    print(X_test.shape)
    y_pred = perceptron.predict(X_test)
    print(f"Naive Bayes Test Accuracy: {accuracy(y_test, y_pred)}")

    y_train = [1 if i > 0 else 0 for i in y_train]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlabel("X1")
    plt.ylabel("X2")
    
    x0_min = np.amin(X_train[:, 0])
    x0_max = np.amax(X_train[:, 0])
    x1_min = (-perceptron.weights[0] * x0_min + perceptron.bias) / perceptron.weights[1]
    x1_max = (-perceptron.weights[0] * x0_max + perceptron.bias) / perceptron.weights[1] 
    plt.plot([x0_min, x0_max], [x1_min, x1_max], "k", label="boundary")

    plt.legend()
    plt.show()