import numpy as np
from linear_regression import BaseRegression


class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
            # X shape: (N, D)
            # w shape: (D, )
            linear_model = np.dot(X, w) + b
            # linear_model shape: (N, )
            return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_pred = self._sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)

    print(f"accuracy: {accuracy(pred, y_test)}")
