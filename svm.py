import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                if y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1:
                    # x_i shape: (n_features, )
                    # self.w shape: (n_features, )
                    self.w -=  self.lr * (self.w*2*self.lambda_param)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        linear = np.dot(X, self.w) - self.b
        return np.sign(linear)


if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X,y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)
    print(clf.w, clf.b)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    def get_hyperplane_x1(w, x0, b, offset):
        return (-w[0]*x0 + b + offset) / w[1]
    
    # w1x1+w0x0+b=0
    x0_min = np.amin(X[:, 0])
    x0_max = np.amax(X[:, 0])
    x1_min = get_hyperplane_x1(clf.w, x0_min, clf.b, 0)
    x1_max = get_hyperplane_x1(clf.w, x0_max, clf.b, 0)

    # w1x1+w0x0+b=1
    x1_min_positive = get_hyperplane_x1(clf.w, x0_min, clf.b, 1)
    x1_max_positive = get_hyperplane_x1(clf.w, x0_max, clf.b, 1)

    # w1x1+w0x0+b=-1
    x1_min_negative = get_hyperplane_x1(clf.w, x0_min, clf.b, -1)
    x1_max_negative = get_hyperplane_x1(clf.w, x0_max, clf.b, -1)

    plt.plot([x0_min, x0_max], [x1_min, x1_max])
    plt.plot([x0_min, x0_max], [x1_min_positive, x1_max_positive])
    plt.plot([x0_min, x0_max], [x1_min_negative, x1_max_negative])
    plt.show()
