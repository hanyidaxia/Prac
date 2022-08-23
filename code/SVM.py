import numpy as np

# only for test
from sklearn import datasets
# from svm import SVM

x, y = datasets.make_blobs(n_samples=100, n_features=4, centers=3, cluster_std=1.05,random_state=55)
y = np.where(y == 0, -1, 1)
#only for test







'''
SVM a classic classification method for binary classification problems
'''
class SVM:
    def __init__(self, lr = 0.001, lambda_param = 0.02, n_iter = 1000):
        self.lr = lr
        self.lambda_param = lambda_param # the regulization term
        self.n_iter = n_iter
        self.w = None
        self.b = None


    def train(self, x, y):
        y_ = np.where(y <= 0, -1, 1)
        samples, features = x.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(x):
                re = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if re:
                    self.w -= self.lr*(2 * self.lambda_param*self.w)
                else:
                    self.w -= self.lr*(2*self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr*y_[idx]


    def pred(self, x):
        output = np.dot(x, self.w) - self.b
        return np.sign(output)



trainer = SVM()
trainer.train(x, y)
print(trainer.w, trainer.b)
