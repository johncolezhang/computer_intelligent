import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# rbf function
def gaussian(x, c):
    x = np.matrix(x[:, 0: 2])
    c = np.matrix(c * x.shape[0]).reshape(x.shape[0], x.shape[1])
    x = np.sum(np.square(np.subtract(x, c)), axis=1)
    return np.exp(x)


def drawLind(w):
    slope = - w[0, 0] / w[0, 1]


if __name__ == "__main__":
    X = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    XOR = [[0], [1], [1], [0]]  # 0: [1, 0], 1: [0, 1]
    OR = [[0], [1], [1], [1]]

    w1 = np.random.random((3, 1))

    for i in range(100):
        y1 = np.tanh(np.matrix(X) * w1)
        w1 = w1 + np.transpose(X) * (OR - y1)
    print("w1 in OR:")
    print(w1)

    print("Adaline result in OR:")
    print(np.tanh(np.matrix(X) * w1))

    # rbf
    x1 = gaussian(np.matrix(X), [1, 1])
    x2 = gaussian(np.matrix(X), [0, 0])
    x_gaussian = np.hstack((x1, x2))
    x_gaussian = np.hstack((x_gaussian, [[1], [1], [1], [1]]))

    w2 = np.random.random((3, 1))

    for i in range(100):
        y2 = np.tanh(np.matrix(x_gaussian) * w2)
        w2 = w2 + np.transpose(x_gaussian) * (XOR - y2)
    print("w2 in XOR:")
    print(w2)

    print("Adaline result in XOR using RBF")
    print(np.tanh(x_gaussian * w2))

    x = np.arange(-2, 2, 0.1)
    y = x * (-(w1[0, 0] / w1[1, 0]))
    plt.title("Adaline in OR")
    plt.plot([0, 0, 1, 1], [0, 1, 0, 1], 'ro')
    plt.plot(x, y)
    plt.show()