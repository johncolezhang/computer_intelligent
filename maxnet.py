import numpy as np


if __name__ == "__main__":
    weight = -0.15
    u = {0: np.array([0.1, 0.3, 0.5, 0.7, 0.9])}
    time = 10
    for i in range(time):
        vec = u[i]
        new_vec = []
        for j in range(5):
            w = np.array([weight] * 5)
            w[j] = 1
            x = np.matmul(vec, w)
            new_vec.append(x if x > 0 else 0)
        u[i + 1] = np.array(new_vec)
    print(u[time])