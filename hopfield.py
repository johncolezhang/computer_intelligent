import numpy as np
from matplotlib import pyplot as plt

def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0 # in order to stable
    return w



def retrieve_pattern(weights, data, steps=10):
    res = np.array(data) # v

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res) # u
            if raw_v > 0: # sgn
                res[i] = 1
            else:
                res[i] = -1
    return res


def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data


def plot_images(images, title, no_i_x, no_i_y=3):
    fig = plt.figure(figsize=(4, 6))
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, 3, 3)
    images = np.pad(images, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=-1)
    for i in range(no_i_x): # row
        for j in range(no_i_y): # column
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 0 and i == 1:
                ax.set_title("Distorted")
            elif j == 1 and i == 0:
                ax.set_title("Reconstructed")


if __name__ == "__main__":
    neurons = 9
    perfect = {
        "A": [1, 1, 1, -1, 1, -1, -1, 1, -1],
        "B": [1, 1, 1, 1, -1, -1, 1, 1, 1]
    }
    distort = {
        "A": [1, 1, 1, -1, 1, -1, 1, 1, -1],
        "B": [1, 1, 1, 1, -1, -1, 1, 1, -1]
    }
    train_data = [np.array(p) for p in perfect.values()]
    test_data = [(np.array(perfect[k]), np.array(distort[k])) for k in perfect.keys()]
    W = train(neurons, train_data)
    accuracy, op_imgs = test(W, test_data)
    plot_images(op_imgs, "Reconstructed Data", 2)
    plt.show()