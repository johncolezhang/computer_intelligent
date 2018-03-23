import tensorflow as tf
import numpy as np

n_hidden = 16 # 16 nodes in hidden layer


# rbf function
def gaussian(x, c):
    x = np.matrix(x)
    c = np.matrix(c * x.shape[0]).reshape(x.shape[0], x.shape[1])
    x = np.sum(np.square(np.subtract(x, c)), axis=1)
    return np.exp(x)


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR = [[1, 0], [0, 1], [0, 1], [1, 0]] # 2 classes, 0: [1, 0], 1: [0, 1]
    # XOR = [[1], [0], [0], [1]]



    x = tf.placeholder(dtype='float32', shape=[4, 2])
    z = tf.placeholder(dtype='float32', shape=[4, 2])

    # multi perceptron
    # parameters in 2 layer
    weight1 = tf.Variable(tf.random_normal([2, n_hidden]))
    bias1 = tf.Variable(tf.random_normal([n_hidden]))
    weight2 = tf.Variable(tf.random_normal([n_hidden, 2]))
    bias2 = tf.Variable(tf.random_normal([2]))

    #hidden_layer
    hidden_layer = tf.sigmoid(tf.add(tf.matmul(x, weight1), bias1))
    tf.summary.histogram('hidden_layer', hidden_layer)

    #output
    y = tf.add(tf.matmul(hidden_layer, weight2), bias2)
    tf.summary.histogram('y', y)

    # loss function and gradient
    loss = tf.reduce_mean(tf.square(z - y))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)
    tf.summary.scalar('loss', loss)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(z, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # RBF
    node1 = gaussian(X, [1, 1]) # [1, 1] as center point
    node2 = gaussian(X, [0, 0]) # [0, 0] as center point

    #node1 = np.hstack((node1, node2))
    node1 = X

    init = tf.global_variables_initializer()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/problem2/2', sess.graph)
    sess.run(init)

    for step in range(1001):
        summary, _ = sess.run([merged, train], feed_dict={x: node1, z: XOR})
        writer.add_summary(summary, step)
        if step % 1000 == 0:
            print(step, "train multi network, accuracy: \n", sess.run(accuracy, feed_dict={x: node1, z: XOR}))
            print(step, "train multi network, loss: \n", sess.run(loss, feed_dict={x: node1, z: XOR}))
            print(step, "train multi network, y: \n", sess.run(y, feed_dict={x: node1, z: XOR}))

    result = sess.run(y, feed_dict={x: node1, z: XOR})
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = 0 if result[i, j] < 0.5 else 1
    print("result:\n", result)
