import tensorflow as tf


def threshold(x, thres=0.5):
    if x < thres:
        return 0.0
    else:
        return 1.0


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR = [[0], [1], [1], [0]] # 0: [1, 0], 1: [0, 1]
    OR = [[0], [1], [1], [1]]

    #parameter definition
    x_adaline = tf.placeholder(tf.float32, shape=[4, 2])
    z = tf.placeholder(tf.float32, shape=[4, 1])
    theta_adaline = tf.Variable(tf.random_uniform([2, 1]))
    cons = tf.constant([[0], [0], [0], [0]], dtype='float32')

    #perceptron
    y = tf.tanh(tf.add(tf.matmul(x_adaline, theta_adaline), cons)) #percetron function
    tf.summary.histogram('y', y)

    # loss function and gradient
    loss = tf.reduce_mean(tf.square(z - y))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)
    tf.summary.scalar('loss', loss)

    # calculate accuracy
    correct_prediction = tf.equal(tf.to_int32(y > 0.5), tf.to_int32(z))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/problem1/2', sess.graph)
    sess.run(init)


    # for step in range(5001):
    #     sess.run(train, feed_dict={x_adaline: X, z: XOR})
    #     if step % 1000 == 0:
    #         print(step, "train adaline in XOR, loss: \n", sess.run(loss, feed_dict={x_adaline: X, z: XOR}))
    #         print(step, "train adaline in XOR, y: \n", sess.run(y, feed_dict={x_adaline: X, z: XOR}))
    #         print(step, "train adaline in XOR, accuracy: \n", sess.run(accuracy, feed_dict={x_adaline: X, z: XOR}))
    #         print(step, "train adaline in XOR, theta: \n", sess.run(theta_adaline, feed_dict={x_adaline: X, z: XOR}))



    for step in range(1001):
        summary, _ = sess.run([merged, train], feed_dict={x_adaline: X, z: OR})
        writer.add_summary(summary, step)
        if step % 1000 == 0:
            print(step, "train perceptron in OR, loss: \n", sess.run(loss, feed_dict={x_adaline: X, z: OR}))
            print(step, "train perceptron in OR, y: \n", sess.run(y, feed_dict={x_adaline: X, z: OR}))
            print(step, "train perceptron in OR, accuracy: \n", sess.run(accuracy, feed_dict={x_adaline: X, z: OR}))
            print(step, "train perceptron in OR, theta: \n", sess.run(theta_adaline, feed_dict={x_adaline: X, z: OR}))