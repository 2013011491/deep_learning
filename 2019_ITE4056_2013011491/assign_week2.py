import tensorflow as tf

x_data = [[0, 0], [0,1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

logits = tf.add(tf.matmul(X, W),b)
hidden_output = tf.nn.sigmoid(logits)

W2 = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

logits2 = tf.add(tf.matmul(hidden_output, W2),b2)
output = tf.nn.sigmoid(logits2)

cost = tf.reduce_mean(tf.square(output - Y))
opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        for x, y in zip(x_data, y_data):
            _, cost_val = sess.run([train_op, cost], feed_dict={X:[x], Y:[y]})
        print(step, cost_val, sess.run(W), sess.run(b))

    print(sess.run(output, feed_dict={X:x_data}))