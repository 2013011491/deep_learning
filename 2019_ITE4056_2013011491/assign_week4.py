import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform([784, 300], -1., 1.))
b0 = tf.Variable(tf.random_uniform([300], -1., 1.))
L0 = tf.sigmoid(tf.matmul(X, W0) + b0)
L0 = tf.nn.dropout(L0, keep_prob)

W1 = tf.Variable(tf.random_uniform([300, 100], -1., 1.))
b1 = tf.Variable(tf.random_uniform([100], -1., 1.))
L1 = tf.sigmoid(tf.matmul(L0, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_uniform([100, 10], -1., 1.))
b2 = tf.Variable(tf.random_uniform([10], -1., 1.))
model = tf.matmul(L1, W2) + b2


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.6})
        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))


is_correct = tf.equal(tf.arg_max(model, 1), tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1}))

# nodrop-out gd lr 0.3  0.953
# dropout gd lr 0.3 0.946
# dropout adam lr 0.01 0.9775