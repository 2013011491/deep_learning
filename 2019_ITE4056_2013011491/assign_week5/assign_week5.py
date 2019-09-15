import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 784])

W0 = tf.Variable(tf.random_uniform([784, 128], -1., 1.))
b0 = tf.Variable(tf.random_uniform([128], -1., 1.))
L0 = tf.sigmoid(tf.matmul(X, W0) + b0)
L0 = tf.nn.dropout(L0, keep_prob)

W2 = tf.Variable(tf.random_uniform([128, 64], -1., 1.))
b2 = tf.Variable(tf.random_uniform([64], -1., 1.))
L2 = tf.sigmoid(tf.matmul(L0, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob)

W11 = tf.Variable(tf.random_uniform([64, 128], -1., 1.))
b11 = tf.Variable(tf.random_uniform([128], -1., 1.))
L11 = tf.sigmoid(tf.matmul(L2, W11) + b11)
L11 = tf.nn.dropout(L11, keep_prob)

W3 = tf.Variable(tf.random_uniform([128, 784], -1., 1.))
b3 = tf.Variable(tf.random_uniform([784], -1., 1.))
model = tf.sigmoid(tf.matmul(L11, W3) + b3)


#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#cost = tf.reduce_mean(tf.pow(Y-model, 2))
cost = tf.reduce_mean(tf.abs(Y-model))
#optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(25):
    total_cost = 0

    for i in range(total_batch):
        batch_xs,batch_ys= mnist.train.next_batch(batch_size)
        batch_x_noisy= batch_xs + np.random.rand(100,784)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_x_noisy, Y: batch_xs, keep_prob: 0.7})
        total_cost += cost_val
        
    print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    

x, y = mnist.train.next_batch(1)
xs=x+np.random.rand(1,784)

mnist_image = np.array(x).reshape((28,28))
plt.imshow(mnist_image, cmap="gray")
plt.show()

mnist_image1 = np.array(xs).reshape((28,28))
plt.imshow(mnist_image1, cmap="gray")
plt.show()

mnist_image2 = model.eval(session=sess,feed_dict={X: xs, Y: x, keep_prob: 1.0})

mnist_image2=np.array(mnist_image2).reshape((28,28))
plt.imshow(mnist_image2, cmap="gray")
plt.show()

# nodrop-out gd lr 0.3  0.953
# dropout gd lr 0.3 0.946
# dropout adam lr 0.01 0.9775