import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sample = ("if you want to build a ship, don't drum up people together to "
"collect wood and don't assign them tasks and work, but rather "
"teach them to long for the endless immenstiy of the sea.")

idx2char = list(set(sample)) # index->char
char2idx = {c: i for i, c in enumerate(idx2char)} #char -> index

#hyperparameter
dic_size = len(char2idx) #RNN input size (one hot size)
hidden_size = len(char2idx) #RNN output size
num_classes = len(char2idx) #final ouput size (RNN or softmax, etc.)
batch_size = 170 #one sample data, one batch
sequence_length = 10 # number of lstm rollings (unit #)
learning_rate = 0.1


#x_data = [sample_idx[:-1]] #X data sample (0 ~ n-1) hello: hell
#y_data = [sample_idx[1:]]  #Y label sample (1 ~ n) hello: ello'''

#placholder settings
X = tf.placeholder(tf.int32, [None,sequence_length])
Y = tf.placeholder(tf.int32, [None,sequence_length]) 
x_one_hot = tf.one_hot(X,num_classes)
dataX = []
dataY = []

sample_idx = [char2idx[c] for c in sample] #char to index
for i in range(0,len(sample)-10):
    dataX.append(sample_idx[i:i+10])
    dataY.append(sample_idx[i+1:i+11])


#RNN architectur settings

cell=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot,initial_state=initial_state, dtype=tf.float32)

#FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(400):
        l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
        print(epoch,': ',l)

    results = sess.run(outputs, feed_dict={X:dataX})

    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:
            print(''.join([idx2char[t] for t in index]), end='')
        else:
            print(idx2char[index[-1]], end='')
        