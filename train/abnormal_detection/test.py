import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timesteps = 20


def MaxMinNormalization(x, max, min):
    x = (x - min) / (max - min)
    return x

def generate_data(seq):
    """
    :param seq: continuous sequence of value
    :return:
    """
    x = []
    y = []

    for i in range(len(seq) - timesteps - 1):
        x.append(seq[i: i + timesteps])
        y.append(seq[i + timesteps])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


data = pd.read_csv("../../datasets/db_os_stat(180123-180128).csv")
data = data['FREE_MEM_SIZE'].values.tolist()
data = MaxMinNormalization(data, np.max(data, axis=0), np.min(data, axis=0))

train_x, train_y = generate_data(data)
test_x, test_y = generate_data(data)

lstm_size = 30
lstm_layers = 2
batch_size = 64

x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')





def lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])


outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

outputs = outputs[:, -1]


predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.sigmoid)

cost = tf.losses.mean_squared_error(y_, predictions)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# 获取一个batch_size大小的数据
def get_batches(x, y, batch_size=20):
    for i in range(0, len(x), batch_size):
        begin_i = i
        end_i = i + batch_size if (i + batch_size) < len(x) else len(x)

        yield x[begin_i:end_i], y[begin_i:end_i]


epochs = 100
session = tf.Session()
with session.as_default() as sess:

    tf.global_variables_initializer().run()

    iteration = 1

    for e in range(epochs):
        for xs, ys in get_batches(train_x, train_y, batch_size):

            feed_dict = {x: xs[:, :, None], y_: ys[:, None], keep_prob: .5}

            loss, train_step = sess.run([cost, optimizer], feed_dict=feed_dict)

            if iteration % 100 == 0:
                print('Epochs:{}/{}'.format(e, epochs),
                      'Iteration:{}'.format(iteration),
                      'Train loss: {}'.format(loss))
            iteration += 1

with session.as_default() as sess:
    print(test_x)
    feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
    results = sess.run(predictions, feed_dict=feed_dict)
    plt.plot(results, 'r', label='predicted')
    plt.plot(test_y, 'g--', label='real')
    plt.legend()
    plt.show()









