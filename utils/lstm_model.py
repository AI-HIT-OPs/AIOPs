import tensorflow as tf


def initPlaceholder(timesteps):
    x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return x, y_, keep_prob


def lstm_model(x, lstm_size, lstm_layers, keep_prob):
    # define basis structure LSTM cell
    def lstm_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    # multi layer LSTM cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    # dynamic rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    # reverse
    outputs = outputs[:, -1]
    # fully connected
    predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.sigmoid)
    return predictions
