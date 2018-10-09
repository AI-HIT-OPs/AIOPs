import tensorflow as tf

lstm_size = 32
lstm_layers = 2


def lstm_model(X, keep_prob):
    # define basic cell
    def lstmCell():
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    # define multi cell
    cell = tf.contrib.rnn.MultiRNNCell([lstmCell() for _ in range(lstm_layers)])
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    outputs = outputs[:, -1]
    predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.sigmoid)
    return predictions
