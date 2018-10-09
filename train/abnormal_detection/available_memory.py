from utils.read_csv import Reader
from utils.lstm_model import lstm_model

import tensorflow as tf

timesteps = 20
epochs = 100
batch_size = 64


def train_model():
    # define training process
    x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # loading lstm model
    predictions = lstm_model(x, keep_prob)
    # mse loss function
    cost = tf.losses.mean_squared_error(y_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # prepare data
    reader = Reader("../../datasets/db_os_stat(180123-180128).csv", "FREE_MEM_SIZE")

    # define session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        iteration = 1
        for e in range(epochs):
            for xs, ys in reader.getBatch(reader.train_x, reader.train_y, batch_size):
                feed_dict = {x: xs[:, :, None], y_: ys[:, None], keep_prob: .5}
                loss, train_step = sess.run([cost, optimizer], feed_dict=feed_dict)
                if iteration % 100 == 0:
                    print('Epochs:{}/{}'.format(e, epochs),
                          'Iteration:{}'.format(iteration),
                          'Train loss: {}'.format(loss))
            iteration += 1


if __name__ == "__main__":
    train_model()



