from utils.preprocessing import Data

import tensorflow as tf
import matplotlib.pyplot as plt

timesteps = 20
batch_size = 64

filename = "../../datasets/db_stat(180123-180128).csv"
model = "../../model/abnormal_detection_model/execute_count_model/EXECUTE_COUNT_MODEL"
column = "EXECUTE_COUNT"


def test_model():

    # set GPU memory limitation
    gpu_options = tf.GPUOptions(allow_growth=True)
    # initial Session width settings of GPU option
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # initial saver from meta data graph
        saver = tf.train.import_meta_graph(model + ".meta")
        # restore session
        saver.restore(sess, model)
        # get predict position by collection
        predictions = tf.get_collection('predictions')[0]

        graph = tf.get_default_graph()
        # get input frame placeholder by operation name
        x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        data = Data(filename=filename, column=column, timesteps=timesteps)
        test_data = data.getFakeData(data.train_x, batch_size=1)
        feed_dict = {x: test_data[:, :, None], keep_prob: 1.0}
        results = sess.run(predictions, feed_dict=feed_dict)

        plt.plot(timesteps + 1, results[0], '.', label='predicted')
        plt.plot([i for i in range(timesteps)], test_data[0], 'g--', label='real')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    test_model()
