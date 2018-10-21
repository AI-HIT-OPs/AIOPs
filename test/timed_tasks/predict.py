import tensorflow as tf
from utils.preprocessing import Data


class Predict:
    def __init__(self, model):
        # initial session
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model + ".meta")
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model)
                self.predictions = tf.get_collection('predictions')[0]
                self.graph = tf.get_default_graph()
                # get input frame placeholder by operation name
                self.x = self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]

    def predict(self, filename, column, timesteps):
        data = Data(filename=filename, column=column, timesteps=timesteps)
        test_data = data.getFakeData(data.train_x, batch_size=1)
        feed_dict = {self.x: test_data[:, :, None], self.keep_prob: 1.0}
        results = self.sess.run(self.predictions, feed_dict=feed_dict)
        print(results)


if __name__ == "__main__":
    # settings
    timesteps = 20
    batch_size = 64

    filename = "../../datasets/db_os_stat(180123-180128).csv"
    filename1 = "../../datasets/db_stat(180123-180128).csv"
    model = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
    model1 = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
    column1 = "CONSISTENT_GETS"
    column = "FREE_MEM_SIZE"
    # multi model test
    predict1 = Predict(model)
    predict1.predict(filename, column, timesteps)
    predict2 = Predict(model1)
    predict2.predict(filename1, column1, timesteps)
