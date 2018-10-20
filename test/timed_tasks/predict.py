import tensorflow as tf
from utils.preprocessing import Data

timesteps = 20
batch_size = 64

filename = "../../datasets/db_os_stat(180123-180128).csv"
filename1 = "../../datasets/db_stat(180123-180128).csv"
model = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
model1 = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
column1 = "CONSISTENT_GETS"
column = "FREE_MEM_SIZE"

class Predict:
    def __init__(self, model, filename, column, timesteps):
        self.graph=tf.Graph()#为每个类(实例)单独创建一个graph
        with self.graph.as_default():
            self.saver=tf.train.import_meta_graph(model + ".meta")#创建恢复器
             #注意！恢复器必须要在新创建的图里面生成,否则会出错。
        self.sess=tf.Session(graph=self.graph)#创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model)
                predictions = tf.get_collection('predictions')[0]

                graph = tf.get_default_graph()
                # get input frame placeholder by operation name
                x = graph.get_operation_by_name('input_x').outputs[0]
                keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

                data = Data(filename=filename, column=column, timesteps=timesteps)
                test_data = data.getFakeData(data.train_x, batch_size=1)
                feed_dict = {x: test_data[:, :, None], keep_prob: 1.0}
                results = self.sess.run(predictions, feed_dict=feed_dict)
                print(results)



if __name__ == "__main__":
    predict = Predict(model, filename, column, timesteps)
    predict = Predict(model1, filename1, column1, timesteps)