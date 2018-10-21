import tensorflow as tf
from utils.preprocessing import Data

# settings
timesteps = 20
batch_size = 64

filename_available_memory = "../../datasets/db_os_stat(180123-180128).csv"
filename_consistent_gets = "../../datasets/db_stat(180123-180128).csv"
filename_cpu_usage = ""
model = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
model1 = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
column1 = "CONSISTENT_GETS"
column = "FREE_MEM_SIZE"


g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2

# 加载第一个模型
with sess1.as_default():
    with g1.as_default():
        tf.global_variables_initializer().run()
        # initial saver from meta data graph
        saver = tf.train.import_meta_graph(model + ".meta")
        # restore session
        saver.restore(sess1, model)

# 加载第二个模型
with sess2.as_default():  # 1
    with g2.as_default():
        tf.global_variables_initializer().run()
        # initial saver from meta data graph
        saver = tf.train.import_meta_graph(model1 + ".meta")
        # restore session
        saver.restore(sess2, model1)


def test1():
    with sess1.as_default():
        with sess1.graph.as_default():
            # get predict position by collection
            predictions = tf.get_collection('predictions')[0]

            # get input frame placeholder by operation name
            x = g1.get_operation_by_name('input_x').outputs[0]
            keep_prob = g1.get_operation_by_name('keep_prob').outputs[0]

            data = Data(filename=filename, column=column, timesteps=timesteps)
            test_data = data.getFakeData(data.train_x, batch_size=1)
            feed_dict = {x: test_data[:, :, None], keep_prob: 1.0}
            results = sess1.run(predictions, feed_dict=feed_dict)
            print(results)


def test2():
    with sess2.as_default():
        with sess2.graph.as_default():
            # get predict position by collection
            predictions = tf.get_collection('predictions')[0]

            # get input frame placeholder by operation name
            x = g2.get_operation_by_name('input_x').outputs[0]
            keep_prob = g2.get_operation_by_name('keep_prob').outputs[0]

            data = Data(filename=filename1, column=column1, timesteps=timesteps)
            test_data = data.getFakeData(data.train_x, batch_size=1)
            feed_dict = {x: test_data[:, :, None], keep_prob: 1.0}
            results = sess2.run(predictions, feed_dict=feed_dict)
            print(results)

test1()
test2()