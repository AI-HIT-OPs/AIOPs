import sys
import time
from timed_tasks.predict import Predict
from apscheduler.schedulers.background import BackgroundScheduler

# sys.path.append("") cmd need system path added


# all settings
timesteps = 20
batch_size = 64

filename_available_memory = "../../datasets/db_os_stat(180123-180128).csv"
filename_consistent_gets = "../../datasets/db_stat(180123-180128).csv"
model_available_memory = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
model_consistent_gets = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
column_available_memory = "FREE_MEM_SIZE"
column_consistent_gets = "CONSISTENT_GETS"

# multi model initial
predict_available_memory = Predict(model_available_memory)
predict_consistent_gets = Predict(model_consistent_gets)


def predict_task():
    print(time.strftime('start time: ' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # predict tasks
    predict_available_memory.predict(filename=filename_available_memory,
                                     column=column_available_memory,
                                     timesteps=timesteps)
    predict_consistent_gets.predict(filename=filename_consistent_gets,
                                    column=column_consistent_gets,
                                    timesteps=timesteps)

'''
def predict_task2():
    .....
'''

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(predict_task, 'interval', seconds=5)
    # scheduler.add_job(predict_task_2, 'interval', seconds=3)
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
