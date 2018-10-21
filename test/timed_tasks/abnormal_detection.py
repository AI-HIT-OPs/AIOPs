import tensorflow as tf
from utils.preprocessing import Data
from timed_tasks.predict import Predict

import sys
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# sys.path.append("") cmd need system path added


def tick():
    print('Tick! The time is: %s' % datetime.now())


def tick1():
    print('Tick! The time1 is: %s' % datetime.now())

if __name__ == '__main__':

    # settings
    timesteps = 20
    batch_size = 64

    filename_available_memory = "../../datasets/db_os_stat(180123-180128).csv"
    filename_consistent_gets = "../../datasets/db_stat(180123-180128).csv"
    model_available_memory = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
    model_consistent_gets = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
    column_available_memory = "FREE_MEM_SIZE"
    column_consistent_gets = "CONSISTENT_GETS"
    # multi model test
    predict1 = Predict(model_available_memory)
    predict1.predict(filename_available_memory, column_available_memory, timesteps)
    predict2 = Predict(model_consistent_gets)
    predict2.predict(filename_consistent_gets, column_consistent_gets, timesteps)

    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, 'interval', seconds=3)
    scheduler.add_job(tick1, 'interval', seconds=3)
    scheduler.start()

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()