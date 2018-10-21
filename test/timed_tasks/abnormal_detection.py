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
    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, 'interval', seconds=3)
    scheduler.add_job(tick1, 'interval', seconds=3)
    scheduler.start()

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()