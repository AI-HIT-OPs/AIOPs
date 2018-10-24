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
filename_cpu_usage = "../../datasets/db_os_stat(180123-180128).csv"
filename_execute_count = "../../datasets/db_stat(180123-180128).csv"
filename_parse_count_hard = "../../datasets/db_stat(180123-180128).csv"
filename_parse_count = "../../datasets/db_stat(180123-180128).csv"
filename_redo_size = "../../datasets/db_stat(180123-180128).csv"
filename_session_active = "../../datasets/db_os_stat(180123-180128).csv"
filename_session_logical_reads = "../../datasets/db_stat(180123-180128).csv"
filename_sessions = "../../datasets/db_stat(180123-180128).csv"
filename_user_calls = "../../datasets/db_stat(180123-180128).csv"
model_available_memory = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
model_consistent_gets = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
model_cpu_usage = "../../model/abnormal_detection_model/cpu_usage_model/USER_CPU_MODEL"
model_execute_count = "../../model/abnormal_detection_model/execute_count_model/EXECUTE_COUNT_MODEL"
model_parse_count_hard = "../../model/abnormal_detection_model/parse_count_hard_model/PARSE_COUNT_HARD_MODEL"
model_parse_count = "../../model/abnormal_detection_model/parse_count_model/PARSE_COUNT_TOTAL_MODEL"
model_redo_size = "../../model/abnormal_detection_model/redo_size_model/REDO_SIZE_MODEL"
model_session_active = "../../model/abnormal_detection_model/session_active_model/ACTIVE_SESSIONS_MODEL"
model_session_logical_reads = "../../model/abnormal_detection_model/session_logical_reads_model/SESSION_LOGICAL_READS_MODEL"
model_sessions = "../../model/abnormal_detection_model/sessions_model/TOTAL_SESSIONS_MODEL"
model_user_calls = "../../model/abnormal_detection_model/user_calls_model/USER_CALLS_MODEL"
column_available_memory = "FREE_MEM_SIZE"
column_consistent_gets = "CONSISTENT_GETS"
column_cpu_usage = "USER_CPU"
column_execute_count = "EXECUTE_COUNT"
column_parse_count_hard = "PARSE_COUNT_HARD"
column_parse_count = "PARSE_COUNT_TOTAL"
column_redo_size = "REDO_SIZE"
column_session_active = "ACTIVE_SESSIONS"
column_session_logical_reads = "SESSION_LOGICAL_READS"
column_sessions = "TOTAL_SESSIONS"
column_user_calls = "USER_CALLS"

# multi model initial
predict_available_memory = Predict(model_available_memory)
predict_consistent_gets = Predict(model_consistent_gets)
predict_cpu_usage = Predict(model_cpu_usage)
predict_execute_count = Predict(model_execute_count)
predict_parse_count_hard = Predict(model_parse_count_hard)
predict_parse_count = Predict(model_parse_count)
predict_redo_size = Predict(model_redo_size)
predict_session_active = Predict(model_session_active)
predict_session_logical_reads = Predict(model_session_logical_reads)
predict_sessions = Predict(model_sessions)
predict_user_calls = Predict(model_user_calls)


def predict_task():
    print(time.strftime('start time: ' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # predict tasks
    predict_available_memory.predict(filename=filename_available_memory,
                                     column=column_available_memory,
                                     timesteps=timesteps)
    predict_consistent_gets.predict(filename=filename_consistent_gets,
                                    column=column_consistent_gets,
                                    timesteps=timesteps)
    predict_cpu_usage.predict(filename=filename_cpu_usage,
                              column=column_cpu_usage,
                              timesteps=timesteps)
    predict_execute_count.predict(filename=filename_execute_count,
                                  column=column_execute_count,
                                  timesteps=timesteps)
    predict_parse_count_hard.predict(filename=filename_parse_count_hard,
                                     column=column_parse_count_hard,
                                     timesteps=timesteps)
    predict_parse_count.predict(filename=filename_parse_count,
                                column=column_parse_count,
                                timesteps=timesteps)
    predict_redo_size.predict(filename=filename_redo_size,
                              column=column_redo_size,
                              timesteps=timesteps)
    predict_session_active.predict(filename=filename_session_active,
                                   column=column_session_active,
                                   timesteps=timesteps)
    predict_session_logical_reads.predict(filename=filename_session_logical_reads,
                                          column=column_session_logical_reads,
                                          timesteps=timesteps)
    predict_sessions.predict(filename=filename_sessions,
                             column=column_sessions,
                             timesteps=timesteps)
    predict_user_calls.predict(filename=filename_user_calls,
                               column=column_user_calls,
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
