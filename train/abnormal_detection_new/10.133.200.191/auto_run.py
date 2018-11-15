import os
lst = os.listdir("/home/AIOPs-master/train/abnormal_detection_new/10.133.200.191")
print(lst)
for c in lst:
    if c.endswith(".py") and c.find("auto_run") == -1:
        os.system("python3 {}".format(c))
