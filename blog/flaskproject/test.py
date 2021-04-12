import os
#os.chdir('/home/www/drebin/drebin')
#advanced.speed.booster
with os.popen ("python /home/www/drebin/drebin/classification_drebin.py -f '/home/www/blog/flaskproject/advanced.speed.booster.apk'") as tt:
    read=tt.read()
print("结束")