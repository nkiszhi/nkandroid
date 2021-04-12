#coding=utf-8
#import sys
#import os
#import multiprocessing
#path_of_current_file=os.path.abspath(__file__)
#path_of_current_dir=os.path.split(path_of_current_file)[0]
#_file_name=os.path.basename(__file__)
#sys.path.insert(0,path_of_current_dir)
#worker_class='sync'
#workers=multiprocessing.cpu_count()*2+1
#chdir=path_of_current_dir
worker_connections=10

#timeout=30

#max_requests=2000

#graceful_timeout=30

#loglevel='info'
reload=True
debug=False
bind="127.0.0.1:5000"
pidfile='/home/www/blog/flaskproject/logs/gun.pid'
errorlog='/home/www/blog/flaskproject/logs/gunerror.log'
accesslog='/home/www/blog/flaskproject/logs/acc.log'