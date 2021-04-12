import os
import time
import sys
import json
#import importlib

sumname=[]
sump0=[]
sump1=[]
sumr=[]
summ=[]
rootDir="/home/www/nothing"
for lists in os.listdir(rootDir):
    path = os.path.join(rootDir, lists)
    if lists[-1]=='t':
        continue
    print(lists)
    print(path)
    #mamadroid_ivap
    starttime1=time.ctime(os.path.getmtime('/home/www/mamadroid_ivap/results.json'))
    tf1_command="python /home/www/mamadroid_ivap/main_mamadroid.py -f "+path
    
    os.chdir('/home/www/mamadroid_ivap')
    print(tf1_command)
    t_f = os.popen (tf1_command)
    #等半分钟
    time.sleep(30)
    i=0
    
    for i in range(40):
        time.sleep(5)
        print(i)
        mtime1 = time.ctime(os.path.getmtime('/home/www/mamadroid_ivap/results.json'))
        
        if mtime1>starttime1:
            #提取
            with open('/home/www/mamadroid_ivap/results.json','r') as load_f:
                data_dict = json.load(load_f)
        
            my_dict = data_dict
            p0=my_dict['p0']
            p1=my_dict['p1'] 
            m=my_dict['predicts'][0][0]
            r=my_dict['predicts'][0][1]

            sumname.append(lists)
            sump0.append(p0)
            sump1.append(p1)
            sumr.append(r)
            summ.append(m)
            t_f.close()
            with open('/home/www/blog/mama.json','w') as file_obj:
                json.dump({"name":sumname,"p0":sump0,"p1":sump1,"r":sumr,"m":summ},file_obj)
            
            break


        os.chdir('/home/www/blog')

    if i==39:
        print("有问题！！！")
        sumname.append("有问题！！！")
        with open('/home/www/blog/mama.json','w') as file_obj:
            json.dump({"name":sumname,"p0":sump0,"p1":sump1,"r":sumr,"m":summ},file_obj)
        t_f.close()