import os
import time
import sys
import json
#import importlib
filecout=0
sumname=[]
sump0=[]
sump1=[]
sumr=[]
summ=[]
errorlist=[]
#rootDir="/home/www/test_dataset/2017_benign_test100"
rootDir="/home/www/test_dataset/2018_benign_100"
target_path='/home/www/mamadroid_ivap/'   #改
result_jsonfile=target_path+'results.json' #改
#filess=os.listdir(rootDir)
for lists in os.listdir(rootDir):
    path = os.path.join(rootDir, lists)
    if lists[-1]!='k':
        continue
    print(lists)
    print(path)
    #mamadroid_ivap
    starttime1=time.ctime(os.path.getmtime(result_jsonfile))
    tf1_command="python /home/www/mamadroid_ivap/main_mamadroid.py -f "+path    #改
    
    os.chdir(target_path)
    print(tf1_command)
    filecout+=1
    print(filecout)
    with os.popen (tf1_command) as t_f:
        read=t_f.read()
    #等半分钟
    #time.sleep(30)
    i=0
    
    
    mtime1 = time.ctime(os.path.getmtime(result_jsonfile))
        
    if mtime1>starttime1:
        #提取
        with open(result_jsonfile,'r') as load_f:
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
        
        with open('/home/www/blog/newmama6_2.json','w') as file_obj:   #改
            json.dump({"name":sumname,"p0":sump0,"p1":sump1,"r":sumr,"m":summ},file_obj)
        
        


    

    else:
        print(path+"有问题！！！")
        errorlist.append(path+"有问题！！！")
        with open('/home/www/mamaerror2.json','w') as file_obj:  #改
            json.dump({"name":errorlist},file_obj)
        

    os.chdir('/home/www/blog')