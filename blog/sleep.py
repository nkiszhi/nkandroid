import time
import json
import os
from sklearn.metrics import classification_report
#mama cnn dre
model_choose=input("单模型：")
def fuckf(kk):
    if kk=="malware":
        return 0
    else :
        return 1

def ivap(find_m,find_c,find_d):  #标准
    #print((find_m,find_c,find_d))
    acc=[]
    for i in [find_m,find_c,find_d]:
        if i[2]=='accept':
            acc.append(i)
    #print(acc)
    min=100
    if acc==[]:
        return ivap3(find_m,find_c,find_d)
    for i in acc:
        s=i[1]-i[0]
        if min>s:
            min=s
    print(min)
    for i in acc:
        s=i[1]-i[0]
        if min == s:
            result=i[3]
    if result=='malware':
        return 0
    else:
        return 1

def ivap2(find_m,find_c,find_d): #统计单个模型
    #print((find_m,find_c,find_d))
    if model_choose=="m":
        find=find_m
    if model_choose=="c":
        find=find_c
    if model_choose=="d":
        find=find_d

    
    if find[0]==-1:
        return -1
    if find[3]=='malware':
        return 0
    else:
        return 1

def ivap3(find_m,find_c,find_d):#选取最小p0p1的
    #print((find_m,find_c,find_d))
    acc=[]
    for i in [find_m,find_c,find_d]:
        if i[0]!=-1 and i[1]!=-1:
            acc.append(i)
    #print(acc)
    min=100
    if acc==[]:
        return -1
    for i in acc:
        s=i[1]-i[0]
        if min>s:
            min=s
    print(min)
    for i in acc:
        s=i[1]-i[0]
        if min == s:
            result=i[3]
    if result=='malware':
        return 0
    else:
        return 1

def ivap4(find_m,find_c,find_d): #投票
    #print((find_m,find_c,find_d))
    mal=0
    good=0
    vote=[0,0]
    acc=[]
    for i in [find_m,find_c,find_d]:
        if i[0]!=-1:
            acc.append(i)
    for i in acc:
        if i[2]=="reject" and fuckf(i[3])==0:
            mal+=0.6
        if i[2]=="accept" and fuckf(i[3])==0:
            mal+=1
        if i[2]=="reject" and fuckf(i[3])==1:
            good+=0.6
        if i[2]=="accept" and fuckf(i[3])==1:
            good+=1
    print([mal,good])
    if mal==good:
        return ivap3(find_m,find_c,find_d) #ivap3(find_m,find_c,find_d)
    if mal<good:
        return 1
    if mal>good:
        return 0

def ivap_fun(find_m,find_c,find_d):
    return ivap2(find_m,find_c,find_d)



data_year=input("年份")
if data_year=="17":
    mamab="2017mama6_2.json"
    mamam="2017mama3_2.json"
    cnnb="2017CNN2.json"
    cnnm="2017CNN1.json"
    drebinb="results_17_benign.json"
    drebinm="results_17_mal.json"

    rootDir="/home/www/test_dataset/2017_benign_test100"
if data_year=="18":

    #mamab="newmama6.json"
    #mamam="newmama3.json"
    mamab="newmama6_2.json"
    mamam="newmama3_2.json"
    cnnb="newCNN2.json"
    cnnm="newCNN1.json"
    drebinb="newdrebin.json"
    drebinm="newdrebin1.json"
    rootDir="/home/www/test_dataset/2018_benign_100"
if data_year=="18x":
    mamab="mama6.json"
    mamam="mama3.json"
    cnnb="CNN2.json"
    cnnm="CNN1.json"
    drebinb="drebin.json"
    drebinm="drebin1.json"
    rootDir="/home/www/test_dataset/2018_benign_100"





"""
mamab="mama6.json"
mamam="mama3.json"
cnnb="CNN2.json"
cnnm="CNN1.json"
drebinb="drebin.json"
drebinm="drebin1.json"
"""
"""
mamab="2017mama6.json"
mamam="2017mama3.json"
cnnb="2017CNN2.json"
cnnm="2017CNN1.json"
drebinb="results_17_benign.json"
drebinm="results_17_mal.json"
"""

sumname=[]
sump0=[]
sump1=[]
sumr=[]
summ=[]
errorlist=[]


#rootDir="/home/www/test_dataset/2017_benign_test100"   #改！！！！！！！！
mal=0
good=0
fails=0
with open(mamab, 'r') as load_f:
    mamab_dict = json.load(load_f)
with open(mamam, 'r') as load_f:
    mamam_dict = json.load(load_f)
with open(cnnb, 'r') as load_f:
    cnnb_dict = json.load(load_f)
with open(cnnm, 'r') as load_f:
    cnnm_dict = json.load(load_f)
with open(drebinb, 'r') as load_f:
    drebinb_dict = json.load(load_f)
with open(drebinm, 'r') as load_f:
    drebinm_dict = json.load(load_f)
#filess=os.listdir(rootDir)
fail_=[-1,-1,'reject','malware']
for lists in os.listdir(rootDir):
    my_dict = mamab_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_m=[my_dict['p0'][index][0],my_dict['p1'][index][0],my_dict['r'][index],my_dict['m'][index]]
    if index==-1:
        find_m=fail_
    if find_m[0]==-1:
        find_m=fail_
    #find_m=fail_

    
    my_dict = cnnb_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_c=[my_dict['p0'][index][0],my_dict['p1'][index][0],my_dict['r'][index],my_dict['m'][index]]
    if index==-1:
        find_c=fail_
    if find_c[0]==-1:
        find_c=fail_
    

    my_dict = drebinb_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_d=[my_dict['p0'][index],my_dict['p1'][index],my_dict["predicts"][index][1],my_dict['predicts'][index][0]]
    if index==-1:
        find_d=fail_ 
    if find_d[0]==-1:
        find_d=fail_
    
    

    k=ivap_fun(find_m,find_c,find_d)
    if k==1:
        print("goodware")
        good+=1
    if k==0:
        print("malware")
        mal+=1
    if k==-1:
        print("检测失败:")
        print((find_m,find_c,find_d))
        fails+=1

first_s=[mal,good,fails]


#accout_t=[1]*(100-fails)
mal_fix=mal+fails
accout_t=[1]*100
accout_p=[0]*mal_fix+[1]*good


mal=0
good=0
fails=0
if data_year=="17":
    rootDir="/home/www/test_dataset/2017_malware_test100"
if data_year=="18":
    rootDir="/home/www/test_dataset/2018_malware_100"
if data_year=="18x":
    rootDir="/home/www/test_dataset/2018_malware_100"
#rootDir="/home/www/test_dataset/2017_malware_test100"   #改！！！！！！！！
for lists in os.listdir(rootDir):
    my_dict = mamam_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_m=[my_dict['p0'][index][0],my_dict['p1'][index][0],my_dict['r'][index],my_dict['m'][index]]
    if index==-1:
        find_m=fail_
    if find_m[0]==-1:
        find_m=fail_
    #find_m=fail_


    my_dict = cnnm_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_c=[my_dict['p0'][index][0],my_dict['p1'][index][0],my_dict['r'][index],my_dict['m'][index]]
    if index==-1:
        find_c=fail_
    if find_c[0]==-1:
        find_c=fail_

    my_dict = drebinm_dict
    index=my_dict['name'].index(lists) if (lists in my_dict['name']) else -1
    print(index)
    find_d=[my_dict['p0'][index],my_dict['p1'][index],my_dict["predicts"][index][1],my_dict['predicts'][index][0]]
    if index==-1:
        find_d=fail_ 
    if find_d[0]==-1:
        find_d=fail_
    


    k=ivap_fun(find_m,find_c,find_d)
    if k==1:
        print("goodware")
        good+=1
    if k==0:
        print("malware")
        mal+=1
    if k==-1:
        print("检测失败:")
        print((find_m,find_c,find_d))
        fails+=1
print(first_s)      
print([mal,good,fails])
#accout_t=accout_t+[0]*(100-fails)
good_fix=good+fails
accout_t=accout_t+[0]*100
accout_p=accout_p+[0]*mal+[1]*good_fix
#print(accout_t)
#print(accout_p)
print(classification_report(accout_t,accout_p))
'''
    p02=mamab_dict['p0']
    p12=mamab_dict['p1']
    m2=mamab_dict['predicts'][0][0]
    r2=mamab_dict['predicts'][0][1]

    with open(mamam, 'r') as load_f:
        mamab_dict = json.load(load_f)

    my_dict = mamab_dict
    p02 = mamab_dict['p0']
    p12 = mamab_dict['p1']
    m2 = mamab_dict['predicts'][0][0]
    r2 = mamab_dict['predicts'][0][1]
'''