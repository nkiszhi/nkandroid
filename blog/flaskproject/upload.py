from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask import Flask, render_template,request,redirect,url_for
#from werkzeug.utils import secure_filename
import json
import os
#import sys
import time

#from main_mamadroid import main #scriptName without .py extension

#with open('/home/www/mamadroid_ivap/results.json','r') as load_f:
#    data_dict = json.load(load_f)
def my_print(str):
    try:
        print(str)
    except:
        pass
    else:
        pass
    
my_print("访问网址：59.110.71.49:8000")
app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] =os.getcwd() #os.getcwd()'/home/www/mamadroid_ivap/'

photos = UploadSet('photos',['apk'])
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB
cur_apk="miku"

def getcolor(r):
    if r=="reject":
        return "danger"
    if r=="accept":
        return "success"
    

class UploadForm(FlaskForm):
    photo = FileField(validators=[
        FileAllowed(photos, u'只能上传apk！'), 
        FileRequired(u'文件未选择！')], render_kw = { 'style':'width:65%;' })#, render_kw = { 'style':'width:290px;' }
    submit = SubmitField(u'上传',render_kw = { 'style':'width:30%;' })

#mama cnn dre


find_fail=[-1,-1,'reject','malware']

def ivap(find_m,find_c,find_d):  #标准
    #my_print((find_m,find_c,find_d))
    acc=[]
    for i in [find_m,find_c,find_d]:
        if i[2]=='accept':
            acc.append(i)
    #my_print(acc)
    min=100
    if acc==[]:
        return ivap3(find_m,find_c,find_d)
    for i in acc:
        s=i[1]-i[0]
        if min>s:
            min=s
    my_print(min)
    for i in acc:
        s=i[1]-i[0]
        if min == s:
            result=i[3]
    if result=='malware':
        return 0
    else:
        return 1


def ivap3(find_m,find_c,find_d):#选取最小p0p1的
    #my_print((find_m,find_c,find_d))
    acc=[]
    for i in [find_m,find_c,find_d]:
        if i[0]!=-1 and i[1]!=-1:
            acc.append(i)
    #my_print(acc)
    min=100
    if acc==[]:
        return -1
    for i in acc:
        s=i[1]-i[0]
        if min>s:
            min=s
    my_print(min)
    for i in acc:
        s=i[1]-i[0]
        if min == s:
            result=i[3]
    if result=='malware':
        return 0
    else:
        return 1

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    tar_page='index2.html'
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        cur_apk=filename
        my_print(cur_apk)
        file_url ="已保存" #photos.url(filename)
        target="'/home/www/blog/flaskproject/"+filename+"'"
        tf1_command="python /home/www/mamadroid_ivap/main_mamadroid.py -f "+target
        tf2_command="python /home/www/CNN/main_cnn.py -f "+target
        tf3_command="python /home/www/drebin/drebin/classification_drebin.py -f "+target
        my_print(tf1_command)
        my_print(tf2_command)
        my_print(tf3_command)
        
        filename = 'uploadapk.json'
        curtime=time.ctime(time.time())
        with open(filename, 'w') as file_obj:
            json.dump({'apk': cur_apk, 'time': curtime}, file_obj)
        
        #执行模型。。    


        with os.popen (tf3_command) as t_f3:
            read=t_f3.read()
        my_print("结束")


        #mamadroid_ivap
        os.chdir('/home/www/mamadroid_ivap')
        t_f = os.popen (tf1_command)
        #为了稳一点，暂停1秒先
        time.sleep(1)
        
        #CNN_ivap
        os.chdir('/home/www/CNN')
        t_f2 = os.popen (tf2_command)
        #为了稳一点，暂停1秒先
        #time.sleep(1)
        
        
        #derbin
        #os.chdir('/home/www/drebin/drebin')
        #t_f3 = os.popen (tf3_command)
        os.chdir('/home/www/blog/flaskproject')
        

        
    else:
        file_url = "欢迎使用，请上传文件后查看结果"
        tar_page='index.html'
    
    return render_template(tar_page, form=form, file_url=file_url)



@app.route('/result',methods = ['POST', 'GET'])
def result():
    with open('/home/www/mamadroid_ivap/results.json','r') as load_f:
        data_dict = json.load(load_f)
    mtime = time.ctime(os.path.getmtime('/home/www/mamadroid_ivap/results.json'))
    my_dict = data_dict
    p0=my_dict['p0']
    p1=my_dict['p1']
    m=my_dict['predicts'][0][0]
    r=my_dict['predicts'][0][1]
    color=getcolor(r)
    find_m=[p0[0],p1[0],r,m]
    if find_m[0]==-1:
        find_m=find_fail


    with open('/home/www/CNN/results.json','r') as load_f:
        data_dict2 = json.load(load_f)
    mtime2 = time.ctime(os.path.getmtime('/home/www/CNN/results.json'))
    my_dict = data_dict2
    p02=my_dict['p0']
    p12=my_dict['p1']
    m2=my_dict['predicts'][0][0]
    r2=my_dict['predicts'][0][1]
    color2=getcolor(r2)
    find_c=[p02[0],p12[0],r2,m2]
    if find_c[0]==-1:
        find_c=find_fail

    with open('/home/www/drebin/drebin/results_of_destination_file.json','r') as load_f:
        data_dict3 = json.load(load_f)
    mtime3 = time.ctime(os.path.getmtime('/home/www/drebin/drebin/results_of_destination_file.json'))
    my_dict = data_dict3
    p03=my_dict['p0']
    p13=my_dict['p1']
    m3=my_dict['predicts'][0][0]
    r3=my_dict['predicts'][0][1]
    color3=getcolor(r3)
    find_d=[p03[0],p13[0],r3,m3]
    
    
    
    
    
   
    #find_d=find_fail
    
    #选择最终结果
    sss=ivap(find_m,find_c,find_d)
    if sss==-1:
        final=["info","检测失败"]
    if sss==1:
        final=["success","非恶意软件"]
    if sss==0:
        final=["danger","恶意软件"]

    #与上传时间比较
    with open('/home/www/blog/flaskproject/uploadapk.json','r') as load_f:
        updata = json.load(load_f)
    my_dict=updata
    apkname=my_dict['apk']
    up_time=my_dict['time']
    target_page="result.html"
    
    
    my_print((up_time,mtime,mtime2,mtime3))
    if up_time>mtime or up_time>mtime2 or up_time>mtime3:
        my_print("\033[1;31m wait!!!!!!\033[0m\n")
        target_page="wait.html"

    
    return render_template(target_page,p0 = p0,p1=p1,mtime=mtime,m=m,r=r, p02 = p02,p12=p12,mtime2=mtime2,mtime3=mtime3,m2=m2,r2=r2,r3=r3,m3=m3,p03=p03,p13=p13,cur_apk=cur_apk,apkname=apkname,up_time=up_time,color=color,color2=color2,color3=color3,final=final )

#后门
@app.route('/nextdoor',methods = ['POST', 'GET'])
def nextdoor():
    with open('/home/www/mamadroid_ivap/results.json','r') as load_f:
        data_dict = json.load(load_f)
    mtime = time.ctime(os.path.getmtime('/home/www/mamadroid_ivap/results.json'))
    my_dict = data_dict
    p0=my_dict['p0']
    p1=my_dict['p1']
    m=my_dict['predicts'][0][0]
    r=my_dict['predicts'][0][1]
    color=getcolor(r)
    find_m=[p0[0],p1[0],r,m]
    if find_m[0]==-1:
        find_m=find_fail

    with open('/home/www/CNN/results.json','r') as load_f:
        data_dict2 = json.load(load_f)
    mtime2 = time.ctime(os.path.getmtime('/home/www/CNN/results.json'))
    my_dict = data_dict2
    p02=my_dict['p0']
    p12=my_dict['p1']
    m2=my_dict['predicts'][0][0]
    r2=my_dict['predicts'][0][1]
    color2=getcolor(r2)
    find_c=[p02[0],p12[0],r2,m2]
    if find_c[0]==-1:
        find_c=find_fail

    with open('/home/www/drebin/drebin/results_of_destination_file.json','r') as load_f:
        data_dict3 = json.load(load_f)
    mtime3 = time.ctime(os.path.getmtime('/home/www/drebin/drebin/results_of_destination_file.json'))
    my_dict = data_dict3
    p03=my_dict['p0']
    p13=my_dict['p1']
    m3=my_dict['predicts'][0][0]
    r3=my_dict['predicts'][0][1]
    color3=getcolor(r3)
    find_d=[p03[0],p13[0],r3,m3]
    
    
    
    
    
   
    #find_d=find_fail
    
    #选择最终结果
    sss=ivap(find_m,find_c,find_d)
    if sss==-1:
        final=["info","检测失败"]
    if sss==1:
        final=["success","非恶意软件"]
    if sss==0:
        final=["danger","恶意软件"]

    #与上传时间比较
    with open('/home/www/blog/flaskproject/uploadapk.json','r') as load_f:
        updata = json.load(load_f)
    my_dict=updata
    apkname=my_dict['apk']
    up_time=my_dict['time']
    target_page="result.html"
    

    
    return render_template(target_page,p0 = p0,p1=p1,mtime=mtime,m=m,r=r, p02 = p02,p12=p12,mtime2=mtime2,mtime3=mtime3,m2=m2,r2=r2,r3=r3,m3=m3,p03=p03,p13=p13,cur_apk=cur_apk,apkname=apkname,up_time=up_time,color=color,color2=color2,color3=color3,final=final )



if __name__ == '__main__':
    app.run()