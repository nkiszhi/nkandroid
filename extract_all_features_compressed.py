import sys

sys.path.append('/home/vikram_mm/.local/lib/python3.5/site-packages/')

from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from androguard.core.androconf import load_api_specific_resource_module
import numpy as np
import re
import pickle as cPickle
import os

max_h = 50
max_calls = 50
seq_no = 0


def extract_all_features():
    print("loading dict...")
    external_api_dict = cPickle.load(open("common_dict.save", "rb"))
    print("done!")

    # X = []if __name__ if __name__ == '__main__':== '__main__':
    # Y = []

    # path_list = ["Dataset/benign","Dataset/all_drebin"]
    path_list = ["F:/test_dataset/2017_malware_300_calib/dataset", "F:/test_dataset/2017_benign_300_calib/dataset"]
    index = 0
    for i in range(2):
        count = 0
        for path in os.listdir(path_list[i])[::-1]:
            count += 1
            index += 1
            print(count, os.path.join(path_list[i], path))
            # X.append(get_feature_vector(os.path.join(path_list[i],path), external_api_dict))
            # Y.append(i)
            try:
                x = get_compressed_feature_vector(os.path.join(path_list[i], path), external_api_dict)
            except:
                continue
            # print x.shape
            # print x
            # exit(0)
            if not isinstance(x,np.ndarray):
                continue
            data_point = {}
            data_point['x'] = x
            data_point['y'] = i
            # fp = open(os.path.join('features',str(index) + '.save'), 'wb')
            # fp = open(os.path.join('all_compressed_features',str(path) + '.save'), 'wb')
            fp = open(os.path.join('2017_600_save', str(path) + '.save'), 'wb')
            cPickle.dump(data_point, fp, protocol=cPickle.HIGHEST_PROTOCOL)
            fp.close()
        # count=count-1

def is_system_api(api0):
    api_candidates = ["Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/",
                      "Lorg/apache/",
                      "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/"]
    for candidate in api_candidates:
        if api0.startswith(candidate):
            return True
    return False

def get_compressed_feature_vector(path, external_api_dict):
    flag = 0
    result = None

    call_no = 0
    seq_no = 0
    if path.endswith('.apk'):
        app = apk.APK(path)
        app_dex = dvm.DalvikVMFormat(app.get_dex())
    else:
        app_dex = dvm.DalvikVMFormat(open(path, "rb").read())
    app_x = analysis.Analysis(app_dex)
    cs = [cc.get_name() for cc in app_dex.get_classes()]
    # print len(app_dex.get_methods())
    for method in app_dex.get_methods():
        g = app_x.get_method(method)
        if method.get_code() == None:
            continue
        # print "***********"
        # print "method beeing investigated - ", g
        i_list=[]
        for i in g.get_basic_blocks().get():
            if isinstance(result,np.ndarray) and result.shape[0]>max_h:
                break
            # print "i.childs : " ,i.childs
            if i.childs != [] and i.name not in i_list:
                i_list.append(i.name)
                feature_vector = np.zeros((1, max_calls), dtype=int)
                call_no = 0
                for ins in i.get_instructions():
                    output = ins.get_output()
                    match = re.findall(r'(L[^;^$]*;)', output)
                    for ma in match:
                        if ma not in cs and is_system_api(ma):
                            un_match = None
                            if ma in external_api_dict:
                                un_match = external_api_dict[ma]
                            else:
                                un_match = len(external_api_dict)
                            if call_no >= max_calls:
                                if un_match == feature_vector[-1, -1]:
                                    break
                                new_feature_vector = np.zeros((1, max_calls), dtype=int)
                                feature_vector = np.row_stack((feature_vector, new_feature_vector))
                                call_no = 0
                            if call_no!=0 and un_match==feature_vector[-1,call_no-1]:
                                break
                            feature_vector[-1, call_no] = un_match
                            call_no += 1
                            break
                        if feature_vector.shape[0] >=max_h:
                                break


                # for ins in i.get_instructions():
                #     # This is a string that contains methods, variables, or
                #     # anything else.
                #     output = ins.get_output()
                #     match = re.findall(r'(L[^;^$]*;)', output)
                #     for ma in match:
                #         if ma not in cs and call_no<max_calls:
                #             feature_vector[seq_no, call_no] = external_api_dict[ma]
                #             call_no+=1
                # if match and match.group(1) not in cs and call_no < max_calls:
                # print "instruction : ", ins.get_basic_blocks()
                # print "output : ", output
                # print "external api detected: ", match.group()

                # if(i.childs!=[]):
                # print "-------->",i.childs[0][2].childs
                # break
                # feature_vector[seq_no, call_no] = external_api_dict[match.group()]
                # call_no += 1

                # rand_child_selected = np.random.randint(len(i.childs))
                # print rand_child_selecte
                # traverse_graph(i.childs[rand_child_selected][2], feature_vector, cs, call_no, external_api_dict)
                feature_vector = traverse_graph(i.childs[0][2], feature_vector, cs, call_no,  external_api_dict,[])
                if isinstance(feature_vector,np.ndarray):
                    feature_vector=np.unique(feature_vector,axis=0)
                    if isinstance(result,np.ndarray):
                        result=np.row_stack((result,feature_vector))
                        result=np.unique(result,axis=0)
                    else:
                        result=feature_vector
                    if result.shape[0]>=max_h:
                        return result[:max_h,:]
                else:
                    continue
    if not isinstance(result,np.ndarray):
        return None
    for i in range(result.shape[0],max_h):
        result=np.row_stack((result,result[i%result.shape[0]]))
    return result[:max_h,:]


def fill_feature(feature_vector):
    for i in range(len(feature_vector[-1])):
        if feature_vector[-1][i] == 0:
            if i == 0:
                return None
            for j in range(i, len(feature_vector[-1])):
                feature_vector[-1][j] = feature_vector[-1][j % i]
            return feature_vector
    return feature_vector


def traverse_graph(node, feature_vector, cs, call_no, external_api_dict, node_list):
    finish_flag = False
    # if not isinstance(feature_vector,np.ndarray):
    #     print(call_no)
    #     exit(0)
    node_list.append(node.name)
    for ins in node.get_instructions():
        name = ins.get_name()
        if name.startswith("return-"):
            finish_flag = True
        output = ins.get_output()
        match = re.findall(r'(L[^;^$]*;)', output)
        for ma in match:
            api_candidates = ["Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/",
                              "Lorg/apache/",
                              "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/"]
            if ma not in cs:
                if is_system_api(ma):
                    un_match = None
                    if ma in external_api_dict:
                        un_match = external_api_dict[ma]
                    else:
                        un_match = len(external_api_dict)
                    if call_no >= max_calls:
                        if feature_vector[-1][-1]==un_match:
                            continue
                        new_feature_vector = np.zeros((1, max_calls), dtype=int)
                        feature_vector = np.row_stack((feature_vector, new_feature_vector))
                        call_no = 0
                    if call_no == 0 or feature_vector[-1, call_no - 1] != un_match:
                        feature_vector[-1, call_no] = un_match
                        call_no += 1
                    break

        # if match and match.group(1) not in cs and call_no < max_calls:
        #     feature_vector[seq_no, call_no] = external_api_dict[match.group()]
        #     call_no += 1
    if node.childs != [] and feature_vector.shape[0] < 5:
        feature_vector0 = None
        for i in range(len(node.childs)):
            if node.childs[i][2].name in node_list:
                continue
            new_feature_vector = traverse_graph(node.childs[i][2], feature_vector, cs, call_no, external_api_dict,
                                                node_list)
            if isinstance(new_feature_vector, np.ndarray):
                if finish_flag == False:
                    feature_vector0 = new_feature_vector
                    finish_flag = True
                else:
                    feature_vector0 = np.row_stack((feature_vector0, new_feature_vector))
            else:
                continue
        if finish_flag == True:
            feature_vector = fill_feature(feature_vector0)
            return feature_vector
        else:
            return None
    else:
        if finish_flag == True:
            if feature_vector.shape[0] == 1 and call_no == 0:
                return None
            feature_vector = fill_feature(feature_vector)
            return feature_vector
        else:
            return None


def main():
    """
    For test
    """


if __name__ == '__main__':
    # x,y = load_data()
    extract_all_features()

    '''print x.shape
    print y.shape
  
    np.save('x200.npy', x)
    np.save('y200.npy', y)'''
