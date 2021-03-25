import sys
# sys.path.append('./androguard')
# sys.path.append('/usr/lib/python3/dist-packages')
# sys.path.append('/home/vikram_mm/.local/lib/python3.5/site-packages/')

from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import numpy as np
import re
import pickle as cPickle
import os

max_h = 100
max_calls = 50


def gen_dict(list_path):
    file = open("error_files.txt", "w")
    external_api_dict = {}
    path_idx = 0
    for path1 in list_path:
        count = 0
        for f in os.listdir(path1):
            if path_idx==0 and len(external_api_dict)>=1000:
                path_idx += 1
                break
            elif path_idx==1 and len(external_api_dict)>=1500:
                break
            # print count
            count += 1
            if (count == 300):
                break
            path = os.path.join(path1, f)
            print(path)
            # try:
            if path.endswith('.apk'):
                app = apk.APK(path)
                app_dex = dvm.DalvikVMFormat(app.get_dex())
            else:
                app_dex = dvm.DalvikVMFormat(open(path, "rb").read())
            app_x = analysis.Analysis(app_dex)

            methods = []
            cs = [cc.get_name() for cc in app_dex.get_classes()]

            ctr = 0
            # print len(app_dex.get_methods())
            for method in app_dex.get_methods():
                g = app_x.get_method(method)

                if method.get_code() == None:
                    continue

                for i in g.get_basic_blocks().get():

                    for ins in i.get_instructions():
                        # This is a string that contains methods, variables, or
                        # anything else.

                        output = ins.get_output()

                        # match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
                        match = re.findall(r'(L[^;^$]*;)', output)
                        if match not in cs:
                            # methods.append(match.group())
                            # print "instruction : ", ins.get_basic_blocks()

                            # print "external api detected: ", match.group()
                            for ma in match:
                                api_candidates = ["Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/",
                                                  "Ljavax/",
                                                  "Lorg/apache/",
                                                  "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/",
                                                  "Ljunit/"]
                                if  ma not in external_api_dict and ma not in cs:
                                    for candidate in api_candidates:
                                        if ma.startswith(candidate):
                                            external_api_dict[ma] = len(external_api_dict)
                                            break
            # except:
            #     print("error")
            #     file.write(path)
            print(len(external_api_dict))
    file.close()
    return external_api_dict


if __name__ == '__main__':
    common_dict = gen_dict(["2017_dataset/malware", "2017_dataset/benign"])

    fp = open('common_dict' + '.save', 'wb')
    cPickle.dump(common_dict, fp, protocol=cPickle.HIGHEST_PROTOCOL)
    fp.close()

    print(len(common_dict))
