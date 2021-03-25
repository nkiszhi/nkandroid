import sys

sys.path.append('/home/www/drebin/drebin/binary-ivap')
sys.path.append('/home/www/drebin/drebin')
import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as CountV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score
from sklearn.utils import shuffle
import CommonModules as CM
import sys
import argparse
import PScoutMapping as PScoutMapping
import psutil
import keras
from GetApkData import GetApkData
from GetApkData import ProcessingDataForGetApkData
from scipy.integrate import trapz

from tqdm import tqdm

from ivap import IVAP

from VennABERS import getFVal

"""
This script performs Drebin's classification using SVM LinearSVC classifier.
Inputs are described in parseargs function.
Recall and Accuracy are calculated 10 times. Each experiment is performed with 66% of training set and 33% of test set, and scores are averaged.
Roc curve is plotted using the last trained classifier from the 10 experiments.
The Outputs are the results text file, and roc curve pdf graph that are located in the drebin directory.
"""
BATCH_SIZE = 128
EPOCHS = 50


def normalize(arr):
    min_val, max_val = arr.min(), arr.max()
    arr -= min_val
    arr /= max_val - min_val
    return arr

def get_file_name(path):
    list_files=[]
    i=0
    for root, dirs, files in os.walk(path):
        for file in files:
            (filename, extension) = os.path.splitext(file)
            if (extension == '.data'):
                list_files.append(filename+'.apk')
    return list_files


def report(threshold, metrics):
    print('IVAP @ {}:'.format(threshold))
    print('\tACC: {}'.format(metrics.acc))
    print('\tTPR: {}'.format(metrics.tpr))
    print('\tFPR: {}'.format(metrics.fpr))
    print('\tTRR: {}'.format(metrics.trr))
    print('\tFRR: {}'.format(metrics.frr))
    print('\tREJ: {}'.format(metrics.rej))


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sd",
        "--seed",
        help=
        "(Optional) The random seed used for the experiments (useful for 100% identical results replication)",
        type=int,
        required=False,
        default=np.random.randint(0, 2 ** 16 - 1)
    )
    parser.add_argument("-f", "--file",
                        default='/home/www/dataset/2018_malware_100/3f453025259fcbb3f2e898618b6147d4ffc90182ae8ede9658017e8c889a3fc0.apk',
                        help="APK file to analyze or a directory with more than one APK to analyze.",
                        type=str)
    args = parser.parse_args()
    return args


def Classification(MalwarePath, GoodwarePath, Dst_Path, file_scores, ROC_GraphName):
    Malware = MalwarePath
    Goodware = GoodwarePath

    # the list of absolute paths of the ".data" malware files
    AllMalSamples = CM.ListFiles(MalwarePath, Extension)
    AllMalCalib = CM.ListFiles('/home/www/drebin/drebin/testdata_mal_300_calib',Extension)
    AllMalTest = CM.ListFiles('/home/www/drebin/drebin/testdata_2018_mal_100',Extension)
    IvapMalTest = CM.ListFiles('/home/www/drebin/drebin/new_testdata_mal', Extension)
    Dst_data_file = '/home/www/drebin/drebin/testdata/' + os.path.basename(Dst_Path).split('.')[0] + '.data'
    if os.path.exists(Dst_data_file) == False:
        Dst_Store = '/home/www/drebin/drebin/testdata/'
        ApkDirectoryPath = os.path.dirname(Dst_Path)
        ApkFile = Dst_Path
        ProcessingDataForGetApkData(ApkDirectoryPath,Dst_Store,ApkFile)

    #cpu_count=psutil.cpu_count()
    #ApkFile = Dst_Path

    #GetApkData(cpu_count,Dst_Store,ApkFile)

    # the list of absolute paths of the ".data" goodware files
    AllGoodSamples = CM.ListFiles(GoodwarePath, Extension)
    AllGoodCalib = CM.ListFiles('/home/www/drebin/drebin/testdata_benign_300_calib',Extension)
    DstSample = [Dst_data_file]
    AllGoodTest = CM.ListFiles('/home/www/drebin/drebin/testdata_2018_benign_100',Extension)
    IvapGoodTest = CM.ListFiles('/home/www/drebin/drebin/new_testdata_good', Extension)
    AllSampleNames = AllMalSamples + AllGoodSamples  # combine the two lists
    AllCalibNames=AllMalCalib + AllGoodCalib
    AllTestNames=AllMalTest + AllGoodTest
    FeatureVectorizer = CountV(
        input='filename',
        tokenizer=lambda x: x.split('\n'),
        token_pattern=None,
        binary=True
    )
    Good_labels = np.ones(len(AllMalSamples))  # label malware as 1
    Mal_labels = np.empty(len(AllGoodSamples))
    Good_labels1 = np.ones(len(AllMalCalib))  # label malware as 1
    Mal_labels1 = np.empty(len(AllGoodCalib))
    Good_labels2 = np.ones(len(AllMalTest))  # label malware as 1
    Mal_labels2 = np.empty(len(AllGoodTest))
    Mal_labels.fill(0)  # label goodware as -1
    Mal_labels1.fill(0)
    Mal_labels2.fill(0)
    # concatenate the two lists of labels
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    y1 = np.concatenate((Mal_labels1, Good_labels1), axis=0)
    y2 = np.concatenate((Mal_labels2, Good_labels2), axis=0)
    scoresRec, scoresAcc = [], []  # empty lists to store recall and accuracy
    # The experiment is run 10 times
    # for i in range(10):
    # shuffle the list in each experiment
    AllSampleNames, y = shuffle(AllSampleNames, y)
    AllCalibNames, y1 = shuffle(AllCalibNames, y1)
    AllTestNames, y2 = shuffle(AllTestNames, y2)
    # split the lists into 66% for training and 33% for test.
    x_train, y_train = AllSampleNames, y
    x_test, y_test = AllTestNames, y2
    x_calib, y_calib= AllCalibNames, y1
    x_dst= DstSample
    # learn the vocabulary dictionary from the training set
    x_train = FeatureVectorizer.fit_transform(x_train)
    # transform the test set using vocabulary learned
    x_test = FeatureVectorizer.transform(x_test)
    x_calib = FeatureVectorizer.transform(x_calib)
    x_dst = FeatureVectorizer.transform(x_dst)
    x_ivap_mal = FeatureVectorizer.transform(IvapMalTest)
    x_ivap_good = FeatureVectorizer.transform(IvapGoodTest)

    clf2 = LinearSVC(max_iter=2000)  # create the classifier
    clf2.fit(x_train, y_train)  # fit (=train) the training data
    y_predict = clf2.predict(x_test)  # predict the labels of test data
    target_names = ['class1','class2']
    print(classification_report(y2.tolist(), y_predict.tolist(),target_names=target_names))

    clf = LinearSVC()  # create the classifier
    # idx_train = np.logical_or(y_train == 0, y_train == 1)
    # idx_test = np.logical_or(y_test == 0, y_test == 1)

    # x_train, y_train = x_train[idx_train], y_train[idx_train]
    # x_test, y_test = x_test[idx_test], y_test[idx_test]

    # convert class vectors to binary class matrices

    x_train = normalize(x_train.astype(np.float64))
    x_calib = normalize(x_calib.astype(np.float64))
    x_test = normalize(x_test.astype(np.float64))
    x_dst = normalize(x_dst.astype(np.float64))
    x_ivap_mal = normalize(x_ivap_mal.astype(np.float64))
    x_ivap_good = normalize(x_ivap_good.astype(np.float64))
    # splits for proper training, proper testing, calibration and validation sets
    x_proper_train, y_proper_train = x_train, y_train

    idx = int(.8 * x_test.shape[0])
    x_proper_test, y_proper_test = x_test, y_test
    x_valid, y_valid = x_calib, y_calib

    print('Training samples: {}'.format(x_proper_train.shape))
    print('Calibration samples: {}'.format(x_calib.shape))
    print('Test samples: {}'.format(x_test.shape))
    print('Validation samples: {}'.format(x_valid.shape))

    clf.fit(x_train, y_train)

    acc = clf.score(x_proper_test, y_proper_test)
    print('Proper test accuracy: {}'.format(acc))

    betas = np.linspace(0, 1, 1000)
    points = []
    ivap = IVAP(clf, 0., x_calib, y_calib)
    print(ivap.F0)
    print(ivap.F1)
    y_opt = -np.inf
    y_val = 0.
    threshold = 0.023
    ivap.beta=threshold
    # for beta in tqdm(betas):
    #     ivap.beta = beta
    #     metrics = ivap.evaluate(x_valid, y_valid)
    #     points.append((metrics.frr, metrics.trr))

    #     y = metrics.trr - metrics.frr
    #     if y > y_opt and beta > 0:
    #         y_opt = y
    #         threshold = beta
    #         y_val = metrics.frr
    # print(threshold)
    # ivap.beta=threshold
    points = np.array(sorted(points, key=lambda p: p[0]))
    print('Optimal threshold: {}'.format(threshold))

    p0_d, p1_d, predicts = ivap.batch_predictions(x_dst)
    print('the result of the destnation file:')
    print(p0_d)
    print(p1_d)
    print(predicts)

    # auc = trapz(points[:, 1], points[:, 0])

    # plt.title('AUC: {}'.format(auc))
    # plt.plot(points[:, 0], points[:, 1], color='blue', label='ROC')
    # plt.plot([0, 1], [0, 1], color='orange', label='chance')
    # plt.axvline(x=y_val, label='threshold', color='black', ls='dotted')
    # plt.xlabel('FRR')
    # plt.ylabel('TRR')
    # plt.legend()
    # plt.show()

    #ivap.beta = threshold
    y_p_t=keras.utils.to_categorical(y_proper_test, 2)
    metrics = ivap.evaluate(x_proper_test, y_p_t)
    report(threshold, metrics)

    # calculate the accuracy score
    scoreAcc = accuracy_score(y_test, y_predict)
    # store the calculated recall in the list of recall scores
    #scoresRec.append(scoreRec)
    # store the calculated accuarcy in the list of accuracy scores
    #scoresAcc.append(scoreAcc)

    p0_d = p0_d.tolist()
    p1_d = p1_d.tolist()
    for i in range(len(predicts)):
        new = []
        if predicts[i][0] == 1:
            new.append("malware")
        else:
            new.append("goodware")
        if predicts[i][2] == 1:
            new.append("reject")
        else:
            new.append("accept")
        predicts[i] = new

    filename = '/home/www/drebin/drebin/results_of_destination_file.json'
    #test_mal_name=get_file_name('/home/www/drebin/drebin/new_testdata_mal')
    #test_good_name = get_file_name('/home/www/drebin/drebin/new_testdata_good')
    #with open(filename, 'w') as file_obj:
        #json.dump({'name':test_good_name, 'p0': p0_d, 'p1': p1_d, 'predicts': predicts}, file_obj)
    with open(filename, 'w') as file_obj:
        json.dump({'p0': p0_d, 'p1': p1_d, 'predicts': predicts}, file_obj)



    # 上次缩进
    # write the results into the results-run.txt file
    outp = open(file_scores, "w")
    outp.write(
        "Recall scores of classifiaction with 0.66 training and 0.33 test\n"
    )
    outp.write(str((scoresRec)) + "\n")
    outp.write(str((np.mean(scoresRec))) + "\n")  # store the mean of the recall
    outp.write(
        "Accuracy scores of classifiaction with 0.66 training and 0.33 test\n"
    )
    outp.write(str((scoresAcc)) + "\n")  # store the mean of the accuracy
    outp.write(str((np.mean(scoresAcc))) + "\n")
    outp.close()

    # last classifier of the 10 experiments is chosen for the plot of ROC curve, and used to predict confidence scores
    y_score = clf.decision_function(x_test)
    # create empty dictionaries for false positive rate, true positive rate
    fpr, tpr = dict(), dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)  # compute the ROC
    plt.figure(figsize=(5, 4))
    plt.title('Receiver Operating Characteristic')
    plt.grid(color='k', linestyle=':')
    plt.plot(fpr, tpr, '-k')  # plot the roc curve
    plt.xlim([0, 0.1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # save the roc curve in roc.pdf file
    plt.savefig(ROC_GraphName + ".pdf", bbox_inches='tight')
    plt.show()

Extension = ".data"  # features files that are generated by GetApkData.py script ends with ".data"

if __name__ == '__main__':
    Args = parseargs()  # retrieve the parameters
    MalwarePath = '/home/www/drebin/drebin/testdata_malware_500'
    GoodwarePath = '/home/www/drebin/drebin/testdata_benign_500'
    file_scores = 'file_scores'
    ROC_GraphName = 'file_roc'
    SEED = Args.seed
    Dst_File = Args.file
    np.random.seed(SEED)
    Classification(MalwarePath, GoodwarePath, Dst_File, file_scores, ROC_GraphName)
