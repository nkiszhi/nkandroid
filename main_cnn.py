import tensorflow as tf
from data_reader import load_data
from uncompress import *
import os
from ivap import IVAP
import pickle
from extract_all_features_compressed import get_compressed_feature_vector
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    parser.add_argument("-f", "--file", default='malware02.apk',
                        help="APK file to analyze or a directory with more than one APK to analyze.",
                        type=str)
    args = parser.parse_args()
    return args


def apk2vector(path):
    print("loading dict...")
    external_api_dict = pickle.load(open("common_dict.save", "rb"))
    print("done!")
    feature = get_compressed_feature_vector(path, external_api_dict)
    return feature


def main():
    try:
        args = parseargs()
        test_apk = args.file
        test_apk='malware02.apk'
        test_X = apk2vector(test_apk)
        calib_X, calib_Y = load_data('2017_600_save')
        cnn_model = tf.keras.models.load_model(os.path.join('cnn_logs_4', 'last' + '.ckpt'))
        model_ivap = IVAP(cnn_model, 0., calib_X, calib_Y)
        model_ivap.beta = 0.04
        p0, p1, predicts = model_ivap.batch_predictions(np.asarray([test_X]))
        print(p0)
        print(p1)
        print(predicts)
        p0 = p0.tolist()
        p1 = p1.tolist()
        predicts = predicts.tolist()
    except:
        p0=[-1]
        p1=[-1]
        predicts=[1,0,0]
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

    filename = 'results.json'
    with open(filename, 'w') as file_obj:
        json.dump({'p0': p0, 'p1': p1, 'predicts': predicts}, file_obj)


if __name__ == '__main__':
    main()
