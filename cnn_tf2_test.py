import tensorflow as tf
from data_reader import load_data
import numpy as np
from uncompress import *
import os
from ivap import IVAP
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential, layers

# slim = tf.contrib.slim
all_num = 1550
learning_rate = 0.0001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()

def one_hot(batch_size, Y):
    B = np.zeros((batch_size, 2))
    B[np.arange(batch_size), Y] = 1
    return B

def report(threshold, metrics):
    print('IVAP @ {}:'.format(threshold))
    print('\tACC: {}'.format(metrics.acc))
    print('\tTPR: {}'.format(metrics.tpr))
    print('\tFPR: {}'.format(metrics.fpr))
    print('\tTRR: {}'.format(metrics.trr))
    print('\tFRR: {}'.format(metrics.frr))
    print('\tREJ: {}'.format(metrics.rej))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    data_X, data_Y = load_data('2018_200_save')
    calib_X,calib_Y=load_data('2017_600_save')
    # indices = np.random.permutation(np.arange(data_X.shape[0]))
    # data_X = data_X[indices, :, :]
    # data_Y = data_Y[indices]
    # cnn_model = CNN(learning_rate)
    # cnn_model = cnn_model.create_model()
    batch_size = 8
    # cnn_model.build((50*batch_size,50,all_num,1))
    # print(cnn_model.summary())
    # exit(0)
    cnn_model = tf.keras.models.load_model(os.path.join('cnn_logs_4', 'last.ckpt'))
    all_acc=None
    all_sum= 0
    y_pre = []
    ivap=IVAP(cnn_model,0.,calib_X,calib_Y)
    print(ivap.F0)
    print(ivap.F1)
    p0,p1,lab=ivap.batch_predictions(images=calib_X)
    print(p0)
    print(p1)
    print(lab)
    sum=0
    rej_sum=0
    acc_sum=0
    for i in range(len(p1)):
        sum+=1
        if abs(p1[i]-p0[i])>0.008:
            rej_sum+=1
        elif lab[i][1]==calib_Y[i]:
            acc_sum+=1
    print(rej_sum/sum)
    print(acc_sum/(sum-rej_sum))



    # for step in range(int(data_Y.shape[0]/batch_size)):
    #     batch_x, batch_y = data_X[step * batch_size:(step + 1) * batch_size], \
    #                        data_Y[step * batch_size:(step + 1) * batch_size]
    #     batch_x = uncompress(batch_x, all_num)
    #     batch_y = one_hot(batch_size, batch_y)
    #     batch_y = np.repeat(batch_y, 50, axis=0)
    #     y_pred = cnn_model(batch_x)
    #     categorical_accuracy.update_state(y_true=batch_y, y_pred=y_pred)
    #     test_acc = categorical_accuracy.result()
    #     this_sum = 0
    #     for i in range(batch_size):
    #         sum_y0 = 0
    #         sum_y1 = 0
    #
    #         for j in range(50):
    #             if y_pred[i * 50 + j, 0] < y_pred[i * 50 + j, 1]:
    #                 sum_y1+=1
    #             else:
    #                 sum_y0+=1
    #         if (batch_y[i*50,0]==0 and sum_y0<sum_y1)or(batch_y[i*50,0]==1 and sum_y0>sum_y1):
    #             all_sum+=1
    #             this_sum+=1
    #         else:
    #             all_sum+=0
    #         if sum_y0 < sum_y1:
    #             y_pre.append(1)
    #         else:
    #             y_pre.append(0)
    #
    #     all_acc=all_sum/((step+1)*batch_size)
    #     this_acc=this_sum/batch_size
    #     print("step %d: test_accuracy:%f all_acc:%f this_acc:%f" % (
    #         step, test_acc,all_acc,this_acc))
    print(classification_report(data_Y[:192],y_pre))
if __name__ == "__main__":
    main()
