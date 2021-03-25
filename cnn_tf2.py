import tensorflow as tf
from data_reader import load_data
import numpy as np
from uncompress import *
import os
from tensorflow.keras import Sequential, layers
from sklearn import decomposition
pca = decomposition.PCA()

# slim = tf.contrib.slim
all_num = 1550
learning_rate = 0.0001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)



class CNN:
    def __init__(self, rate):
        self.rate = rate

    def create_model(self):
        model = Sequential()
        # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(2, activation='softmax'))
        # model.add(layers.ReLU())
        model.add(layers.Conv2D(512, (3, all_num)))
        model.add(layers.MaxPool2D((4, 1)))
        model.add(layers.Conv2D(512, (5, 1)))
        model.add(layers.MaxPool2D((4, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(128))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2,activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

def one_hot(batch_size, Y):
    B = np.zeros((batch_size, 2))
    B[np.arange(batch_size), Y] = 1
    return B


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    data_X, data_Y = load_data('2017_1000_save')


    # test_X=data_X[:4,:,]
    # test_Y=data_Y[:4]
    # data_X=data_X[4:,:,]
    # data_Y=data_Y[4:]
    cnn_model = CNN(learning_rate)
    cnn_model = cnn_model.create_model()
    batch_size = 8
    cnn_model.build((50*batch_size,50,all_num,1))
    print(cnn_model.summary())
    # exit(0)
    step = 0
    epoch_num = 3


    count = 0
    for epoch in range(epoch_num):
        indices = np.random.permutation(np.arange(data_X.shape[0]))
        data_X = data_X[indices, :, :]
        data_Y = data_Y[indices]
        categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
        all_loss=0
        for step in range(int(data_X.shape[0] / batch_size)):
            with tf.GradientTape() as tape:
                # img=[img]
                # img = uncompress(img, all_num)
                # y_pred = cnn_model(img)
                # batch_y = one_hot(1, [data_Y[count]])
                # batch_y = np.repeat(batch_y, 50, axis=0)

                batch_x, batch_y = data_X[step * batch_size:(step + 1) * batch_size], \
                                   data_Y[step * batch_size:(step + 1) * batch_size]

                batch_x=uncompress(batch_x,all_num)
                batch_y = one_hot(batch_size, batch_y)
                batch_y = np.repeat(batch_y, 50, axis=0)
                y_pred = cnn_model(batch_x)

                loss = tf.keras.losses.mean_squared_error(y_true=batch_y, y_pred=y_pred)
                count += 1
                loss = tf.reduce_mean(loss)
                all_loss += loss

                categorical_accuracy.update_state(y_true=batch_y, y_pred=y_pred)
                train_acc = categorical_accuracy.result()
            grads = tape.gradient(loss, cnn_model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, cnn_model.variables))
            if True or step!=int(data_X.shape[0] / batch_size)-1:
                print("step %d: loss:%f,train_accuracy:%f" % (
                    step, loss.numpy(), train_acc))
            if count % 20 == 0:
                print('saving checkpoint')
                cnn_model.save(os.path.join('cnn_logs_5050', 'epoch' + str(epoch) + \
                                                          'i' + str(count) + '.ckpt'), save_format='tf')
                print("Model saved")
        avg_loss=all_loss/int(data_X.shape[0]/batch_size)
        print("epoch:"+str(avg_loss))
        if all_loss/int(data_X.shape[0]/batch_size) < 0.02:
            break
    print('saving checkpoint')
    cnn_model.save(os.path.join('cnn_logs_4', 'last' + '.ckpt'), save_format='tf')
    print("Model saved")
    # categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    # y_pre=cnn_model(uncompress(test_X,all_num))
    # test_Y=one_hot(4,test_Y)
    # test_Y = np.repeat(test_Y, 50, axis=0)
    # categorical_accuracy.update_state(test_Y,y_pre)
    # test_acc=categorical_accuracy.result()
    # print("test_accuracy:%f "%(test_acc))



if __name__ == "__main__":
    main()
