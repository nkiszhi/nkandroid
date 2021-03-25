import pickle
import os
import numpy as np
import sys


def load_pickle(path, verbose=True):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and
    disabling garbage collector helps with loading speed."""
    # assert osp.exists(path), "File not exists: {}".format(path)
    # gc.disable()
    with open(path, 'rb') as f:
        # check the python version
        if sys.version_info.major == 3:
            ret = pickle.load(f, encoding='bytes')
        elif sys.version_info.major == 2:
            ret = pickle.load(f)
    # gc.enable()
    if verbose:
        print('Loaded pickle file {}'.format(path))
    return ret


def load_data(path_root):
    X = []
    Y = []
    for path in os.listdir(path_root):
        data_point = load_pickle(os.path.join(path_root, path), "rb")
        X.append(data_point['x'])
        Y.append(data_point['y'])

    # count = 0
    # for path in os.listdir('acf2'):
    #
    #     if count == 23:
    #         break
    #     count += 1
    #     data_point = load_pickle(os.path.join('acf2', path), "rb")
    #     X.append(data_point[b'x'])
    #     Y.append(data_point[b'y'])

    return np.asarray(X), np.asarray(Y)


if __name__ == '__main__':
    X, Y = load_data()
    print(len(X))
    print(len(Y))
    print(Y)
    print(X[0].shape)
