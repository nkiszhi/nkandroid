import foolbox#对抗机器学习库
import numpy as np
from VennABERS import ScoresToMultiProbs, computeF, getFVal, prepareData
#from foolbox.gradient_estimators import evolutionary_strategies_gradient_estimator as GM
from collections import namedtuple

class IVAP():
    def __init__(self, model, beta, x_calib, y_calib, eps=.01):
        #super(IVAP, self).__init__(self, GM(eps))
        self.model = model
        self.beta = beta

        # prepare isotonic regression
        self.x_calib, self.y_calib = x_calib, y_calib
        self.calib_points = [(score, label) for score, label in zip(self._score(x_calib)[:,1], y_calib)]
        print(self.calib_points)
        yPrime, yCsd, xPrime, self.ptsUnique = prepareData(self.calib_points)
        self.F0, self.F1 = computeF(xPrime, yCsd)

    def maxminnorm(self,array):
        maxcols = array.max()
        mincols = array.min()
        data_shape = array.shape
        data_rows = data_shape[0]
        t = np.empty(data_rows)
        t[:] = (array[:] - mincols) / (maxcols - mincols)
        return t

    def bounds(self):
        return (0, 1)
    
    def channel_axis(self):
        return 3
    
    def __enter__(self):
        return self
    
    def __exit__(self, t, v, tb):
        return None
    
    def _score(self, images, batch_size=128):
        scores = np.zeros((images.shape[0], 2))
        num_batches = images.shape[0] // batch_size
        #for i in range(num_batches):
        start = 0
        end = images.shape[0]
        #batch = images[start:end]
        #print(self.maxminnorm(self.model.decision_function(images)))
        #scores[start:end,0] = self.maxminnorm(self.model.decision_function(images))
        scores[start:end,:]= self.model._predict_proba_lr(images)
        #scores[start:end,1]= self.model.predict(images)
        #print(scores[start:end,:])
        #print(scores)
        return scores

    def num_classes(self):
        return 3

    def batch_predictions(self, images, batch_size=128):
        if images.shape[0] == 0:
            return np.array([])

        logits = np.zeros((images.shape[0], 2))
        #num_batches = images.shape[0] // batch_size
        #for i in range(num_batches):
        start = 0
        end = images.shape[0]
        batch = images[start:end]
        logits[start:end,:] = self._score(batch)

        p0s, p1s = getFVal(self.F0, self.F1, self.ptsUnique, logits[:, 1])

        labels = []
        for i, (p0, p1, x) in enumerate(zip(p0s, p1s, images)):
            p = p1 / (1 - p0 + p1)
            y = np.zeros(2)
            if p > .5:
                y[1] = 1
            else:
                y[0] = 1
            if p1-p0 <= self.beta:
                labels.append([*y, 0])
            else:
                labels.append([*y, 1])
        #print(labels)
        return p0s,p1s,labels

    def batch_predictions1(self, images, batch_size=128):
        if images.shape[0] == 0:
            return np.array([])

        logits = np.zeros((images.shape[0], 2))
        #num_batches = images.shape[0] // batch_size
        #for i in range(num_batches):
        start = 0
        end = images.shape[0]
        batch = images[start:end]
        logits[start:end,:] = self._score(batch)
        logits=self.model._predict_proba_lr(images)

        p0s, p1s = getFVal(self.F0, self.F1, self.ptsUnique, logits[:, 1])

        labels = []
        for i, (p0, p1, x) in enumerate(zip(p0s, p1s, images)):
            p = p1 / (1 - p0 + p1)
            y = np.zeros(2)
            if p > .5:
                y[1] = 1
            else:
                y[0] = 1
            if p1-p0 <= self.beta:
                labels.append([*y, 0])
            else:
                labels.append([*y, 1])
        # print(labels)
        return labels
    
    def predictions(self, image):
        return self.batch_predictions1(image.reshape(1, *image.shape), batch_size=1)
    
    def evaluate(self, x_test, y_test, batch_size=1):
        labels = self.batch_predictions1(x_test, batch_size)
        tps = sum([np.array(y).argmax()== 1 and np.array(y_hat[:2]).argmax() == 1 and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])
        fps = sum([np.array(y).argmax() == 0 and np.array(y_hat[:2]).argmax() == 1 and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])
        tns = sum([np.array(y).argmax() == 0 and np.array(y_hat[:2]).argmax() == 0 and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])
        fns = sum([np.array(y).argmax() == 1 and np.array(y_hat[:2]).argmax() == 0 and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])
        
        rej = sum([y_hat[2] for y_hat in labels]) / x_test.shape[0]
        trs = sum([np.array(y).argmax() != np.array(y_hat[:2]).argmax() and y_hat[2] == 1 for y, y_hat in zip(y_test, labels)])
        frs = sum([np.array(y).argmax() == np.array(y_hat[:2]).argmax() and y_hat[2] == 1 for y, y_hat in zip(y_test, labels)])
        tas = sum([np.array(y).argmax() == np.array(y_hat[:2]).argmax() and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])
        fas = sum([np.array(y).argmax() != np.array(y_hat[:2]).argmax() and y_hat[2] == 0 for y, y_hat in zip(y_test, labels)])

        Metrics = namedtuple('Metrics', ['acc', 'tpr', 'fpr', 'trr', 'frr', 'rej'])
        acc = (tps + tns) / ((1 - rej) * x_test.shape[0])
        tpr = tps / (tps + fns)
        fpr = fps / (fps + tns)
        trr = trs / (trs + fas)
        frr = frs / (frs + tas)

        return Metrics(acc, tpr, fpr, trr, frr, rej)

class IVAPCriterion(foolbox.criteria.Criterion):
    def __init__(self):
        super(IVAPCriterion, self).__init__()
    
    def is_adversarial(self, predictions, label):
        return predictions[2] == 0 and predictions[:2].argmax() != label
    
    def name(self):
        return 'IVAPCriterion'
