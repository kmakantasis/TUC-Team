# -*- coding: utf-8 -*-
import sklearn.svm as SVM
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def TestSVM(features, labels, silence=True):
    X_train = features[0:1600,:]
    Y_train = labels[0:1600]

    X_test = features[1600:,:]
    Y_test = labels[1600:]

    clf = SVM.SVC()
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)  

    error = np.mean(abs(predictions-Y_test))

    cm = confusion_matrix(Y_test, predictions)

    cm_sum = np.sum(cm, axis=1)

    cm_mean = cm.T / cm_sum
    cm_mean = cm_mean.T
    
    if silence==False:
        plt.matshow(cm_mean)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    return error, cm_mean
