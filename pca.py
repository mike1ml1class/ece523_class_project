#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

def analysis(X_train,Y_train):

    pca = PCA(n_components=3)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    new_feat0 = X_train_reduced[:,0:1]
    new_feat1 = X_train_reduced[:,1:2]
    new_feat2 = X_train_reduced[:,2:3]

    plt.figure()
    plt.subplot(2,2,1)
    hp.plot_tr_data(np.hstack((new_feat0,new_feat1)),Y_train)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')

    plt.subplot(2,2,2)
    hp.plot_tr_data(np.hstack((new_feat0,new_feat2)),Y_train)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 2')

    plt.subplot(2,2,3)
    hp.plot_tr_data(np.hstack((new_feat1,new_feat2)),Y_train)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    #plt.legend(('Survived0','Survived1'),loc='upper right')
    plt.tight_layout()
