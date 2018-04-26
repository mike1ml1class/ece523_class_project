#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

#def input(dataset):
#    return dataset.images, dataset.labels.astype(np.int32)


def analysis(x_tr,y_tr,x_te=None):
    #HIDDEN = [500,500,500,500,500]
    
    
    #print("Performing DNN Classification!")

    # Classify the data
    if x_te is not None:
        yhat = None
    else:
        yhat = None

    return yhat