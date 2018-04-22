import numpy as np
import matplotlib.pyplot as plt

#Function to generate 2 Multivariate Gaussian distribution
def gen_mvg(mu_1,mu_2,cov_1,cov_2,N):
    g1 = np.random.multivariate_normal(mu_1, cov_1, N)
    g2 = np.random.multivariate_normal(mu_2, cov_2, N)
    y1 = np.zeros((N,))
    y2 = np.ones((N,))
    X  = np.vstack((g1,g2))
    Y  = np.hstack((y1,y2))
    return X,Y


# Helper function to check accuracy of results
def check_accuracy(pred_results,test_truth):
    pass_fail = np.zeros((pred_results.shape[0],))
    for i in range(test_truth.shape[0]):   
        if pred_results[i] == test_truth[i]:
            pass_fail[i] = 1
        else:
            pass_fail[i] = 0
    accuracy = 100*pass_fail.sum()/pass_fail.shape[0]
    return accuracy, pass_fail
    

def plot_tr_data(X,Y):
    idx0 = np.where(Y==0)
    idx1 = np.where(Y==1)
    plt.scatter(X[idx0, 0], X[idx0, 1],c='blue')
    plt.scatter(X[idx1, 0], X[idx1, 1],c='red',marker='s')

# Plotting results of classification
def plot_class_res(X_test,pf_idxs):
    idx_p1 = pf_idxs[0]
    idx_f1 = pf_idxs[1]
    
    idx_p2 = pf_idxs[2]
    idx_f2 = pf_idxs[3]
    
    plt.scatter(X_test[idx_p1, 0], X_test[idx_p1, 1],c='blue',marker='o')
    plt.scatter(X_test[idx_p2, 0], X_test[idx_p2, 1],c='red',marker='s')
    plt.scatter(X_test[idx_f1, 0], X_test[idx_f1, 1],c='black',marker='o')
    plt.scatter(X_test[idx_f2, 0], X_test[idx_f2, 1],c='black',marker='s')

# Helper function for plotting
def get_idxs(results,pass_fail,c_idxs):
    idx_p = np.where(pass_fail==1)
    idx_f = np.where(pass_fail==0)
    idx_g1 = np.where(results == c_idxs[0])
    idx_g2 = np.where(results == c_idxs[1])
    idx_p1 = np.intersect1d(idx_p,idx_g1)
    idx_f1 = np.intersect1d(idx_f,idx_g1)
    idx_p2 = np.intersect1d(idx_p,idx_g2)
    idx_f2 = np.intersect1d(idx_f,idx_g2)
    idx_list = [idx_p1,idx_f1,idx_p2,idx_f2]
    return idx_list
    
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out