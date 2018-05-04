#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mark Miller - ECE523 - Homework #1 - Practice Bonus

A text file, hw1-scores.txt, containing classifier error measurements has been
uploaded to D2L. Each of the columns represents a classifier and each row a data
set that was evaluated. Are all of the classifiers performing equally? Is there
one or more classifiers that is performing better than the others?
"""

# Import necessary libraries
import numpy as np
import scipy.stats as stats

names = {'SemiSupervised',
         'Bagging',
         'RandomForest',
         'RbfSVM',
         'LinSVM',
         'DNN',
         'LogisticRegression',
         'MLPsk',
         'AdaBoost',
         'DecisionTree',
         'KNN'}

# train data
data1 = [
90.9091,
90.9091,
90.9091,
83.6139,
81.0325,
85.5219,
86.9809,
61.6162,
81.8182,
90.9091,
87.4299]

# cross val data
data2 = [
80.02,
81.15,
80.81,
81.94,
80.81,
81.82,
80.71,
61.62,
80.93,
79.91,
78.57]

# test data
data3 = [
76.79,
76.56,
75.36,
77.27,
77.51,
77.75,
75.84,
62.92,
74.64,
75.36,
73.68];

data = np.array([data1,data2,data3])



#fprintf('Classifier 1 & Classifier 2 & Lower Bound & Mean Difference & Upper Bound & P-Value \\\\\n');
#for i=1:length(table)
#   fprintf('%s & %s & %.2f & %.2f & %.2f & %.2f \\\\\n',...
#   names{table(i,1)},...
#   names{table(i,2)},...
#   table(i,3),...
#   table(i,4),...
#   table(i,5),...
#   table(i,6)) ;
#end





# Friedman test
stats_f = stats.friedmanchisquare( *(data[i, :] for i in range(data.shape[0])) )
print(stats_f)
print('')

# Wilcoxon signed-rank test over each unique pair (Couldn't get TukeyHSD working)
nums=list(np.arange(0,data.shape[1]))
print('Wilcoxon signed-rank test over each unique pair: ')
print(' a  b    pval    Reject Null Hyp  Result ')
print('-----------------------------------------')
for a in nums:
    for b in nums[a+1:]:
        stats_w = stats.wilcoxon( data[:,a], data[:,b] )
        # pval <= 0.05 rejects the null hypothesis that data samples are equal
        if stats_w.pvalue <= 0.05:
            rstr = "a  > b"
            pstr = "True"
        else:
            pstr = "False"
            rstr = "a !> b"
        print(' %d  %d  %f  %10s %12s' % (a,b,stats_w.pvalue,pstr,rstr) )
