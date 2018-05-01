close all;clear all;clc
names = {'SemiSupervised';
         'Bagging';
         'RandomForest';
         'RbfSVM';
         'LinSVM';
         'DNN';
         'LogisticRegression';
         'MLPsk';
         'AdaBoost';
         'DecisionTree';
         'KNN'};

% train data
data1 = [
90.9091
90.9091
90.9091
83.6139
81.0325
85.5219
86.9809
61.6162
81.8182
90.9091
87.4299];

% cross val data
data2 = [
80.02
81.15
80.81
81.94
80.81
81.82
80.71
61.62
80.93
79.91
78.57]

% test data
data3 = [
76.79
76.56
75.36
77.27
77.51
77.75
75.84
62.92
74.64
75.36
73.68];

data = [data1,data2,data3]';
[p,tbl,stats] = friedman(data);
disp(stats.meanranks);
table = multcompare(stats);

%yticks([1 2 3 4 5 6 7 8 9 10 11])'
yticklabels(names);
set(gca,'YDir','reverse');

fprintf('Classifier 1 & Classifier 2 & Lower Bound & Mean Difference & Upper Bound & P-Value \\\\\n');
for i=1:length(table)
   fprintf('%s & %s & %.2f & %.2f & %.2f & %.2f \\\\\n',...
   names{table(i,1)},...
   names{table(i,2)},...
   table(i,3),...
   table(i,4),...
   table(i,5),...
   table(i,6)) ;
end
    
    
    
