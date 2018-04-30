data = [];

[p,tbl,stats] = friedman(data);
disp(stats.meanranks)
multcompare(stats)
