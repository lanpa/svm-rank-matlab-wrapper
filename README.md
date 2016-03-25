# TODO:

* return result from svm_rank_classify


# usage:

```
parameterString = '-c 1 -v 0';
filename_model = 'outputmodel.dat';

svm_rank_learn(label, qid, feature, parameterString,  filename_model);
[TotalNumSwappedpairs, AvgSwappedpairsPercent] = svm_rank_classify(label, qid, feature, filename_model);
```

original code from http://svmlight.joachims.org/
