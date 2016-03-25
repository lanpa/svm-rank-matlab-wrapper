# usage:

```
parameterString = '-c 1 -v 0';
filename_model = 'outputmodel.dat';

svm_rank_learn(label, qid, feature, parameterString,  filename_model);
AvgSwappedpairsPercent = svm_rank_classify(label, qid, feature, filename_model);
```

label: stores the target value for each example. (double precision)

qid: indicates the grouping information. (double precision)

feature: a single precision matrix, each column represents a feature vector.

Please check ```wrapperTest.m``` for example.

# credit:
http://svmlight.joachims.org/
