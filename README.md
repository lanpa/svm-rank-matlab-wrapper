# usage:

``` matlab
parameterString = '-c 1 -v 0';
filename_model = 'outputmodel.dat';

svm_rank_learn(label, qid, feature, parameterString,  filename_model);
AvgSwappedpairsPercent = svm_rank_classify(label, qid, feature, filename_model);
```
parameterString: the parameters described in the refenence below.


label: stores the target value for each example. (double precision)

qid: indicates the grouping information. (double precision)

feature: a single precision matrix, each column represents a feature vector.

Please check ```wrapperTest.m``` for example.

# tested environment
ubuntu 14.04/matlab R2014b

OSX 10.11/matlab R2015a


# refenence:
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
