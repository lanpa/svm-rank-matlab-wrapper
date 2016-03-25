clear;
label = [3 2 1 1 1 2 1 1 2 3 4 1];
qid = [1 1 1 1 2 2 2 2 3 3 3 3];

feature = [[1.0 1.0 0.0 0.2 0.0]' [0.0 0.0 1.0 0.1 1.0]' [0.0 1.0 0.0 0.4 0.0]' [0.0 0.0 1.0 0.3 0.0]' [0.0 0.0 1.0 0.2 0.0]'...
 [1.0 0.0 1.0 0.4 0.0]' [0.0 0.0 1.0 0.1 0.0]' [0.0 0.0 1.0 0.2 0.0]' [0.0 0.0 1.0 0.1 1.0]' [1.0 1.0 0.0 0.3 0.0]'...
  [1.0 0.0 0.0 0.4 1.0]' [0.0 1.0 1.0 0.5 0.0]'];
feature = single(feature);
if 1
    dummy_ndim = 2^10;
    dummy_ndata = 2^10;
    groupSize = 2^8;
    label = rand(1, dummy_ndata);
    qid = [];
    for i = 1:dummy_ndata/groupSize
        qid = [qid ones(1, groupSize)*i];
    end
    feature = single(rand(dummy_ndim, dummy_ndata));

    label = round(label*1000)/1000;
    feature = round(feature*1000)/1000;
end
disp('start mex');

parfor i = 1:2
    svm_rank_learn(label, qid, (feature), ' -c 1 -v 0' ,['test_model' num2str(i) '.dat']);
    err = svm_rank_classify(label, qid, feature, ['test_model' num2str(i) '.dat']);
    disp(err);
    clear mex;
end



f = fopen('train001.txt','W');
for i = 1:numel(label)
    fprintf(f,'%f qid:%d ', label(i), qid(i));
    for j = 1:size(feature,1)
        fprintf(f,'%d:%f ',j,feature(j,i));
    end
    fprintf(f,'\n');
end
fclose(f);

% compare the output with original code

parfor i = 1:2
    unix(['/home/dexter/Downloads/svm_rank/svm_rank_learn -v 0 -c ' num2str(1) ' train001.txt model_orig' num2str(i) '.dat']);
end

[s, r] = unix('/home/dexter/Downloads/svm_rank/svm_rank_classify train001.txt model_orig1.dat');
disp(r);
[s,r]=unix('sha1sum *.dat');
disp(r);

