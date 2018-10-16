from svmutil import *
import os

y, x  =svm_read_problem('trainning_data/tics_train.txt')
model = svm_train(y, x, '-s 0 -t 3 -c 4.5 -g 2.6')
svm_save_model('trainning_data/mymodel',model)
print "train successfully!"
