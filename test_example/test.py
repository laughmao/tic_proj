from svmutil import *
import os

yt, xt  =svm_read_problem('trainning_data/tics_test.txt')
model = svm_load_model("trainning_data/mymodel")
p_label,p_acc,p_val = svm_predict(yt, xt, model)
print p_acc
#svm_save_model('trainning_data/mymodel',model)
