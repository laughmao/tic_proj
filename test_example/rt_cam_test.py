# -*- coding:utf-8 -*- - 
import datetime
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from zrq_lib import *
from svmutil import *

def predict(eye_feature, mouth_feature, brow_feature):
	y = 1
	eye_var = get_var(eye_feature) * 10
	mouth_var = get_var(mouth_feature) * 10
	brow_var = get_var(brow_feature) * 100
	result = "%s 0:%s 1:%s 2:%s" %(y, eye_var, mouth_var, brow_var)
	f = open('trainning_data/tmp.txt','w')
	f.write(result)
	f.write('\n')
	f.close()

	model = svm_load_model("trainning_data/mymodel")
	yt, xt = svm_read_problem('trainning_data/tmp.txt')
	p_lable,p_acc,p_val = svm_predict(yt, xt, model)
	return p_acc

# 对应特征点的序号
MOUTH_START = 60
MOUTH_END = 67

BROW_START = 17
BROW_END = 26

RIGHT_EYE_START = 36
RIGHT_EYE_END = 41
LEFT_EYE_START = 42 
LEFT_EYE_END = 47

#用来存储关键值数据


flag = 1
count_times = 0

# 获取当前路径
pwd = os.getcwd()
model_path = os.path.join(pwd, 'trainning_data')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

cap = cv2.VideoCapture(0)
while True:
	count_times+=1
	eye_feature = []
	mouth_feature = []
	brow_feature = []
	s_time = datetime.datetime.now()
	while((datetime.datetime.now()-s_time).seconds<=10):
		ret, img = cap.read()# 读取视频流的一帧
		#img = unifo_image(img)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转成灰度图像
		rects = detector(gray, 1)# 人脸检测
		for rect in rects:# 遍历每一个人脸
			#print('-'*20)
			shape = predictor(gray, rect)# 检测特征点
			points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
			
			mouthPoint= points[MOUTH_START:MOUTH_END + 1]# 取出鼻子对应的特征点
			browPoint= points[BROW_START:BROW_END + 1]
			leyePoint = points[LEFT_EYE_START:LEFT_EYE_END + 1]
			reyePoint = points[RIGHT_EYE_START:RIGHT_EYE_END +1]
			
			for po in mouthPoint:
				cv2.circle(img, tuple(po), 5, color=(0, 255, 0))
			for po in browPoint:
				cv2.circle(img, tuple(po), 5, color=(0, 255, 0))
			for po in leyePoint:
				cv2.circle(img, tuple(po), 5, color=(0, 255, 0))
			for po in reyePoint:
				cv2.circle(img, tuple(po), 5, color=(0, 255, 0))
			
			mratio = mouth_aspect_ratio(mouthPoint)
			bratio = brow_aspect_ratio(browPoint)
			lear = eye_aspect_ratio(leyePoint)
			rear = eye_aspect_ratio(reyePoint)
			ear = (lear+rear)/2# 求左右眼EAR的均值
			
			eye_feature.append(ear)
			mouth_feature.append(mratio)
			brow_feature.append(bratio)
			
		#print "the mouth ratio is: %s" %mratio
		#print "the brow ratio is: %s" %bratio
		#print "the eye ratio is: %s" %ear
		
		cv2.imshow("Frame", img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			flag = 0
			break
	#每10s进行一次判断输出
	print "*"*20
	print "NO.%s times test" %count_times
	accuracy = predict(eye_feature, mouth_feature, brow_feature)
	print accuracy[0]
	if accuracy[0]<=0.5:
		print "bad!"
	else:
		print "good!"
	if flag==0:
		break
		
cap.release()
cv2.destroyAllWindows()
