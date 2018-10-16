# -*- coding:utf-8 -*- - 
import datetime
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from zrq_lib import *

flag = 0

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
eye_feature = []
mouth_feature = []
brow_feature = []

#用来存储最后需要画出的点
totalPoint = []

# 获取当前路径
pwd = os.getcwd()
model_path = os.path.join(pwd, 'trainning_data')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

#开启摄像头
cap = cv2.VideoCapture(0)
s_time = datetime.datetime.now()
count_time = 0
while((datetime.datetime.now()-s_time).seconds<=10):
	ret, img = cap.read()# 读取视频流的一帧
	img = unifo_image(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转成灰度图像
	rects = detector(gray, 1)# 人脸检测
	if len(rects)!=0:
		flag = 1
		for rect in rects:# 遍历每一个人脸
			tmpPoint = []
			print('-'*20)
			shape = predictor(gray, rect)# 检测特征点
			points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
			
			mouthPoint= points[MOUTH_START:MOUTH_END + 1]# 取出鼻子对应的特征点
			browPoint= points[BROW_START:BROW_END + 1]
			leyePoint = points[LEFT_EYE_START:LEFT_EYE_END + 1]
			reyePoint = points[RIGHT_EYE_START:RIGHT_EYE_END +1]

			mratio = mouth_aspect_ratio(mouthPoint)
			bratio = brow_aspect_ratio(browPoint)
			lear = eye_aspect_ratio(leyePoint)
			rear = eye_aspect_ratio(reyePoint)
			ear = (lear+rear)/2# 求左右眼EAR的均值
			
			eye_feature.append(ear)
			mouth_feature.append(mratio)
			brow_feature.append(bratio)
			
			#tmpPoint = list(mouthPoint)+list(browPoint)+list(leyePoint)+list(reyePoint)
			#totalPoint = add_group(totalPoint, tmpPoint)
			
		print "the mouth ratio is: %s" %mratio
		print "the brow ratio is: %s" %bratio
		print "the eye ratio is: %s" %ear
		count_time += 1
	#cv2.imshow('detect result', gray)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
cap.release()
if flag!=0:
	y = raw_input("plz input weather the result is good or bad:")
	eye_var = get_var(eye_feature) * 10
	mouth_var = get_var(mouth_feature) * 10
	brow_var = get_var(brow_feature) * 100
	result = "%s 0:%s 1:%s 2:%s" %(y, eye_var, mouth_var, brow_var)
	f = open('trainning_data/tics_train.txt','a')
	f.write(result)
	f.write('\n')
	f.close()
	#totalPoint = avg_group(count_time, totalPoint)
	#draw_img(totalPoint)
	#print "draw successfully!"

cv2.destroyAllWindows()

