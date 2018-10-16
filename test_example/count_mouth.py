# -*- coding:utf-8 -*- - 

import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils

EYE_AR_THRESH = 0.9# EAR阈值
EYE_AR_CONSEC_FRAMES = 1# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
MOUTH_START = 60
MOUTH_END = 67

def mouth_aspect_ratio(point):
	d1 = distance.euclidean(point[1], point[7])
	d2 = distance.euclidean(point[2], point[6])
	d3 = distance.euclidean(point[3], point[5])
	d4 = distance.euclidean(point[0], point[4])
	ratio = (d1+d2+d3)/(d4*0.2*3)
	return ratio

pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'trainning_data')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径

detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

frame_counter = 0# 连续帧计数 
blink_counter = 0# 眨眼计数
cap = cv2.VideoCapture(0)
while(1):
	ret, img = cap.read()# 读取视频流的一帧

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转成灰度图像
	rects = detector(gray, 1)# 人脸检测
	for rect in rects:# 遍历每一个人脸
		print('-'*20)
		shape = predictor(gray, rect)# 检测特征点
		points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
		mouthPoint= points[MOUTH_START:MOUTH_END + 1]# 取出鼻子对应的特征点
		ratio = mouth_aspect_ratio(mouthPoint)
		ear = ratio# 求左右眼EAR的均值
		mouthHull = cv2.convexHull(mouthPoint)# 寻找左眼轮廓
		cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓
	# 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


	cv2.imshow("Frame", img)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

