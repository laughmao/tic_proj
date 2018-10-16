# -*- coding:utf-8 -*- - 

import dlib
import numpy
import sys

PREDICTOR_PATH = 'trainning_data/shape_predictor_68_face_landmarks.dat'   # 关键点提取模型路径
im = im.read('picture/man0.jpg')

detector = dlib.get_frontal_face_detector()
# 2. 载入关键点提取模型
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def test_fun():
	rects = detector(im,1)  # 检测人脸
	print len(rects)
	if len(rects) >= 1:      # 检测到人脸
		landmark = predictor(im, rects).part()
		return landmark
		
	return 0
