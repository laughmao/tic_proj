from scipy.spatial import distance
import dlib
import numpy as np
import cv2

def mouth_aspect_ratio(point):
	d1 = distance.euclidean(point[1], point[7])
	d2 = distance.euclidean(point[2], point[6])
	d3 = distance.euclidean(point[3], point[5])
	d4 = distance.euclidean(point[0], point[4])
	ratio = (d1+d2+d3)/(d4*0.4*3)
	return ratio
	
def eye_aspect_ratio(point):
    # print(point)
	A = distance.euclidean(point[1], point[5])
	B = distance.euclidean(point[2], point[4])
	C = distance.euclidean(point[0], point[3])
	ear = (A + B) / (2.0 * C *0.3)
	return ear
	
def brow_aspect_ratio(point):
	A = point[4]
	B = point[5]
	C = point[2]
	D = point[7]
	d1 = distance.euclidean(A, B)
	d2 = distance.euclidean(D, C)
	ratio = d1/(d2*0.3)
	return ratio

def get_var(data):
	result = 0
	average = sum(data)/len(data)
	for i in data:
		result+=pow(i-average,2)
	result = result/(len(data)*1.0)	
	return result

def unifo_image(img):
	detector = dlib.get_frontal_face_detector()
	dets = detector(img,1)
	if len(dets)!=0:
		for k,d in enumerate(dets):
			height = d.bottom() - d.top()
			width = d.right() - d.left()
			img_blank = np.zeros((height,width,3), np.uint8)

		for i in range(height):
			for j in range(width):
				img_blank[i][j] = img[d.top()+i][d.left()+j]
		img_blank = cv2.resize(img_blank, (500,500), cv2.INTER_LINEAR)
		img = img_blank
	return img

def draw_img(data):
	img = np.zeros((500,500,3), np.uint8)
	pts = np.array( data, np.int32)
	pts = pts.reshape((-1,1,2))
	cv2.polylines(img, [pts], False,(0,255,255),3)
	#for i in data:
		#img[i[1]][i[0]] = 255
	cv2.imshow('test', img)
	cv2.imwrite('./picture/'+"test_face.jpg", img)

def add_group(g1, g2):
	if g1==[]:
		g1 = g1+g2
	else:
		for i in range(len(g1)):
			for j in range(2):
				g1[i][j] = g1[i][j]+g2[i][j]
	return g1

def avg_group(times, g1):
	for i in range(len(g1)):
		for j in range(2):
			g1[i][j] /= times
	return g1
