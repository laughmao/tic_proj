# coding=utf-8
import cv2
import dlib


# 初始化dlib人脸检测器
detector = dlib.get_frontal_face_detector()

# 初始化显示窗口
win = dlib.image_window()

# opencv加载视频文件
#cap = cv2.VideoCapture('/home/ljx/ImageDatabase/WaterBar.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, cv_img = cap.read()
    if cv_img is None:
        break

    # OpenCV默认是读取为RGB图像，而dlib需要的是BGR图像，因此这一步转换不能少
    img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
	# 检测人脸
    dets = detector(img, 0)

    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
