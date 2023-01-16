######################头文件区#############################
import os
import threading
import time

import cv2 as cv
import numpy as np
### kouzhao
import pygame
##ifmask
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
## Emotion
from keras.models import Sequential
from keras.models import load_model
from keras.utils import img_to_array


##########################人脸识别代码##############################
def detect_face(detector, image):
    h, w, _ = image.shape
    detector.setInputSize([w, h])
    flag, faces = detector.detect(image)
    return faces


def extract_feature(recognizer, image, faces):
    features = []
    for face in faces:
        aligned_face = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned_face)
        features.append(feature)
    return features


def match(recognizer, feature1, feature2, dis_type=1):
    l2_threshold = 1.275
    cosine_threshold = 0.363
    if dis_type == 0:  # COSINE
        cosine_score = recognizer.match(feature1, feature2, dis_type)
        return 1 if cosine_score >= cosine_threshold else 0
    else:  # NORM_L2
        norml2_distance = recognizer.match(feature1, feature2, dis_type)
        return 1 if norml2_distance <= l2_threshold else 0


############################################################################

###########################数据库############################################
def load_database(database_path, detector, recognizer):
    db_features = dict()
    print('Loading database ...')
    # load pre-extracted features first
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            identity = filename[:-4]
            if identity not in db_features:
                db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
    # load images and extract features
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            identity = filename[:-4]
            if identity not in db_features:
                print(identity)
                image = cv.imread(os.path.join(database_path, filename))

                faces = detect_face(detector, image)
                features = extract_feature(recognizer, image, faces)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    print("success")
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print(
        'Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt - npy_cnt))
    return db_features


#######################################################################################################################
###################################################################################################拍摄背景
def getbackground():
    cap = cv.VideoCapture(0)
    num = 200
    while True:
        ok, frame = cap.read()
        image = cv.GaussianBlur(frame, (5, 5), 0)  # 高斯滤波
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 将图片转化成灰度
        num = num - 1
        if num // 10 * 10 == num:
            print("倒计时", num / 10)
        if num == 1:
            backgound = gray
            cv.imwrite("images/backgound.jpg", backgound)
            break


#################################################################################################### 提醒1
def slogan_short():
    timeplay = 1.5
    global playflag_short
    playflag_short = 0
    while True:
        if playflag_short == 1:
            track = pygame.mixer.music.load(file_slogan_short)
            print("------------请您戴好口罩")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag_short = 0
        time.sleep(0)


#################################################################################################### 提醒2
def slogan():
    timeplay = 18
    global playflag
    playflag = 0
    while True:
        if playflag == 1:
            track = pygame.mixer.music.load(file_slogan)
            print("------------全国疾控中心提醒您：预防千万条，口罩第一条。")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag = 0
        time.sleep(0)


###################################################################################################
def nothing(x):  # 滑动条的回调函数
    pass


#################################################################################################### 必要初始化
def facesdetecter_init():
    global thread_slogan
    global thread_slogan_short
    # 多线程进行播放
    thread_slogan = threading.Thread(target=slogan).start()
    thread_slogan_short = threading.Thread(target=slogan_short).start()
    image = cv.imread("images/backgound.jpg")
    cv.imshow('skin', image)
    # 滑动条
    cv.createTrackbar("minH", "skin", 15, 180, nothing)
    cv.createTrackbar("maxH", "skin", 25, 180, nothing)


########################################################### 主要程序，识别特征，和比较肤色区域
def facesdetecter(image):
    image = cv.GaussianBlur(image, (5, 5), 0)  # 高斯滤波
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 将图片转化成灰度
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # 将图片转化成HSV格式
    H, S, V = cv.split(hsv)
    minH = cv.getTrackbarPos("minH", 'skin')
    maxH = cv.getTrackbarPos("maxH", 'skin')
    if minH > maxH:
        maxH = minH
    thresh_h = cv.inRange(H, minH, maxH)  # 0-180du 提取人体肤色区域
    cv.imshow("skin", thresh_h)  # 显示肤色图
    faces = facecasc.detectMultiScale(gray, 1.3, 5)  # 人脸检测
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)  # 眼睛检测
    for (x, y, w, h) in faces:
        frame = cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画框标识脸部
    for (x, y, w, h) in eyes:
        frame = cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画框标识眼部
    # 眼部区域和口罩部区域确定以及面积计算
    total_area_mask = 0  # 口罩区域面积初始化
    total_area_eyes = 0  # 口罩区域面积初始化
    # 如果找到眼睛将进行区域确定和面积计算，确定区域方法就是左眼睛左上角和右眼睛右下角的框为眼部区域，往下两倍高度为口罩部
    if len(eyes) > 1:
        rect_eyes = []
        (x1, y1, w1, h1) = eyes[0]  # 即左眼坐标
        for (x, y, w, h) in eyes[1:]:
            (x2, y2, w2, h2) = (x, y, w, h)
            rect_eyes.append((x1, y1, x2 + w2 - x1, y2 + h2 - y1))
            (x1, y1, w1, h1) = (x2, y2, w2, h2)
        for (x, y, w, h) in rect_eyes:
            frame = cv.rectangle(image, (x, y), (x + w, y + h), (255, 250, 255), 2)
            thresh_eyes = thresh_h[y:y + h, x:x + w]
            contours, hierarchy = cv.findContours(thresh_eyes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 寻找前景
            cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            for cont in contours:
                Area = cv.contourArea(cont)  # 计算轮廓面积
                total_area_eyes += Area
        print("total_area_eyes=", total_area_eyes)
        frame = cv.putText(image, "Eyes Area : {:.3f}".format(total_area_eyes), (120, 40), cv.FONT_HERSHEY_COMPLEX,
                           0.5, (0, 255, 0), 1)  # 绘制
        # 口罩区域
        rect_mask = [(x, y + h, w, h * 2)]
        for (x, y, w, h) in rect_mask:
            frame = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            thresh_mask = thresh_h[y:y + h, x:x + w]
            # image2[y:y+h,x:x+w]=thresh_h
            contours, hierarchy = cv.findContours(thresh_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 寻找前景
            cv.drawContours(image, contours, -1, (0, 0, 255), 3)
            for cont in contours:
                Area = cv.contourArea(cont)  # 计算轮廓面积
                total_area_mask += Area
        print("total_area_mask=", total_area_mask)
        frame = cv.putText(image, "Mask Area : {:.1f}".format(total_area_mask), (120, 80), cv.FONT_HERSHEY_COMPLEX,
                           0.5, (0, 0, 255), 1)  # 绘制
        # 面积比较以及播放语音
        if total_area_eyes < total_area_mask:
            print("------------无口罩")
            global playflag_short
            playflag_short = 1
            frame = cv.putText(image, "NO MASK", (rect_eyes[0][0], rect_eyes[0][1] - 10), cv.FONT_HERSHEY_COMPLEX,
                               0.5, (0, 255, 0), 1)  # 绘制
        if total_area_eyes > total_area_mask:
            global thread_slogan
            print("------------------已经戴口罩")
            global playflag
            playflag = 1
            frame = cv.putText(image, "HAVE MASK", (rect_eyes[0][0], rect_eyes[0][1] - 10), cv.FONT_HERSHEY_COMPLEX,
                               0.5, (0, 255, 0), 1)  # 绘制


###################################################################################################
def visualize0(image, faces, identities, fps, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()
    # put fps in top-left corner
    cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    for face, identity in zip(faces, identities):
        # draw bounding box
        bbox = face[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
        # put identity
        cv.putText(output, '{}'.format(identity), (bbox[0], bbox[1] - 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
    return output


###################################################################################################
###################################################################################################
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224),
                                (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


################################路径参数配置
# 语音提醒
file_slogan = r'radio/slogan.mp3'
file_slogan_short = r'radio/slogan_short.mp3'
pygame.mixer.init(frequency=16000, size=-16, channels=2, buffer=4096)
if __name__ == '__main__':
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Initialize FaceDetectorYN
    detector = cv.FaceDetectorYN.create(
        model="./models/face_detection_yunet_2022mar.onnx",
        config="",
        input_size=[w, h],  # [width, height]
        score_threshold=0.99,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,  # optional
        target_id=cv.dnn.DNN_TARGET_CPU,  # optional
    )
    # Initialize FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(
        model="./models/face_recognition_sface_2021dec.onnx",
        config="",
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,  # optional
        target_id=cv.dnn.DNN_TARGET_CPU,  # optional
    )
    # Load database
    database = load_database("./database", detector, recognizer)
    stranger_count = 0
    # Create the Emotion model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('./models/model.h5')
    facecasc = cv.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    # 主要还是通过眼睛位置进行区域判断
    eyes_cascade = cv.CascadeClassifier("./models/haarcascade_eye_tree_eyeglasses.xml")
    ##############################
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    prototxtPath = r".\models\deploy.prototxt"
    weightsPath = r".\models\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    maskNet = load_model(".\models\mask_detector.model")
    # initialize the video stream
    print("[INFO] starting video stream...")
    print("我们需要拍一张背景,请人离开一下")
    getbackground()
    print("正在初始化")
    facesdetecter_init()
    print("初始化完成，可以回来了")
    # Real-time face recognition
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        tm.start()
        # detect faces
        faces = detect_face(detector, frame)
        if faces is None:
            continue
        # extract features
        features = extract_feature(recognizer, frame, faces)
        # match detected faces with database
        identities = []
        for feature in features:
            isMatched = False
            for identity, db_feature in database.items():
                isMatched = match(recognizer, feature, db_feature)
                if isMatched:
                    identities.append(identity)
                    break
            if not isMatched:
                identities.append('Unknown')
        ## 面部表情检测
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces2 = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces2:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                       2, cv.LINE_AA)
        ## 面部遮挡检测
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            print(label)
            color = (0, 143, 255) if label == "Mask" else (255, 0, 0)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv.putText(frame, label, (startX, startY + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        ## 口罩检测
        facesdetecter(frame)
        # Draw results on the input image
        tm.stop()
        frame = visualize0(frame, faces, identities, tm.getFPS())
        cv.imshow('Face recognition system', frame)
        tm.reset()
