import os
import cv2 as cv
import numpy as np
import threading
import time
import pygame
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import img_to_array

# 人脸识别代码
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

# 数据库操作
def load_database(database_path, detector, recognizer):
    db_features = dict()
    print('Loading database ...')
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            identity = filename[:-4]
            db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
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
                    np.save(os.path.join(database_path, f'{identity}.npy'), features[0])
    cnt = len(db_features)
    print(f'Database: {cnt} loaded in total, {npy_cnt} loaded from .npy, {cnt - npy_cnt} loaded from images.')
    return db_features

# 拍摄背景
def getbackground():
    cap = cv.VideoCapture(0)
    num = 200
    while True:
        ok, frame = cap.read()
        image = cv.GaussianBlur(frame, (5, 5), 0)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        num -= 1
        if num % 10 == 0:
            print(f"倒计时 {num / 10}")
        if num == 1:
            backgound = gray
            cv.imwrite("images/backgound.jpg", backgound)
            break

# 提醒1
def slogan_short():
    timeplay = 1.5
    global playflag_short
    playflag_short = 0
    while True:
        if playflag_short == 1:
            pygame.mixer.music.load(file_slogan_short)
            print("------------请您戴好口罩")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag_short = 0
        time.sleep(0)

# 提醒2
def slogan():
    timeplay = 18
    global playflag
    playflag = 0
    while True:
        if playflag == 1:
            pygame.mixer.music.load(file_slogan)
            print("------------全国疾控中心提醒您：预防千万条，口罩第一条。")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag = 0
        time.sleep(0)

def nothing(x):
    pass

# 必要初始化
def facesdetecter_init():
    global thread_slogan, thread_slogan_short
    thread_slogan = threading.Thread(target=slogan).start()
    thread_slogan_short = threading.Thread(target=slogan_short).start()
    image = cv.imread("images/backgound.jpg")
    cv.imshow('skin', image)
    cv.createTrackbar("minH", "skin", 15, 180, nothing)
    cv.createTrackbar("maxH", "skin", 25, 180, nothing)

def facesdetecter(image):
    # 高斯滤波去噪
    image = cv.GaussianBlur(image, (5, 5), 0)
    # 转换为灰度图像，减少计算复杂度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 转换为HSV色彩空间，便于肤色检测
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)

    # 获取滑动条的位置值，用于动态调整肤色检测的范围
    minH = cv.getTrackbarPos("minH", 'skin')
    maxH = cv.getTrackbarPos("maxH", 'skin')
    if minH > maxH: maxH = minH

    # 肤色检测
    thresh_h = cv.inRange(H, minH, maxH)
    cv.imshow("skin", thresh_h)  # 显示肤色检测结果

    # 人脸检测
    faces = facecasc.detectMultiScale(gray, 1.3, 5)
    # 眼睛检测
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)

    # 在检测到的人脸区域绘制矩形框
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 在检测到的眼睛区域绘制矩形框
    for (x, y, w, h) in eyes:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 初始化眼睛和口罩区域的总面积
    total_area_eyes = 0
    total_area_mask = 0

    # 确定眼睛区域，计算眼睛区域的面积
    if len(eyes) > 1:
        rect_eyes = []
        (x1, y1, w1, h1) = eyes[0]  # 左眼坐标
        for (x, y, w, h) in eyes[1:]:
            (x2, y2, w2, h2) = (x, y, w, h)
            rect_eyes.append((x1, y1, x2 + w2 - x1, y2 + h2 - y1))
            (x1, y1, w1, h1) = (x2, y2, w2, h2)
        for (x, y, w, h) in rect_eyes:
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 250, 255), 2)
            thresh_eyes = thresh_h[y:y + h, x:x + w]
            contours, hierarchy = cv.findContours(thresh_eyes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            for cont in contours:
                Area = cv.contourArea(cont)
                total_area_eyes += Area
        print("total_area_eyes=", total_area_eyes)

    # 确定口罩区域，计算口罩区域的面积
    if len(eyes) > 1:
        # 假设口罩区域位于眼睛下方
        for (x, y, w, h) in eyes:
            rect_mask = (x, y + h, w, h * 2)  # 假设口罩区域高度为眼睛高度的两倍
            cv.rectangle(image, (x, y + h), (x + w, y + h * 3), (0, 255, 255), 2)
            thresh_mask = thresh_h[y + h:y + h * 3, x:x + w]
            contours, hierarchy = cv.findContours(thresh_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image, contours, -1, (0, 0, 255), 3)
            for cont in contours:
                Area = cv.contourArea(cont)
                total_area_mask += Area
        print("total_area_mask=", total_area_mask)

    # 根据眼睛和口罩区域的面积判断是否佩戴口罩，并进行相应的操作
    if total_area_eyes < total_area_mask:
        print("------------无口罩")
        playflag_short = 1  # 触发短语音提醒
    else:
        print("------------------已经戴口罩")
        playflag = 1  # 触发长语音提醒


# 显示识别和预测结果
def visualize0(image, faces, identities, fps, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()
    cv.putText(output, f'FPS: {fps:.2f}', (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
    for face, identity in zip(faces, identities):
        bbox = face[:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1]), box_color, 2)
        cv.putText(output, str(identity), (bbox[0], bbox[1] - 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
    return output

if __name__ == '__main__':
    # 初始化pygame的mixer模块，用于播放声音
    pygame.mixer.init(frequency=16000, size=-16, channels=2, buffer=4096)

    # 设置摄像头设备ID
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 初始化人脸检测器
    detector = cv.FaceDetectorYN.create(
        model="./models/face_detection_yunet_2022mar.onnx",
        config="",
        input_size=[w, h],  # [width, height]
        score_threshold=0.99,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv.dnn.DNN_TARGET_CPU,
    )

    # 初始化人脸识别器
    recognizer = cv.FaceRecognizerSF.create(
        model="./models/face_recognition_sface_2021dec.onnx",
        config="",
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv.dnn.DNN_TARGET_CPU,
    )

    # # 加载数据库
    # database = load_database("./database", detector, recognizer)

    # 创建情绪识别模型
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights('./models/model.h5')

    # 初始化人脸和眼睛检测器
    facecasc = cv.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    eyes_cascade = cv.CascadeClassifier("./models/haarcascade_eye_tree_eyeglasses.xml")

    # 情绪字典
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # # 加载面部和口罩识别模型
    # prototxtPath = r"./models/deploy.prototxt"
    # weightsPath = r"./models/res10_300x300_ssd_iter_140000.caffemodel"
    # faceNet = cv.dnn.readNet(prototxtPath, weightsPath)
    # maskNet = load_model("./models/mask_detector.model")

    # # 开始视频流
    # print("[INFO] starting video stream...")
    # print("我们需要拍一张背景,请人离开一下")
    # getbackground()
    # print("正在初始化")
    # facesdetecter_init()
    # print("初始化完成，可以回来了")

    # 实时面部识别
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        # 检测面部
        # faces = detect_face(detector, frame)
        # if faces is None:
            # continue

        # 提取特征
        # features = extract_feature(recognizer, frame, faces)

        # # 与数据库中的面部进行匹配
        # identities = []
        # for feature in features:
        #     isMatched = False
        #     for identity, db_feature in database.items():
        #         isMatched = match(recognizer, feature, db_feature)
        #         if isMatched:
        #             identities.append(identity)
        #             break
        #     if not isMatched:
        #         identities.append('Unknown')

        # 情绪检测
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces2 = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces2:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # # 口罩检测
        # (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        # for (box, pred) in zip(locs, preds):
        #     (startX, startY, endX, endY) = box
        #     (mask, withoutMask) = pred
        #     label = "Mask" if mask > withoutMask else "No Mask"
        #     color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #     label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        #     cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # # 执行其他检测
        # facesdetecter(frame)

        # 在输入图像上绘制结果
        tm.stop()
        # frame = visualize0(frame, faces, identities, tm.getFPS())
        cv.imshow('Face Recognition System', frame)
        tm.reset()

