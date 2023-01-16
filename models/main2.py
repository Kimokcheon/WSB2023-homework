import os
import argparse
import random

from mp_handpose import MPHandPose
import numpy as np
import cv2 as cv
from mp_palmdet import MPPalmDet

parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', '-db', type=str, default='./database')
parser.add_argument('--face_detection_model', type=str,
                    default=r'D:\work\python\WSB2022-assignment-main\face_recognition_system\models\face_detection_yunet_2021dec-quantized.onnx')
parser.add_argument('--face_recognition_model', type=str,
                    default=r'D:\work\python\WSB2022-assignment-main\face_recognition_system\models\face_recognition_sface_2021dec-quantized.onnx')


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv.dnn.DNN_BACKEND_TIMVX]
    targets += [cv.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print(
        'This version of OpenCV does not support TIM-VX and NPU. Visit https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more information.')

# parser = argparse.ArgumentParser(description='Hand Pose Estimation from MediaPipe')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str,
                    default=r'D:\work\python\WSB2022-assignment-main\face_recognition_system\models\handpose_estimation_mediapipe_2022may.onnx',
                    help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--conf_threshold', type=float, default=0.8,
                    help='Filter out hands of confidence < conf_threshold.')
parser.add_argument('--save', '-s', type=str, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True,
                    help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()


def detect_face(detector, image):
    ''' Run face detection on input image.

    Paramters:
        detector - an instance of cv.FaceDetectorYN
        image    - a single image read using cv.imread

    Returns:
        faces    - a np.array of shape [n, 15]. If n = 0, return an empty list.
    '''
    faces = []
    ### TODO: your code starts here
    h, w, c = image.shape
    detector.setInputSize([w, h])

    flag, faces = detector.detect(image)

    ### your code ends here
    return faces


def extract_feature(recognizer, image, faces):
    ''' Run face alignment on the input image & face bounding boxes; Extract features from the aligned faces.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF
        image      - a single image read using cv.imread
        faces      - the return of detect_face

    Returns:
        features   - a length-n list of extracted features. If n = 0, return an empty list.
    '''
    features = []
    ### TODO: your code starts here
    for face in faces:
        aligned_face = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned_face)
        features.append(feature)
    ### your code ends here
    return features


def match(recognizer, feature1, feature2, dis_type=1):
    ''' Calculate the distatnce/similarity of the given feature1 and feature2.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF. Call recognizer.match to calculate distance/similarity
        feature1   - extracted feature from identity 1
        feature2   - extracted feature from identity 2
        dis_type   - 0: cosine similarity; 1: l2 distance; others invalid

    Returns:
        isMatched  - True if feature1 and feature2 are the same identity; False if different
    '''
    l2_threshold = 1.218
    cosine_threshold = 0.363
    isMatched = False
    ### TODO: your code starts here
    score = recognizer.match(feature1, feature2, dis_type)
    if score <= l2_threshold and dis_type == 1:
        isMatched = True
    elif score >= cosine_threshold and dis_type == 0:
        isMatched = True
    ### your code ends here
    return isMatched


def load_database(database_path, detector, recognizer):
    ''' Load database from the given database_path into a dictionary. It tries to load extracted features first, and call detect_face & extract_feature to get features from images (*.jpg, *.png).

    Parameters:
        database_path - path to the database directory
        detector      - an instance of cv.FaceDetectorYN
        recognizer    - an instance of cv.FaceRecognizerSF

    Returns:
        db_features   - a dictionary with filenames as key and features as values. Keys are used as identity.
    '''
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
                image = cv.imread(os.path.join(database_path, filename))

                faces = detect_face(detector, image)
                features = extract_feature(recognizer, image, faces)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print(
        'Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt - npy_cnt))
    return db_features


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


def visualize1(image, hands, print_result=False):
    output = image.copy()

    for idx, handpose in enumerate(hands):
        conf = handpose[-1]
        bbox = handpose[0:4].astype(np.int32)
        landmarks = handpose[4:-1].reshape(21, 2).astype(np.int32)

        # Print results
        if print_result:
            print('-----------hand {}-----------'.format(idx + 1))
            print('conf: {:.2f}'.format(conf))
            print('hand box: {}'.format(bbox))
            print('hand landmarks: ')
            for l in landmarks:
                print('\t{}'.format(l))

        # Draw line between each key points
        cv.line(output, landmarks[0], landmarks[1], (255, 255, 255), 2)
        cv.line(output, landmarks[1], landmarks[2], (255, 255, 255), 2)
        cv.line(output, landmarks[2], landmarks[3], (255, 255, 255), 2)
        cv.line(output, landmarks[3], landmarks[4], (255, 255, 255), 2)

        cv.line(output, landmarks[0], landmarks[5], (255, 255, 255), 2)
        cv.line(output, landmarks[5], landmarks[6], (255, 255, 255), 2)
        cv.line(output, landmarks[6], landmarks[7], (255, 255, 255), 2)
        cv.line(output, landmarks[7], landmarks[8], (255, 255, 255), 2)

        cv.line(output, landmarks[0], landmarks[9], (255, 255, 255), 2)
        cv.line(output, landmarks[9], landmarks[10], (255, 255, 255), 2)
        cv.line(output, landmarks[10], landmarks[11], (255, 255, 255), 2)
        cv.line(output, landmarks[11], landmarks[12], (255, 255, 255), 2)

        cv.line(output, landmarks[0], landmarks[13], (255, 255, 255), 2)
        cv.line(output, landmarks[13], landmarks[14], (255, 255, 255), 2)
        cv.line(output, landmarks[14], landmarks[15], (255, 255, 255), 2)
        cv.line(output, landmarks[15], landmarks[16], (255, 255, 255), 2)

        cv.line(output, landmarks[0], landmarks[17], (255, 255, 255), 2)
        cv.line(output, landmarks[17], landmarks[18], (255, 255, 255), 2)
        cv.line(output, landmarks[18], landmarks[19], (255, 255, 255), 2)
        cv.line(output, landmarks[19], landmarks[20], (255, 255, 255), 2)

        for p in landmarks:
            cv.circle(output, p, 2, (0, 0, 255), 2)

    return output


if __name__ == '__main__':

    # Initialize video stream
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Initialize FaceDetectorYN
    detector = cv.FaceDetectorYN.create(
        model=args.face_detection_model,
        config="",
        input_size=[w, h],  # [width, height]
        score_threshold=0.99,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,  # optional
        target_id=cv.dnn.DNN_TARGET_CPU,  # optional
    )
    # Initialize FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(
        model=args.face_recognition_model,
        config="",
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,  # optional
        target_id=cv.dnn.DNN_TARGET_CPU,  # optional
    )
    # palm detector
    palm_detector = MPPalmDet(
        modelPath=r'D:\work\python\WSB2022-assignment-main\face_recognition_system\models\palm_detection_mediapipe_2022may.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.8,
        backendId=args.backend,
        targetId=args.target)
    # handpose detector
    handpose_detector = MPHandPose(
        modelPath=r'D:\work\python\WSB2022-assignment-main\face_recognition_system\models\handpose_estimation_mediapipe_2022may.onnx',
        confThreshold=args.conf_threshold,
        backendId=args.backend,
        targetId=args.target)
    # Load database
    database = load_database(args.database_dir, detector, recognizer)

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
        palms = palm_detector.infer(frame)
        hands = np.empty(shape=(0, 47))

        # Estimate the pose of each hand
        for palm in palms:
            # Handpose detector inference
            handposes = handpose_detector.infer(frame, palm)
            if handposes is not None:
                hands = np.vstack((hands, handposes))

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
                if len(palms) == 2 and len(faces) == 1:
                    np.save(os.path.join('./database', 'add_other.npy'), features[0])

        tm.stop()

        # Draw results on the input image
        frame = visualize0(frame, faces, identities, tm.getFPS())
        frame = visualize1(frame, hands)
        # Visualize results in a new Window
        cv.imshow('Face recognition system', frame)

        tm.reset()
