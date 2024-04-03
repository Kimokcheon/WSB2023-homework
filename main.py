import cv2 as cv
import argparse
from emotion_recognizer import EmotionRecognizer 
from pose_estimation import PoseEstimator  
def main(args):
    if args.emotion:
        ER = EmotionRecognizer(model_path='./models/model.h5', cascade_path='./models/haarcascade_frontalface_default.xml')

    if args.pose:
        backend_target_pairs = [
            [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
            [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
            [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
            [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
            [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
        ]
        backend_id, target_id = backend_target_pairs[args.backend_target]
        PE = PoseEstimator(model_path=args.model_pose,
                           person_model_path='./models/person_detection_mediapipe_2023mar.onnx',
                           backend_id=backend_id,
                           target_id=target_id,
                           conf_threshold=args.conf_threshold)

    if args.input:  # Process image file
        image = cv.imread(args.input)
        if args.emotion:
            image = ER.recognize_emotions(image)
        if args.pose:
            image, _ = PE.infer(image)
        cv.imshow("Result", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:  # Process video stream
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.emotion:
                frame = ER.recognize_emotions(frame)
            if args.pose:
                frame, _ = PE.infer(frame)
            cv.imshow("Result", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MediaPipe-based Emotion and Pose Estimation')
    parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--emotion', action='store_true', help='Enable emotion recognition.')
    parser.add_argument('--pose', action='store_true', help='Enable pose estimation.')
    parser.add_argument('--model_pose', '-mp', type=str, default='./models/pose_estimation_mediapipe_2023mar.onnx', help='Path to the pose model.')
    parser.add_argument('--backend_target', '-bt', type=int, default=0, help='Choose backend-target pair to run pose estimation demo.')
    parser.add_argument('--conf_threshold', type=float, default=0.8, help='Confidence threshold for pose estimation.')
    args = parser.parse_args()
    
    main(args)
