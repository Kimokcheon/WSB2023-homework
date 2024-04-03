import cv2 as cv
from emotion_recognizer import EmotionRecognizer

def start_video_stream(device_id=0):
    ER = EmotionRecognizer('./models/model.h5', './models/haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(device_id)
    print("[INFO] Starting video stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        predictions = ER.recognize_emotions(frame)
        for emotion, (x, y, w, h) in predictions:
            cv.putText(frame, emotion, (x + 20, y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Emotion Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    start_video_stream()
