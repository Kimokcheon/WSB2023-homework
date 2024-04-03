import os
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

def initialize_emotion_recognition_model():
    """Initializes and returns the CNN model for emotion recognition."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights('./models/model.h5')
    return model

if __name__ == '__main__':
    device_id = 0  # Camera device ID
    cap = cv.VideoCapture(device_id)  # Start video capture
    face_cascade = cv.CascadeClassifier('./models/haarcascade_frontalface_default.xml')  # Load face detector
    model = initialize_emotion_recognition_model()  # Initialize emotion recognition model
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    print("[INFO] Starting video stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]  # Extract region of interest
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion_label = emotion_dict[maxindex]
            cv.putText(frame, emotion_label, (x + 20, y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Emotion Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv.destroyAllWindows()
