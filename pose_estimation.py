import cv2 as cv
import numpy as np
from mp_pose import MPPose
from mp_persondet import MPPersonDet

class PoseEstimator:
    def __init__(self, model_path, person_model_path, backend_id=0, target_id=0, conf_threshold=0.8):
        self.pose_estimator = MPPose(modelPath=model_path,
                                     confThreshold=conf_threshold,
                                     backendId=backend_id,
                                     targetId=target_id)
        self.person_detector = MPPersonDet(modelPath=person_model_path,
                                           nmsThreshold=0.3,
                                           scoreThreshold=0.5,
                                           topK=5000,
                                           backendId=backend_id,
                                           targetId=target_id)

    def infer(self, image):
        persons = self.person_detector.infer(image)
        poses = []

        for person in persons:
            pose = self.pose_estimator.infer(image, person)
            if pose is not None:
                poses.append(pose)
        
        return self.visualize(image, poses)

    def visualize(self, image, poses):
        display_screen = image.copy()
        display_3d = np.zeros((400, 400, 3), np.uint8)
        cv.line(display_3d, (200, 0), (200, 400), (255, 255, 255), 2)
        cv.line(display_3d, (0, 200), (400, 200), (255, 255, 255), 2)
        cv.putText(display_3d, 'Main View', (0, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv.putText(display_3d, 'Top View', (200, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv.putText(display_3d, 'Left View', (0, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv.putText(display_3d, 'Right View', (200, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        is_draw = False  # ensure only one person is drawn

        for idx, pose in enumerate(poses):
            bbox, landmarks_screen, landmarks_word, mask, heatmap, conf = pose

            # Visualization code based on the landmarks and confidence scores
            # Include the drawing logic for edges, keypoints, and any other visual elements
            # Similar to the original script, adapted to work within this class

        return display_screen, display_3d
