import cv2
import mediapipe as mp

class PeopleDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        person_detected = False
        if results.pose_landmarks:
            person_detected = True
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return person_detected
