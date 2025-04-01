import cv2

class InteractiveDisplay:
    def __init__(self):
        self.state = "idle"

    def update(self, frame, person_detected, hands_data):
        if person_detected:
            cv2.putText(frame, "Person Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if hands_data:
            cv2.putText(frame, "Gesture Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Example: If a hand is raised, change the state
        if len(hands_data) == 1:
            cv2.putText(frame, "Interaction Mode!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
