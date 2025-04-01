import cv2
from people_detection import PeopleDetector
from hand_gesture import HandGestureDetector
from interaction import InteractiveDisplay

# Initialize detectors
people_detector = PeopleDetector()
gesture_detector = HandGestureDetector()
interactive_display = InteractiveDisplay()

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame for person and gesture detection
    person_detected = people_detector.detect(frame)
    hands_data = gesture_detector.detect(frame)

    # Update interactive display
    interactive_display.update(frame, person_detected, hands_data)

    # Show frame
    cv2.imshow("Interactive Exhibition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
