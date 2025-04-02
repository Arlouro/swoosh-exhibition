import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose()

# Initialize fullscreen window
cv2.namedWindow("Interactive Projection Grid", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Interactive Projection Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get screen resolution after window creation
screen_width = cv2.getWindowImageRect("Interactive Projection Grid")[2]
screen_height = cv2.getWindowImageRect("Interactive Projection Grid")[3]

# Grid settings
GRID_ROWS = 20
GRID_COLS = 40
CELL_SIZE = min(screen_width // GRID_COLS, screen_height // GRID_ROWS)
LINE_LENGTH = CELL_SIZE / 1.5

width = GRID_COLS * CELL_SIZE
height = GRID_ROWS * CELL_SIZE

SWOOSH_CELLS = [
    (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),  # S
    (9, 3),
    (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
    (11, 7),
    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7),

    (8, 9), (8, 13),  # W
    (9, 9), (9, 13),
    (10, 9), (10, 11), (10, 13),
    (11, 9), (11, 10), (11, 12), (11, 13),
    (12, 9), (12, 13),

    (8, 15), (8, 16), (8, 17), (8, 18), (8, 19),  # O
    (9, 15), (9, 19),
    (10, 15), (10, 19),
    (11, 15), (11, 19),
    (12, 15), (12, 16), (12, 17), (12, 18), (12, 19),

    (8, 21), (8, 22), (8, 23), (8, 24), (8, 25),  # O
    (9, 21), (9, 25),
    (10, 21), (10, 25),
    (11, 21), (11, 25),
    (12, 21), (12, 22), (12, 23), (12, 24), (12, 25),

    (8, 27), (8, 28), (8, 29), (8, 30), (8, 31),  # S
    (9, 27),
    (10, 27), (10, 28), (10, 29), (10, 30), (10, 31),
    (11, 31),
    (12, 27), (12, 28), (12, 29), (12, 30), (12, 31),

    (8, 33), (8, 37),  # H
    (9, 33), (9, 37),
    (10, 33), (10, 34), (10, 35), (10, 36), (10, 37),
    (11, 33), (11, 37),
    (12, 33), (12, 37),
]

def draw_grid(frame, angles):
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = row * CELL_SIZE + CELL_SIZE // 2
            
            color = (0, 165, 255) if (row, col) in SWOOSH_CELLS else (255, 255, 255)
            
            radian = np.deg2rad(angles[row, col])
            
            x1 = int(x - (LINE_LENGTH / 2) * np.cos(radian))
            y1 = int(y - (LINE_LENGTH / 2) * np.sin(radian))
            x2 = int(x + (LINE_LENGTH / 2) * np.cos(radian))
            y2 = int(y + (LINE_LENGTH / 2) * np.sin(radian))
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 5)

def update_angles(angles, hand_positions, hands_detected):
    if not hands_detected:
        angles[:] = np.maximum(angles - 2, 0)  # Smoothly reduce angles to zero when no hands detected
    else:
        for hx, hy in hand_positions:
            grid_x = int(hx * GRID_COLS)
            grid_y = int(hy * GRID_ROWS)
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= grid_y + dy < GRID_ROWS and 0 <= grid_x + dx < GRID_COLS:
                        angles[grid_y + dy, grid_x + dx] = (angles[grid_y + dy, grid_x + dx] + 10) % 360

def main():
    cap = cv2.VideoCapture(0) 
    
    angles = np.zeros((GRID_ROWS, GRID_COLS))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_hands = hands.process(rgb_frame)
        results_pose = pose.process(rgb_frame)
        
        hand_positions = []
        hands_detected = results_hands.multi_hand_landmarks is not None
        
        if hands_detected:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                x, y = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y
                hand_positions.append((x, y))
        
        update_angles(angles, hand_positions, hands_detected)
        
        projection = np.zeros((height, width, 3), dtype=np.uint8)
        draw_grid(projection, angles)
        
        cv2.imshow("Interactive Projection Grid", projection)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()