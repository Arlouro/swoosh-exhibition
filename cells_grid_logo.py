import cv2
import numpy as np
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize fullscreen window
cv2.namedWindow("Interactive Projection Grid", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Interactive Projection Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get screen resolution
screen_width = cv2.getWindowImageRect("Interactive Projection Grid")[2]
screen_height = cv2.getWindowImageRect("Interactive Projection Grid")[3]

# Grid settings
GRID_ROWS = 30 
GRID_COLS = 60  
CELL_SIZE = min(screen_width // GRID_COLS, screen_height // GRID_ROWS)
LINE_LENGTH = CELL_SIZE / 2

width = GRID_COLS * CELL_SIZE
height = GRID_ROWS * CELL_SIZE

# Load and process mask image
mask_img = cv2.imread("assets/imgs/image.png", cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise FileNotFoundError("Mask image not found!")

mask_aspect_ratio = mask_img.shape[1] / mask_img.shape[0]
new_width = min(width, int(height * mask_aspect_ratio))
new_height = min(height, int(width / mask_aspect_ratio))
resized_mask = cv2.resize(mask_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Create a grid-sized mask
mask_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
offset_x = (GRID_COLS - new_width // CELL_SIZE) // 2
offset_y = (GRID_ROWS - new_height // CELL_SIZE) // 2
for row in range(new_height // CELL_SIZE):
    for col in range(new_width // CELL_SIZE):
        if resized_mask[row * CELL_SIZE, col * CELL_SIZE] < 128:  
            mask_grid[offset_y + row, offset_x + col] = 1

last_interaction = np.zeros((GRID_ROWS, GRID_COLS))

def update_angles(angles, hand_positions):
    current_time = time.time()
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            time_since_interaction = current_time - last_interaction[row, col]
            
            if time_since_interaction > 2:
                if mask_grid[row, col] == 1:
                    angles[row, col] = max(angles[row, col] - 2, 45)
                else:
                    angles[row, col] = max(angles[row, col] - 2, 0)
    
    for hx, hy in hand_positions:
        grid_x = int(hx * GRID_COLS)
        grid_y = int(hy * GRID_ROWS)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= grid_y + dy < GRID_ROWS and 0 <= grid_x + dx < GRID_COLS:
                    angles[grid_y + dy, grid_x + dx] = (angles[grid_y + dy, grid_x + dx] + 10) % 360
                    last_interaction[grid_y + dy, grid_x + dx] = current_time 

def draw_grid(frame, angles):
    current_time = time.time()
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = row * CELL_SIZE + CELL_SIZE // 2
            
            color = (0, 165, 255) if mask_grid[row, col] == 1 else (255, 255, 255)
            radian = np.deg2rad(angles[row, col])
            
            x1 = int(x - (LINE_LENGTH / 2) * np.cos(radian))
            y1 = int(y - (LINE_LENGTH / 2) * np.sin(radian))
            x2 = int(x + (LINE_LENGTH / 2) * np.cos(radian))
            y2 = int(y + (LINE_LENGTH / 2) * np.sin(radian))
            
            # Adjust thickness based on recent interaction
            time_since_interaction = current_time - last_interaction[row, col]
            if time_since_interaction < 1:
                thickness = 4
            elif time_since_interaction < 2:
                thickness = 3
            else:
                thickness = 2
            
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def main():
    cap = cv2.VideoCapture(0) 
    angles = np.where(mask_grid == 1, 45, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_hands = hands.process(rgb_frame)
        hand_positions = []
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                x, y = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y
                hand_positions.append((x, y))
        
        update_angles(angles, hand_positions)
        projection = np.zeros((height, width, 3), dtype=np.uint8)
        draw_grid(projection, angles)
        
        cv2.imshow("Interactive Projection Grid", projection)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()