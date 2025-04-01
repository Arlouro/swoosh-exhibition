import cv2
import numpy as np

# Grid settings
GRID_ROWS = 20
GRID_COLS = 40
CELL_SIZE = 20
LINE_LENGTH = 15

# Fixed angle for all cells
SYNCED_ANGLE = 90
width = GRID_COLS * CELL_SIZE
height = GRID_ROWS * CELL_SIZE

SWOOSH_CELLS = [
    (8, 30), (8, 29), (8, 28), (8, 16), (8, 15),
    (9, 29), (9, 28), (9, 27), (9, 15), (9, 14),
    (10, 28), (10, 27), (10, 26), (10, 25), (10, 24), (10, 15), (10, 14), (10, 13),
    (11, 26), (11, 25), (11, 24), (11, 23), (11, 22), (11, 21), (11, 15), (11, 14), (11, 13),
    (12, 25), (12, 24), (12, 23), (12, 22), (12, 21), (12, 20), (12, 19), (12, 18), (12, 17), (12, 16), (12, 15), (12, 14), (12, 13),
    (13, 23), (13, 22), (13, 21), (13, 20), (13, 19), (13, 18), (13, 17), (13, 16), (13, 15), (13, 14),
    (14, 21), (14, 20), (14, 19), (14, 18), (14, 17), (14, 16), (14, 15),
]

def draw_grid(frame, angle):
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = row * CELL_SIZE + CELL_SIZE // 2
            
            color = (0, 165, 255) if (row, col) in SWOOSH_CELLS else (255, 255, 255)
            
            radian = np.deg2rad(angle)
            
            x1 = int(x - (LINE_LENGTH / 2) * np.cos(radian))
            y1 = int(y - (LINE_LENGTH / 2) * np.sin(radian))
            x2 = int(x + (LINE_LENGTH / 2) * np.cos(radian))
            y2 = int(y + (LINE_LENGTH / 2) * np.sin(radian))
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def main():
    angle = 0  # Initial angle for synchronized rotation
    
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        angle = (angle + 2) % 360
        
        draw_grid(frame, angle)
        
        cv2.imshow("Interactive Projection Grid", frame)
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
