import cv2
import numpy as np

# Grid settings
GRID_ROWS = 20
GRID_COLS = 40
CELL_SIZE = 20
LINE_LENGTH = 15

SYNCED_ANGLE = 90
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
