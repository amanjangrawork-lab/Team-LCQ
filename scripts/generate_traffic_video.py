import cv2
import numpy as np
from pathlib import Path


def generate_synthetic_traffic_video(output_path, num_frames=300, vehicle_count=12, fps=30):
    """Generate a realistic-looking traffic video with moving vehicles."""
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    vehicles = []
    np.random.seed(42)
    for _ in range(vehicle_count):
        vehicles.append({
            'start_x': np.random.randint(-200, 200),
            'y': np.random.choice([430, 500, 570]),
            'w': np.random.randint(60, 110),
            'h': np.random.randint(40, 70),
            'plate': f"{np.random.choice(['MH','DL','KA','TN'])}{np.random.randint(10,99)}{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{np.random.randint(1000,9999)}",
            'speed': np.random.randint(3, 12),
            'color': tuple(map(int, np.random.randint(40, 200, 3))),
        })
    
    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky
        frame[:, :] = (135, 206, 235)
        # Road
        frame[400:650, :] = (80, 80, 80)
        # Lane markings
        for y in range(420, 630, 60):
            for x in range(0, width, 120):
                cv2.rectangle(frame, (x, y), (x+60, y+10), (255, 255, 255), -1)
        
        for v in vehicles:
            v['start_x'] += v['speed']
            x = v['start_x'] % 1400 - 200
            cv2.rectangle(frame, (int(x), v['y']), (int(x+v['w']), v['y']+v['h']), v['color'], -1)
            cv2.rectangle(frame, (int(x+5), v['y']+5), (int(x+v['w']-5), v['y']+v['h']//2), (150, 200, 230), -1)
            cv2.circle(frame, (int(x), v['y']+v['h']//2), 3, (255, 255, 0), -1)
            cv2.circle(frame, (int(x+v['w']), v['y']+v['h']//2), 3, (255, 255, 0), -1)
        
        cv2.putText(frame, f"CAM_01 | {frame_idx//fps:02d}:{(frame_idx%fps)//60:02d}:{frame_idx%60:02d}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    return num_frames, vehicle_count


def generate_traffic_snapshot(output_path, vehicle_count=8, width=1280, height=720):
    """Generate a single traffic scene snapshot for dashboard screenshots."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (135, 206, 235)
    frame[400:650, :] = (80, 80, 80)
    
    for y in range(420, 630, 60):
        for x in range(0, width, 120):
            cv2.rectangle(frame, (x, y), (x+60, y+10), (255, 255, 255), -1)
    
    for i in range(vehicle_count):
        x = np.random.randint(50, width-150)
        y = np.random.choice([430, 500, 570])
        w = np.random.randint(60, 110)
        h = np.random.randint(40, 70)
        color = tuple(map(int, np.random.randint(40, 200, 3)))
        plate = f"{np.random.choice(['MH','DL','KA'])}{np.random.randint(10,99)}{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{np.random.randint(1000,9999)}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+h//2), (150, 200, 230), -1)
        cv2.circle(frame, (x, y+h//2), 3, (255, 255, 0), -1)
        cv2.circle(frame, (x+w, y+h//2), 3, (255, 255, 0), -1)
    
    cv2.putText(frame, "LIVE STREAM - CAM_01", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(output_path, frame)
    return frame
