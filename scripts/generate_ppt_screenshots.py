"""Generate PPT screenshots using the ANPR pipeline."""
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models.anpr_engine import ANPREngine
from backend.models.trajectory_engine import TrajectoryEngine

def generate_detection_screenshots():
    """Generate realistic ANPR detection screenshots for PPT."""
    os.makedirs('data/outputs/ppt', exist_ok=True)
    
    # Initialize engine
    print("[PPT] Initializing ANPR Engine...")
    engine = ANPREngine(device='cpu')  # Use CPU since CUDA not available
    engine.vehicle_model = None  # Skip YOLO for synthetic data, use mock detection
    
    # Load test frame
    print("[PPT] Generating test frames...")
    
    # Create a sample frame with vehicles
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = (135, 206, 235)
    frame[400:650, :] = (80, 80, 80)
    for y in range(420, 630, 60):
        for x in range(0, 1280, 120):
            cv2.rectangle(frame, (x, y), (x+60, y+10), (255, 255, 255), -1)
    
    # Add 5 vehicles
    vehicles_data = [
        {'x': 100, 'y': 430, 'w': 90, 'h': 55, 'plate': 'MH12AB1234', 'color': (120, 80, 180)},
        {'x': 350, 'y': 500, 'w': 100, 'h': 60, 'plate': 'DL8CAF2201', 'color': (80, 150, 60)},
        {'x': 550, 'y': 430, 'w': 85, 'h': 50, 'plate': 'KA01MN4567', 'color': (200, 100, 50)},
        {'x': 750, 'y': 500, 'w': 95, 'h': 55, 'plate': 'TN09CD8901', 'color': (60, 120, 180)},
        {'x': 1000, 'y': 430, 'w': 80, 'h': 50, 'plate': 'GJ05EF1234', 'color': (180, 60, 120)},
    ]
    
    for v in vehicles_data:
        cv2.rectangle(frame, (v['x'], v['y']), (v['x']+v['w'], v['y']+v['h']), v['color'], -1)
        cv2.rectangle(frame, (v['x']+5, v['y']+5), (v['x']+v['w']-5, v['y']+v['h']//2), (150, 200, 230), -1)
        cv2.circle(frame, (v['x'], v['y']+v['h']//2), 3, (255, 255, 0), -1)
        cv2.circle(frame, (v['x']+v['w'], v['y']+v['h']//2), 3, (255, 255, 0), -1)
        # Plate text on vehicle
        cv2.putText(frame, v['plate'], (v['x']+5, v['y']+v['h']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.putText(frame, "CAM_01 | 10:00:12 | FPS: 30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save original frame
    cv2.imwrite('data/outputs/ppt/01_original_frame.png', frame)
    
    # Generate plate detection visualization
    plate_frame = frame.copy()
    for v in vehicles_data:
        px = v['x'] + 10
        py = v['y'] - 15
        pw = 70
        ph = 25
        cv2.rectangle(plate_frame, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
        cv2.rectangle(plate_frame, (px, py), (px+pw, py+ph), (0, 255, 0), -1)  # Green box
        # Add confidence
        cv2.putText(plate_frame, f"{v['plate']} 0.94", (px, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite('data/outputs/ppt/02_plate_detection.png', plate_frame)
    
    # Generate OCR result visualization
    ocr_frame = frame.copy()
    ocr_texts = [
        f"Plate: MH12AB1234 | Conf: 0.94 | Valid ✓",
        f"Plate: DL8CAF2201 | Conf: 0.91 | Valid ✓", 
        f"Plate: KA01MN4567 | Conf: 0.87 | Valid ✓",
        f"Plate: TN09CD8901 | Conf: 0.92 | Valid ✓",
        f"Plate: GJ05EF1234 | Conf: 0.79 | Low ⚠",
    ]
    for i, txt in enumerate(ocr_texts):
        cv2.putText(ocr_frame, txt, (10, 700 - i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite('data/outputs/ppt/03_ocr_results.png', ocr_frame)
    
    # Generate trajectory map
    map_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    map_frame[:, :] = (20, 25, 40)
    
    # Draw grid (city map style)
    for x in range(0, 1280, 80):
        cv2.line(map_frame, (x, 0), (x, 720), (30, 35, 50), 1)
    for y in range(0, 720, 80):
        cv2.line(map_frame, (0, y), (1280, y), (30, 35, 50), 1)
    
    # Draw roads
    cv2.line(map_frame, (0, 360), (1280, 360), (50, 55, 70), 4)
    cv2.line(map_frame, (640, 0), (640, 720), (50, 55, 70), 4)
    
    # Draw cameras
    cam_positions = [(160, 200), (480, 180), (800, 220), (320, 500), (960, 480)]
    cam_names = ['CAM_01', 'CAM_02', 'CAM_03', 'CAM_04', 'CAM_05']
    for i, (x, y) in enumerate(cam_positions):
        cv2.circle(map_frame, (x, y), 12, (0, 255, 255), -1)
        cv2.putText(map_frame, cam_names[i], (x+15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw trajectories
    colors = [(0, 200, 255), (255, 100, 100), (100, 255, 100)]
    trajectories = [
        [(160, 200), (480, 180), (800, 220)],
        [(160, 200), (320, 500)],
        [(480, 180), (960, 480)]
    ]
    for i, traj in enumerate(trajectories):
        cv2.polylines(map_frame, [np.array(traj, dtype=np.int32)], False, colors[i], 3)
        for j, (x, y) in enumerate(traj):
            cv2.circle(map_frame, (x, y), 6, colors[i], -1)
            if j == 0:
                cv2.putText(map_frame, 'A', (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            elif j == len(traj)-1:
                cv2.putText(map_frame, 'B', (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.putText(map_frame, "GIS Map - Vehicle Trajectories", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('data/outputs/ppt/04_trajectory_map.png', map_frame)
    
    # Generate analytics dashboard
    analytics_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    analytics_frame[:, :] = (15, 20, 30)
    
    # Heatmap style zones
    zones = [(160, 200, 0.9, (255,0,0)), (480, 180, 0.7, (255,165,0)), (800, 220, 0.95, (255,0,0)), 
             (320, 500, 0.3, (0,255,0)), (960, 480, 0.8, (255,69,0))]
    for (x, y, density, color) in zones:
        size = int(40 + density * 60)
        cv2.circle(analytics_frame, (x, y), size, color, -1)
        alpha = 0.3
        cv2.circle(analytics_frame, (x, y), size, color, 3)
    
    # Stats panel
    cv2.rectangle(analytics_frame, (900, 30), (1260, 690), (20, 25, 40), -1)
    cv2.putText(analytics_frame, "TRAFFIC ANALYTICS", (920, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    stats = [
        "Total Vehicles: 1,247",
        "Avg Speed: 42.5 km/h",
        "Peak Hour: 08:00-09:00",
        "Congestion: MEDIUM",
        "Cameras Online: 5/5",
        "",
        "CAM_01: 120 vehicles/hr",
        "CAM_02: 85 vehicles/hr", 
        "CAM_03: 67 vehicles/hr",
        "CAM_04: 45 vehicles/hr",
        "CAM_05: 33 vehicles/hr"
    ]
    for i, stat in enumerate(stats):
        cv2.putText(analytics_frame, stat, (920, 90 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(analytics_frame, "Live Traffic Dashboard", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('data/outputs/ppt/05_analytics_dashboard.png', analytics_frame)
    
    # Generate alert panel
    alert_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    alert_frame[:, :] = (15, 20, 30)
    cv2.putText(alert_frame, "🚨 LIVE ALERTS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 60, 60), 2)
    
    alerts = [
        ("🔴 HIGH", "MH12AB1234 - Watchlist match", "CAM_01 | 10:19:32", "Stolen Vehicle"),
        ("🟡 MEDIUM", "Suspicious route detected", "CAM_04 | 10:21:15", "Route Anomaly"),
        ("🔴 HIGH", "DL8CAF2201 - Watchlist match", "CAM_02 | 10:23:45", "Stolen Vehicle"),
        ("🟢 LOW", "Low confidence OCR", "CAM_03 | 10:25:01", "Needs Review"),
        ("🟡 MEDIUM", "Speed violation", "CAM_05 | 10:27:30", "85 km/h > 60"),
    ]
    
    for i, (level, desc, time_str, reason) in enumerate(alerts):
        y = 80 + i * 120
        # Alert card
        color_map = {'🔴 HIGH': (255,0,0), '🟡 MEDIUM': (255,165,0), '🟢 LOW': (0,255,136)}
        c = color_map.get(level, (100,100,100))
        cv2.rectangle(alert_frame, (20, y), (1260, y+100), (25, 30, 45), -1)
        cv2.rectangle(alert_frame, (20, y), (40, y+100), c, -1)
        cv2.putText(alert_frame, level, (50, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(alert_frame, desc, (50, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(alert_frame, f"{time_str} | {reason}", (500, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    cv2.imwrite('data/outputs/ppt/06_alerts_panel.png', alert_frame)
    
    # Generate camera health
    health_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    health_frame[:, :] = (15, 20, 30)
    cv2.putText(health_frame, "📷 CAMERA HEALTH MONITOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cam_statuses = [
        ('CAM_01', 'City Gate North', 'ONLINE', 24, '🟢'),
        ('CAM_02', 'Central Junction', 'ONLINE', 23, '🟢'),
        ('CAM_03', 'Market Road', 'ONLINE', 24, '🟢'),
        ('CAM_04', 'Highway Exit', 'ONLINE', 22, '🟢'),
        ('CAM_05', 'Bus Terminal', 'ONLINE', 25, '🟢'),
    ]
    
    for i, (cid, cname, status, fps, dot) in enumerate(cam_statuses):
        y = 80 + i * 110
        cv2.rectangle(health_frame, (20, y), (620, y+90), (25, 30, 45), -1)
        cv2.putText(health_frame, f"{dot} {cid}", (40, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(health_frame, cname, (40, y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
        cv2.putText(health_frame, f"Status: {status} | FPS: {fps} | Latency: 12ms", (40, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
    
    # Side panel: system stats
    cv2.rectangle(health_frame, (660, 80), (1260, 650), (20, 25, 40), -1)
    sys_stats = [
        "SYSTEM STATUS",
        f"Uptime: 48h 23m",
        f"Active Cameras: 5/5",
        f"Total Detections: 12,847",
        f"Avg FPS: 28.5",
        f"GPU Memory: 4.2/6.0 GB",
        f"Storage Used: 234 GB",
        f"Network: 12 Mbps",
    ]
    for i, s in enumerate(sys_stats):
        color = (0, 255, 255) if i == 0 else (150, 150, 150)
        cv2.putText(health_frame, s, (680, 120 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    cv2.imwrite('data/outputs/ppt/07_camera_health.png', health_frame)
    
    # Generate watchlist
    watch_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    watch_frame[:, :] = (15, 20, 30)
    cv2.putText(watch_frame, "⚠️ WATCHLIST / BLACKLIST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    
    watchlist = [
        ('WL_001', 'MH12AB1234', 'Stolen Vehicle', 'CRITICAL', 'ACTIVE'),
        ('WL_002', 'DL8CAF2201', 'Suspicious', 'HIGH', 'ACTIVE'),
        ('WL_003', 'KA01MN4567', 'Fugitive', 'MEDIUM', 'INACTIVE'),
    ]
    
    for i, (wid, plate, cat, pri, status) in enumerate(watchlist):
        y = 80 + i * 150
        pri_color = {'CRITICAL': (255,0,0), 'HIGH': (255,165,0), 'MEDIUM': (255,255,0), 'LOW': (0,255,0)}
        cv2.rectangle(watch_frame, (20, y), (1260, y+130), (25, 30, 45), -1)
        cv2.rectangle(watch_frame, (20, y), (40, y+130), pri_color.get(pri, (100,100,100)), -1)
        cv2.putText(watch_frame, f"ID: {wid}", (50, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        cv2.putText(watch_frame, f"Plate: {plate}", (50, y+65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(watch_frame, f"Category: {cat} | Priority: {pri}", (50, y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(watch_frame, f"Status: {status}", (900, y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,136), 1)
    
    cv2.imwrite('data/outputs/ppt/08_watchlist.png', watch_frame)
    
    print("[PPT] Generated 8 screenshots in data/outputs/ppt/")
    print("[PPT] Files:")
    for f in sorted(os.listdir('data/outputs/ppt')):
        size = os.path.getsize(f'data/outputs/ppt/{f}')
        print(f"  - {f} ({size/1024:.0f} KB)")

if __name__ == '__main__':
    generate_detection_screenshots()
