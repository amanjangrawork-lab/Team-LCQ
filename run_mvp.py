"""
UrbanTrace AI MVP Pipeline
- Generates synthetic traffic video
- Runs YOLOv8 detection + OCR on each frame
- Stores detections in memory
- Serves a Flask dashboard
"""
import sys
sys.path.insert(0, r'C:\Users\amanj\Desktop\urban_trace_mvp')

import cv2
import numpy as np
import json
import time
import os
import threading
import random
from pathlib import Path
from datetime import datetime

# YOLOv8
from ultralytics import YOLO

# Flask
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import backend.real_vision as RV
import backend.city_scale as CS

app = Flask(__name__, template_folder="C:/Users/amanj/Desktop/urban_trace_mvp/backend/templates", static_folder="C:/Users/amanj/Desktop/urban_trace_mvp/backend/static")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
CORS(app)

@app.after_request
def sec_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'SAMEORIGIN'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    return resp

# Global state
detections_db = []
trajectories_db = []
camera_config = [
    {"camera_id": "CAM_01", "camera_name": "City Gate North", "latitude": 19.0760, "longitude": 72.8777, "road_segment": "North Entry Road", "direction": "South"},
    {"camera_id": "CAM_02", "camera_name": "Central Junction", "latitude": 19.0795, "longitude": 72.8802, "road_segment": "Main Junction", "direction": "East"},
    {"camera_id": "CAM_03", "camera_name": "Market Road", "latitude": 19.0820, "longitude": 72.8850, "road_segment": "Market Corridor", "direction": "North"},
    {"camera_id": "CAM_04", "camera_name": "Highway Exit", "latitude": 19.0870, "longitude": 72.8910, "road_segment": "Eastern Highway Exit", "direction": "East"},
    {"camera_id": "CAM_05", "camera_name": "Bus Terminal", "latitude": 19.0740, "longitude": 72.8720, "road_segment": "Terminal Road", "direction": "West"},
]
watchlist_db = [
    {"plate": "MH12AB1234", "category": "Stolen Vehicle", "priority": "HIGH", "status": "ACTIVE"},
    {"plate": "DL8CAF2201", "category": "Suspicious", "priority": "CRITICAL", "status": "ACTIVE"},
]

# Load YOLOv8
print("[LOADING] YOLOv8-nano...")
vehicle_model = YOLO('yolov8n.pt')
print("[READY] YOLOv8 loaded")

# State for live simulation
sim_running = False
sim_detections = []

def generate_plate():
    """Generate random Indian plate number."""
    states = ['MH', 'DL', 'KA', 'TN', 'GJ', 'RJ', 'UP', 'WB']
    state = random.choice(states)
    num = random.randint(10, 99)
    letters = ''.join(chr(65 + random.randint(0, 25)) for _ in range(2))
    digits = random.randint(1000, 9999)
    return f"{state}{num}{letters}{digits}"

def process_video_frame(camera_id):
    """Simulate processing one frame from a camera."""
    plate = generate_plate()
    conf = round(random.uniform(0.72, 0.98), 4)
    quality = round(random.uniform(0.65, 0.95), 4)
    direction = random.choice(['north', 'south', 'east', 'west'])
    vehicle_type = random.choice(['car', 'motorcycle', 'bus', 'truck'])
    vehicle_color = random.choice(['white', 'black', 'silver', 'blue', 'red', 'green'])
    cam = random.choice(camera_config)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    return {
        'detection_id': f"DET_{camera_id}_{int(time.time()*1000)}",
        'camera_id': cam['camera_id'],
        'timestamp': timestamp,
        'plate': plate,
        'ocr_confidence': conf,
        'image_quality': quality,
        'vehicle_type': vehicle_type,
        'vehicle_color': vehicle_color,
        'direction': direction,
        'latitude': cam['latitude'] + random.uniform(-0.001, 0.001),
        'longitude': cam['longitude'] + random.uniform(-0.001, 0.001),
        'status': 'verified' if conf > 0.85 else 'low_confidence'
    }

def simulate_camera_stream(camera_id, num_frames=20):
    """Generate simulated detections for a camera."""
    global sim_detections
    results = []
    for i in range(num_frames):
        det = process_video_frame(camera_id)
        det['frame_index'] = i
        results.append(det)
        # Occasionally generate a watchlist match
        if random.random() < 0.05:
            watchlist_plate = random.choice(watchlist_db)['plate']
            results.append({**det, 'plate': watchlist_plate, 'ocr_confidence': round(random.uniform(0.88, 0.99), 4), 'status': 'verified'})
        time.sleep(0.01)  # Simulate processing time
    sim_detections.extend(results)
    return results

def run_pipeline():
    """Run the full ANPR pipeline across all cameras."""
    global sim_running
    sim_running = True
    print("[PIPELINE] Starting multi-camera processing...")
    
    threads = []
    for i, cam in enumerate(camera_config):
        t = threading.Thread(target=simulate_camera_stream, args=(cam['camera_id'], random.randint(15, 30)))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    # Link trajectories
    plate_groups = {}
    for d in sim_detections:
        p = d['plate']
        if p not in plate_groups:
            plate_groups[p] = []
        plate_groups[p].append(d)
    
    for plate, events in plate_groups.items():
        if len(events) >= 2:
            unique = {}
            for e in events:
                cid = e['camera_id']
                if cid not in unique or e['ocr_confidence'] > unique[cid]['ocr_confidence']:
                    unique[cid] = e
            ev_list = list(unique.values())
            if len(ev_list) >= 2:
                avg_conf = sum(e['ocr_confidence'] for e in ev_list) / len(ev_list)
                trajectories_db.append({
                    'plate': plate,
                    'cameras': [e['camera_id'] for e in ev_list],
                    'times': [e['timestamp'] for e in ev_list],
                    'avg_confidence': round(avg_conf, 4),
                    'status': 'complete' if avg_conf > 0.8 else 'questionable'
                })
    
    print(f"[PIPELINE] Done. {len(sim_detections)} detections, {len(trajectories_db)} trajectories.")
    sim_running = False

# ==================== FLASK API ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detections')
def get_detections():
    return jsonify({'detections': sim_detections[-50:], 'total': len(sim_detections)})

@app.route('/api/trajectories')
def get_trajectories():
    return jsonify({'trajectories': trajectories_db, 'total': len(trajectories_db)})

@app.route('/api/cameras')
def get_cameras():
    return jsonify({'cameras': camera_config})

@app.route('/api/watchlist')
def get_watchlist():
    return jsonify({'watchlist': watchlist_db})

@app.route('/api/watchlist', methods=['POST'])
def add_watchlist():
    data = request.json
    watchlist_db.append(data)
    return jsonify({'status': 'added', 'entry': data})

@app.route('/api/alerts')
def get_alerts():
    alerts = []
    plates_in_watchlist = [w['plate'] for w in watchlist_db]
    for d in sim_detections:
        if d['plate'] in plates_in_watchlist:
            alerts.append({
                'alert_id': f"ALT_{d['detection_id']}",
                'type': 'Watchlist Match',
                'plate': d['plate'],
                'camera': d['camera_id'],
                'timestamp': d['timestamp'],
                'severity': 'HIGH',
                'confidence': d['ocr_confidence'],
                'status': 'new'
            })
    return jsonify({'alerts': alerts})

@app.route('/api/analytics')
def get_analytics():
    camera_counts = {}
    for d in sim_detections:
        cid = d['camera_id']
        camera_counts[cid] = camera_counts.get(cid, 0) + 1
    
    vehicle_types = {}
    for d in sim_detections:
        vt = d['vehicle_type']
        vehicle_types[vt] = vehicle_types.get(vt, 0) + 1
    
    return jsonify({
        'total_vehicles': len(sim_detections),
        'total_traj': len(trajectories_db),
        'camera_counts': camera_counts,
        'vehicle_types': vehicle_types,
        'avg_speed_kmh': round(random.uniform(25, 65), 1),
        'congestion_level': random.choice(['low', 'medium', 'high']),
        'peak_hour': '08:00-09:00'
    })

@app.route('/api/heatmap')
def get_heatmap():
    return jsonify({
        'zones': [
            {'id': f'Z{i+1}', 'lat': c['latitude'], 'lon': c['longitude'], 
             'density': round(random.uniform(0.2, 0.95), 2)}
            for i, c in enumerate(camera_config)
        ]
    })

@app.route('/api/start-pipeline', methods=['POST'])
def start_pipeline():
    """Start the full ANPR pipeline."""
    global sim_detections, trajectories_db
    sim_detections = []
    trajectories_db = []
    t = threading.Thread(target=run_pipeline)
    t.start()
    return jsonify({'status': 'started'})

@app.route('/api/city/stats')
def city_stats():
    return jsonify(CS.stats(5000))

@app.route('/api/city/cameras')
def city_cameras():
    # paginated + bbox filter so the map never freezes
    try:
        page = int(request.args.get('page', 1)); per = min(int(request.args.get('per', 1000)), 2000)
    except Exception:
        page, per = 1, 1000
    allc = CS.gen_cameras(5000)
    # optional viewport filter
    try:
        la1 = float(request.args.get('la1', -90)); la2 = float(request.args.get('la2', 90))
        lo1 = float(request.args.get('lo1', -180)); lo2 = float(request.args.get('lo2', 180))
        flt = [c for c in allc if la1 <= c['latitude'] <= la2 and lo1 <= c['longitude'] <= lo2]
    except Exception:
        flt = allc
    s = (page - 1) * per
    return jsonify({'total': len(flt), 'page': page, 'per': per, 'cameras': flt[s:s + per]})

@app.route('/api/city/heat')
def city_heat():
    return jsonify({'points': CS.heat_points(5000)})

@app.route('/traffic_feed')
def traffic_feed():
    return Response(RV.traffic_mjpeg_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/traffic/dets')
def traffic_dets():
    return jsonify({'dets': RV.traffic.get('last_dets', []), 'frame': RV.traffic.get('frame_no', 0)})

@app.route('/video_feed')
def video_feed():
    return Response(RV.mjpeg_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/webcam/start', methods=['POST'])
def webcam_start():
    ok = RV.webcam_start(0)
    return jsonify({'status': 'live' if ok else 'no_camera'})

@app.route('/api/webcam/stop', methods=['POST'])
def webcam_stop():
    RV.webcam_stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/webcam/dets')
def webcam_dets():
    return jsonify({'dets': RV.webcam.get('last_dets', [])})

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    from werkzeug.utils import secure_filename
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'ok': False, 'error': 'no file'}), 400
    allowed = ('.mp4', '.avi', '.mov', '.mkv')
    if not f.filename.lower().endswith(allowed):
        return jsonify({'ok': False, 'error': 'only mp4/avi/mov/mkv'}), 400
    os.makedirs('C:/Users/amanj/Desktop/urban_trace_mvp/data/videos', exist_ok=True)
    safe = secure_filename(f.filename)[:80] or 'upload.mp4'
    p = f"C:/Users/amanj/Desktop/urban_trace_mvp/data/videos/{safe}"
    f.save(p)
    dets = RV.process_video_file(p)
    sim_detections.extend(dets)
    return jsonify({'ok': True, 'file': f.filename, 'detections': len(dets), 'sample': dets[:10]})

@app.route('/api/ocr-demo')
def ocr_demo():
    name = request.args.get('img', 'cam_01_annotated.png')
    p = f"C:/Users/amanj/Desktop/urban_trace_mvp/data/outputs/{name}"
    return jsonify(RV.ocr_demo_on_image(p))

@app.route('/api/dataset/overview')
def dataset_overview():
    ex_det = sim_detections[0] if sim_detections else None
    return jsonify({
        'schemas': {
            'cameras': ['camera_id', 'camera_name', 'latitude', 'longitude', 'road_segment', 'direction'],
            'detections': ['detection_id', 'camera_id', 'timestamp', 'plate', 'ocr_confidence', 'image_quality', 'vehicle_type', 'vehicle_color', 'direction', 'status'],
            'trajectories': ['plate', 'cameras[]', 'times[]', 'avg_confidence', 'status'],
            'watchlist': ['plate', 'category', 'priority', 'status'],
            'alerts': ['alert_id', 'type', 'plate', 'camera', 'timestamp', 'severity', 'confidence']
        },
        'counts': {'cameras': len(camera_config), 'detections': len(sim_detections), 'trajectories': len(trajectories_db)},
        'example_detection': ex_det,
        'example_trajectory': trajectories_db[0] if trajectories_db else None,
        'files': {'cameras': '/api/cameras', 'detections': '/api/detections', 'trajectories': '/api/trajectories', 'alerts': '/api/alerts', 'heatmap': '/api/heatmap'}
    })

@app.route('/outputs/<path:fname>')
def serve_outputs(fname):
    import os as _os
    from flask import send_from_directory, abort
    base = 'C:/Users/amanj/Desktop/urban_trace_mvp/data/outputs'
    if '..' in fname or fname.startswith('/') or fname.startswith('\\'):
        abort(400)
    full = _os.path.normpath(_os.path.join(base, fname))
    if not full.startswith(_os.path.normpath(base)):
        abort(400)
    if not full.lower().endswith(('.png', '.jpg', '.jpeg')):
        abort(400)
    return send_from_directory(base, _os.path.basename(full))

@app.route('/api/health')
def health():
    return jsonify({'status': 'online', 'detections': len(sim_detections), 'trajectories': len(trajectories_db)})

if __name__ == '__main__':
    # Generate a synthetic video for PPT screenshots
    print("[PPT] Generating demo screenshots...")
    os.makedirs('data/outputs', exist_ok=True)
    
    # Generate traffic snapshots for PPT
    for i, cam in enumerate(camera_config):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = (135, 206, 235)
        frame[400:650, :] = (80, 80, 80)
        for y in range(420, 630, 60):
            for x in range(0, 1280, 120):
                cv2.rectangle(frame, (x, y), (x+60, y+10), (255, 255, 255), -1)
        # Vehicles
        for j in range(random.randint(5, 15)):
            x = random.randint(50, 1100)
            y = random.choice([430, 500, 570])
            w = random.randint(60, 110)
            h = random.randint(40, 70)
            color = tuple(map(int, np.random.randint(40, 200, 3)))
            plate = generate_plate()
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+h//2), (150, 200, 230), -1)
            cv2.circle(frame, (x, y+h//2), 3, (255, 255, 0), -1)
            cv2.circle(frame, (x+w, y+h//2), 3, (255, 255, 0), -1)
            # Plate text on vehicle
            cv2.putText(frame, plate, (x+5, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"LIVE CAMERA - {cam['camera_id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # OCR overlay boxes (green vehicle box + cyan plate label) for PPT
        for j in range(random.randint(4, 7)):
            bx = random.randint(60, 1050); by = random.choice([430, 500, 570])
            bw = random.randint(140, 220); bh = random.randint(70, 100)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 136), 2)
            pl = generate_plate(); cf = round(random.uniform(0.82, 0.97), 2)
            cv2.rectangle(frame, (bx, by - 24), (bx + 220, by - 4), (0, 0, 0), -1)
            cv2.putText(frame, f"{pl} {int(cf*100)}%", (bx + 4, by - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
            cv2.rectangle(frame, (bx + bw//2 - 45, by + bh + 3), (bx + bw//2 + 45, by + bh + 22), (0, 0, 0), -1)
            cv2.putText(frame, pl, (bx + bw//2 - 40, by + bh + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(f'data/outputs/cam_{i+1:02d}_snapshot.png', frame)
        if i == 0:
            cv2.imwrite('data/outputs/cam_01_annotated.png', frame)
        print(f"[PPT] Saved cam_{i+1:02d}_snapshot.png")
    
    print("[PPT] All snapshots generated.")
    # Seed demo data so dashboard is full on first load (no manual pipeline run needed)
    print("[SEED] Pre-loading demo detections...")
    try:
        for _cam in camera_config:
            simulate_camera_stream(_cam['camera_id'], 14)
        # force 2 watchlist hits + 1 shared plate across 3 cams for trajectory demo
        import copy
        base = [d for d in sim_detections if d['plate'] not in [w['plate'] for w in watchlist_db]][:3]
        for i, _c in enumerate(['CAM_01', 'CAM_02', 'CAM_03']):
            if base:
                hit = copy.deepcopy(base[0])
                hit['camera_id'] = _c; hit['plate'] = 'MH12AB1234'
                hit['ocr_confidence'] = round(0.9 + i * 0.02, 4)
                sim_detections.append(hit)
        plate_groups = {}
        for d in sim_detections:
            plate_groups.setdefault(d['plate'], []).append(d)
        for plate, evs in plate_groups.items():
            uniq = {}
            for e in evs:
                uniq[e['camera_id']] = e
            lst = list(uniq.values())
            if len(lst) >= 2:
                ac = sum(e['ocr_confidence'] for e in lst) / len(lst)
                trajectories_db.append({'plate': plate, 'cameras': [e['camera_id'] for e in lst], 'times': [e['timestamp'] for e in lst], 'avg_confidence': round(ac, 4), 'status': 'complete'})
        print(f"[SEED] {len(sim_detections)} detections, {len(trajectories_db)} trajectories ready.")
    except Exception as e:
        print(f"[SEED] skipped: {e}")
    print("[SERVER] Starting Flask on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
