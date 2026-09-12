from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# In-memory storage (replace with DB in production)
detections_db = []
trajectories_db = []
analytics_db = {}
watchlist_db = [
    {"plate": "MH12AB1234", "category": "Stolen Vehicle", "priority": "HIGH", "status": "ACTIVE"},
    {"plate": "KA01MN4567", "category": "Suspicious", "priority": "MEDIUM", "status": "ACTIVE"},
]
camera_config = [
    {"camera_id": "CAM_01", "camera_name": "City Gate North", "latitude": 19.0760, "longitude": 72.8777, "road_segment": "North Entry Road", "direction": "South", "status": "ONLINE"},
    {"camera_id": "CAM_02", "camera_name": "Central Junction", "latitude": 19.0795, "longitude": 72.8802, "road_segment": "Main Junction", "direction": "East", "status": "ONLINE"},
    {"camera_id": "CAM_03", "camera_name": "Market Road", "latitude": 19.0820, "longitude": 72.8850, "road_segment": "Market Corridor", "direction": "North", "status": "ONLINE"},
    {"camera_id": "CAM_04", "camera_name": "Highway Exit", "latitude": 19.0870, "longitude": 72.8910, "road_segment": "Eastern Highway Exit", "direction": "East", "status": "ONLINE"},
    {"camera_id": "CAM_05", "camera_name": "Bus Terminal", "latitude": 19.0740, "longitude": 72.8720, "road_segment": "Terminal Road", "direction": "West", "status": "ONLINE"},
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detections', methods=['GET'])
def get_detections():
    return jsonify({"detections": detections_db, "total": len(detections_db)})

@app.route('/api/detections', methods=['POST'])
def add_detection():
    data = request.json
    data['id'] = len(detections_db) + 1
    detections_db.append(data)
    return jsonify({"status": "added", "detection": data})

@app.route('/api/trajectories', methods=['GET'])
def get_trajectories():
    return jsonify({"trajectories": trajectories_db, "total": len(trajectories_db)})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    return jsonify({"watchlist": watchlist_db})

@app.route('/api/watchlist', methods=['POST'])
def add_watchlist():
    data = request.json
    watchlist_db.append(data)
    return jsonify({"status": "added", "entry": data})

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    return jsonify({"cameras": camera_config})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if not analytics_db:
        # Generate summary
        camera_counts = {}
        for d in detections_db:
            cid = d.get('camera_id', 'unknown')
            camera_counts[cid] = camera_counts.get(cid, 0) + 1
        analytics_db = {
            "total_vehicles": len(detections_db),
            "camera_counts": camera_counts,
            "vehicle_types": {"car": 65, "motorcycle": 20, "bus": 8, "truck": 7},
            "average_speed_kmh": 42.5,
            "congestion_level": "medium",
            "peak_hour": "08:00-09:00",
        }
    return jsonify(analytics_db)

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = []
    for d in detections_db:
        if d.get('plate') in [w['plate'] for w in watchlist_db]:
            alerts.append({
                "alert_id": f"ALT_{d.get('id', 0)}",
                "alert_type": "Watchlist Match",
                "plate": d.get('plate'),
                "camera_id": d.get('camera_id'),
                "timestamp": d.get('timestamp'),
                "severity": "HIGH",
                "confidence": d.get('ocr_confidence', 0),
                "status": "new"
            })
    return jsonify({"alerts": alerts})

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Trigger video processing pipeline."""
    data = request.json
    video_path = data.get('video_path', '')
    camera_id = data.get('camera_id', 'CAM_01')
    return jsonify({"status": "processing", "camera": camera_id, "video": video_path})

@app.route('/api/heatmap', methods=['GET'])
def get_heatmap():
    return jsonify({
        "zones": [
            {"id": "Z1", "lat": 19.0760, "lon": 72.8777, "density": 0.85},
            {"id": "Z2", "lat": 19.0795, "lon": 72.8802, "density": 0.72},
            {"id": "Z3", "lat": 19.0820, "lon": 72.8850, "density": 0.45},
            {"id": "Z4", "lat": 19.0870, "lon": 72.8910, "density": 0.91},
            {"id": "Z5", "lat": 19.0740, "lon": 72.8720, "density": 0.33},
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
