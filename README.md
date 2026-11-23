# NetraPath — City-Wide AI Engine for Multi-Camera ANPR, Trajectory Tracking and Urban Traffic Analytics

**SIH 2026 — Problem Statement 26127 | Bharat Electronics Limited (BEL)**

NetraPath transforms a city's existing CCTV/ANPR camera network from a collection of isolated detection points into a unified, privacy-aware urban mobility intelligence system. It links every plate detection across cameras into a continuous vehicle timeline, reconstructs the actual route driven on a GIS map, and converts the aggregated detection stream into macro traffic analytics — congestion heatmaps, origin-destination patterns, and speed estimates — all visible on a single web dashboard in real time.

---

## What It Does

- **Multi-camera ANPR** — YOLOv8 vehicle + plate detection, PaddleOCR/RapidOCR character recognition, temporal OCR fusion across frames for ≥ 90 % accuracy.
- **Trajectory reconstruction** — Links sightings of the same plate across cameras into a time-ordered route with GIS route line, evidence at each node, and route confidence score.
- **Real-time alerts** — Blacklist/watchlist match, geofence violation, wrong-way, speed violation, cloned plate, route anomaly, camera down — delivered via WebSocket in < 5 s.
- **Traffic analytics** — City-wide heatmap, origin-destination matrix, camera-to-camera flow, congestion detection, peak-hour patterns.
- **Privacy-by-design** — HMAC-SHA256 plate hashing for analytics, AES-256 encrypted plate storage, RBAC (4 roles), immutable audit logs. DPDP Act 2023 compliant.
- **Human-in-the-loop** — Low-confidence OCR flagged for review; all alerts require operator acknowledgement; no automatic enforcement.

---

## Technology Stack

| Layer | Technology |
|---|---|
| AI/CV | Python, OpenCV, YOLOv8-nano, PaddleOCR / RapidOCR, ByteTrack |
| Backend | Flask (MVP) / FastAPI (production), PostgreSQL + PostGIS, Redis |
| Event Streaming | Apache Kafka (or MQTT for small scale) |
| Frontend | React 18 + TypeScript, Leaflet, Recharts (production) / Vanilla JS + Leaflet (MVP) |
| Auth | JWT + RBAC (ADMIN / INVESTIGATOR / TRAFFIC_OFFICER / VIEWER) |
| Deployment | Docker Compose |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/amanjangrawork-lab/Team-LCQ
cd Team-LCQ

# 2. Install dependencies
uv sync

# 3. Run the MVP dashboard
python run_mvp.py
```

Open http://localhost:5000 — the dashboard is pre-seeded with demo data on first load.

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) GPU with CUDA for YOLOv8 inference
- (Optional) Webcam or IP camera for live detection

---

## Project Structure

```
Team-LCQ/
├── run_mvp.py                  # Main Flask app — single source of truth
├── backend/
│   ├── app.py                  # Legacy shim (re-exports app from run_mvp.py)
│   ├── city_scale.py           # 5,000 virtual camera generator + city stats/heat
│   ├── fleet.py                # 5,000-camera fleet simulator + event generator
│   ├── real_vision.py          # YOLOv8 + RapidOCR/PaddleOCR for webcam/video/streaming
│   ├── models/
│   │   ├── anpr_engine.py      # ANPR pipeline class (YOLO + plate crop + OCR + validation)
│   │   └── trajectory_engine.py # Multi-camera trajectory linking + analytics
│   └── templates/
│       └── index.html          # Dashboard SPA (Leaflet maps, MJPEG streams)
├── scripts/
│   ├── generate_ppt_screenshots.py  # 8 PPT-ready PNG screenshots
│   └── generate_traffic_video.py    # Synthetic traffic MP4 generator
├── src/netrapath/              # Placeholder package
├── project_spec.json           # Project specification
├── pyproject.toml              # uv project config
└── uv.lock                     # Lockfile
```

---

## Operating Modes

| Mode | Description |
|---|---|
| **LIVE** | RTSP camera streams — real-time detection and alerts |
| **REPLAY** | Pre-recorded .mp4 files — full pipeline on recorded footage |
| **SIMULATION** | Generated synthetic events — large-scale testing without camera access |

---

## Key Features

### AI/CV Pipeline
- Two-stage detection: YOLOv8-nano (vehicle) → YOLOv8 (plate) → CLAHE enhancement → perspective correction → sharpening → OCR
- Plate Quality Score: `0.30×sharpness + 0.25×brightness_ok + 0.20×plate_area_ratio + 0.15×angle_ok + 0.10×detector_confidence`
- Temporal OCR fusion: weighted voting across frames — `Score(plate) = Σ quality_i × confidence_i × I(plate_i == plate)`
- Indian plate format regex validation with I/1, O/0 confusion correction
- Low-confidence readings flagged for human review (never silently discarded)

### Trajectory Engine
- Groups detections by plate hash, sorts chronologically
- Route plausibility check: flags links requiring physically impossible speed (> 150 km/h)
- Dijkstra shortest-path on camera network graph (NetworkX)
- Route confidence score per segment; low-confidence segments shown as dashed lines

### Alert Engine (7 Types)
| Alert | Trigger |
|---|---|
| BLACKLIST | Plate hash matches active watchlist entry (Redis cached) |
| GEOFENCE | Detection location within configured PostGIS polygon |
| WRONG_WAY | Vehicle direction conflicts with road segment allowed direction |
| SPEED_VIOLATION | Calculated speed between camera pair exceeds threshold |
| CLONED_PLATE | Same plate at two physically impossible locations simultaneously |
| ROUTE_ANOMALY | Trajectory route confidence below threshold |
| CAMERA_DOWN | No camera heartbeat for > 5 minutes |

Risk Score: `0.40×B + 0.25×C + 0.20×R + 0.15×Q` → CRITICAL (≥0.75), HIGH (0.55–0.74), MEDIUM (0.35–0.54), LOW (<0.35)

### Dashboard Pages
1. Login (JWT + RBAC role-based routing)
2. Home / Overview (stat tiles, mini map, quick links)
3. Plate Search & Trajectory (text input + date range → route line on map → sighting list)
4. Live Alerts (WebSocket real-time alert cards, acknowledge/resolve)
5. Watchlist Management (add/deactivate, role-gated)
6. Traffic Heatmap & Analytics (Leaflet heatmap, OD matrix, peak-hour charts)
7. Camera Health (green/amber/red status grid, 24-h detection history)
8. Advanced Search (area-based polygon, route-based camera pair, camera+time filter)
9. Reports / Review Queue (low-confidence detections, impossible-travel flags, cloned-plate suspects)

---

## Scalability Path

| Stage | Scale | Infrastructure |
|---|---|---|
| Prototype (Hackathon) | 3–5 cameras | Single GPU workstation, Docker Compose, local PostgreSQL |
| Pilot | 50 cameras | Edge Jetson/T4 nodes, Kafka partitioning, PG read replicas, Redis cluster |
| City-Scale | 1,000+ cameras | Kubernetes HPA, TimescaleDB hypertables, PostGIS spatial indexing, CDN frontend |

---

## Evaluation Metrics (Prototype Targets)

| Metric | Target |
|---|---|
| Plate detection precision | ≥ 90 % |
| Full plate recognition accuracy | ≥ 90 % (clear) / ≥ 70 % (adverse) |
| Camera-to-camera linking accuracy | ≥ 85 % |
| Alert generation latency | < 5 s end-to-end |
| Duplicate event suppression | ≥ 80 % |
| Trajectory display time | < 3 s for 30-day window |

---

## Privacy & Compliance

- DPDP Act 2023 + DPDP Rules 2025 aligned
- HMAC-SHA256 plate hashing for all analytics
- AES-256 encrypted plate storage, decrypted only for ADMIN/INVESTIGATOR roles
- Immutable audit logs for every plate view, search, and alert action
- No automatic challan/penalty — human verification required for all enforcement

---

## Known Limitations

- OCR accuracy drops in heavy rain, full darkness, and severely damaged plates (temporal fusion + human review mitigate)
- No real government camera feed tested — replay/simulation modes provide full demo
- Camera clock mismatch can corrupt trajectories (NTP sync + plausibility check mitigate)
- Prototype is not a certified enforcement tool — all decisions require human verification

---

## License

Built for SIH 2026. See repository for details.