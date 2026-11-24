# NetraPath — City-Wide AI Engine for Multi-Camera ANPR, Trajectory Tracking and Urban Traffic Analytics

**SIH 2026 — Problem Statement 26127 | Bharat Electronics Limited (BEL)**

> **Status**: MVP Prototype (Flask + Vanilla JS) | **Target**: SIH 2026 Final Round

NetraPath transforms a city's existing CCTV/ANPR camera network from isolated detection points into a unified, privacy-aware urban mobility intelligence system. It links every plate detection across cameras into a continuous vehicle timeline, reconstructs the actual route on a GIS map, and converts aggregated detections into macro traffic analytics — all on a single web dashboard in real time.

---

## What Is Actually Built (MVP)

The current repository is a **working Flask MVP** that runs with `python run_mvp.py` on port 5000. No Docker, no PostgreSQL, no Kafka, no React — just a single-file Flask app with an embedded vanilla JS dashboard. This is intentional for a 36-hour hackathon.

### Implemented Features ✅
- **ANPR Pipeline**: YOLOv8 vehicle + plate detection → PaddleOCR/RapidOCR → Indian plate regex validation
- **Temporal OCR Fusion**: Weighted voting across frames for higher accuracy
- **Multi-Camera Trajectory**: Links plate sightings across cameras with haversine distance + speed plausibility check
- **Watchlist & Blacklist Alerts**: Real-time alert when watchlisted plate is detected
- **Traffic Heatmap**: Density overlay on Leaflet map
- **Camera Health Monitoring**: Green/amber/red status grid
- **City-Scale Simulation**: 5,000 virtual cameras with stats and heat data
- **Fleet Simulator**: 5,000-camera event generator
- **MJPEG Live Streams**: Traffic feed + webcam (including phone-as-webcam apps)
- **Async Video Upload**: Upload .mp4 → process → view detections with progress tracking
- **OCR Lab**: Upload image → get plate text + confidence
- **Analytics**: Vehicle counts, speed estimation, flow analysis, OD matrix
- **28 REST API Endpoints**: All documented below

### Tech Stack (Actual MVP)
| Component | Technology |
|---|---|
| Backend | Python + Flask |
| AI/CV | YOLOv8 (Ultralytics) + PaddleOCR/RapidOCR + OpenCV |
| Frontend | Vanilla JS + Leaflet.js (single-page, 700 lines) |
| Data Store | In-memory Python dicts |
| Maps | Leaflet 1.9 + OpenStreetMap/CARTO tiles |
| Video | OpenCV + MJPEG streaming |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/amanjangrawork-lab/Team-LCQ
cd Team-LCQ

# 2. Install dependencies
uv sync

# 3. Run
python run_mvp.py
```

Open http://localhost:5000 — dashboard is pre-seeded with demo data on first load.

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) GPU with CUDA for YOLOv8
- (Optional) Webcam for live detection

---

## Project Structure

```
Team-LCQ/
├── run_mvp.py                  # Main Flask app — single source of truth (551 lines)
├── backend/
│   ├── app.py                  # Legacy shim (re-exports app from run_mvp.py)
│   ├── city_scale.py           # 5,000 virtual camera generator + city stats/heat
│   ├── fleet.py                # 5,000-camera fleet simulator + event generator
│   ├── real_vision.py          # YOLOv8 + RapidOCR/PaddleOCR for webcam/video/streaming
│   ├── models/
│   │   ├── anpr_engine.py      # ANPR pipeline: YOLO + plate crop + OCR + validation + temporal fusion
│   │   └── trajectory_engine.py # Multi-camera trajectory linking + analytics
│   └── templates/
│       └── index.html          # Dashboard SPA (Leaflet maps, MJPEG streams, 700 lines)
├── scripts/
│   ├── generate_ppt_screenshots.py  # 8 PPT-ready PNG screenshots
│   └── generate_traffic_video.py    # Synthetic traffic MP4 generator
├── docs/                       # SIH documentation (v2.0 — Sprint Edition)
│   ├── 01_Implementation_Plan.docx
│   ├── 02_Research_and_Scenarios.docx
│   └── 03_Project_Overview.docx
├── src/netrapath/              # Placeholder package
├── project_spec.json           # Project specification
├── pyproject.toml              # uv project config
└── uv.lock                     # Lockfile
```

---

## API Endpoints (28 routes)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard (index.html) |
| GET | `/api/detections` | List all detections |
| GET | `/api/trajectories` | List all trajectories |
| GET | `/api/cameras` | List cameras with status |
| GET/POST | `/api/watchlist` | View/add watchlist entries |
| GET | `/api/alerts` | List alerts |
| GET | `/api/analytics` | Traffic analytics |
| GET | `/api/heatmap` | Heatmap data |
| POST | `/api/start-pipeline` | Start detection pipeline |
| GET | `/api/city/stats` | City-wide statistics |
| GET | `/api/city/cameras` | City camera list |
| GET | `/api/city/heat` | City heat data |
| GET | `/traffic_feed` | MJPEG traffic video stream |
| GET | `/api/traffic/dets` | Traffic detection results |
| GET | `/api/traffic/status` | Traffic pipeline status |
| POST | `/api/traffic/restart` | Restart traffic pipeline |
| GET | `/video_feed` | MJPEG webcam stream |
| GET | `/api/webcam/list` | List available webcams |
| POST | `/api/webcam/start` | Start webcam capture |
| POST | `/api/webcam/stop` | Stop webcam capture |
| GET | `/api/webcam/dets` | Webcam detection results |
| POST | `/api/upload-video` | Upload video for async processing |
| GET | `/api/upload-video/status/<job_id>` | Upload processing status |
| POST | `/api/ocr-demo` | OCR on uploaded image |
| GET | `/api/dataset/overview` | Dataset overview |
| GET | `/outputs/<path:fname>` | Serve output files |
| GET | `/api/health` | Health check |

---

## SIH Sprint Plan (36-Hour Hackathon)

| Hours | Task Block | Deliverable |
|---|---|---|
| 0-2 | Setup | Flask running, dashboard loads |
| 2-6 | YOLO detection | Vehicles + plates detected |
| 6-10 | OCR + validation | Plate text extracted |
| 10-14 | Trajectory engine | Cross-camera route on map |
| 14-18 | Dashboard core | Map + search + alerts |
| 18-22 | Heatmap + analytics | Density overlay + OD counts |
| 22-26 | Watchlist + review + demo data | CRUD + seeded data |
| 26-30 | Polish | Error handling, loading states |
| 30-34 | Demo rehearsal | Full run-through x3 |
| 34-36 | Buffer | Backup video, final commit |

---

## The "Wow Moment" (Killer Demo)

> Type a plate number → watch the vehicle's entire journey draw itself as an animated line across 3-4 cameras on a Leaflet map, with timestamps and evidence thumbnails at each stop, then a blacklist alert fires in real-time.

This demonstrates: ANPR ✓, Trajectory ✓, GIS ✓, Real-time Alerts ✓, Evidence ✓ — 5 problem-statement requirements in 30 seconds.

---

## Must-Have / Nice-to-Have / Don't-Build

### Must-Have (Demo-Critical)
1. Working ANPR on 2-3 video clips
2. Multi-camera trajectory on a map (THE WOW)
3. Blacklist alert firing live
4. Heatmap updating during replay
5. Camera health status
6. Low-confidence review queue

### Nice-to-Have (If Time Permits)
- OD matrix chart
- Cloned plate detection demo
- RBAC login screen (role toggle)
- Wrong-way detection

### Don't-Build for SIH (Over-Engineering)
- Kafka → use in-memory queue
- PostgreSQL/PostGIS → use in-memory dicts
- React → use vanilla JS
- FastAPI → use Flask
- Docker → `python run_mvp.py`
- JWT/RBAC → role toggle button
- WebSocket → REST polling
- ByteTrack → single-frame detection
- CLAHE → basic crop + OCR

---

## Privacy & Compliance

- DPDP Act 2023 + DPDP Rules 2025 aligned
- HMAC-SHA256 plate hashing for all analytics
- AES-256 encrypted plate storage (production)
- RBAC: VIEWER / TRAFFIC_OFFICER / INVESTIGATOR / ADMIN
- Immutable audit logs (production)
- No automatic challan — human verification required

---

## Production Roadmap (Post-SIH)

The `docs/` folder contains the full production architecture plan:
- **FastAPI** backend with 15+ endpoints + WebSocket
- **React 18 + TypeScript** 9-page dashboard
- **PostgreSQL + PostGIS** for spatial queries
- **Apache Kafka** for event streaming
- **Redis** for watchlist cache
- **Docker Compose** for one-command deployment
- **RBAC with JWT** authentication
- **ByteTrack** multi-object tracking
- **CLAHE** image enhancement

See `docs/01_Implementation_Plan.docx` for the full 24-day production plan.

---

## Known Limitations (Honest)

- OCR accuracy drops in heavy rain, darkness, damaged plates (temporal fusion mitigates)
- No real government camera feed tested — replay/simulation modes provide full demo
- Camera clock mismatch can corrupt trajectories (plausibility check mitigates)
- Prototype is not a certified enforcement tool — all decisions require human verification
- In-memory data store (no persistence across restarts)

---

## Documentation

| File | Description |
|---|---|
| `docs/01_Implementation_Plan.docx` | Full implementation plan + 36-hour SIH sprint |
| `docs/02_Research_and_Scenarios.docx` | Research review + 10 selected scenarios + SIH judging alignment |
| `docs/03_Project_Overview.docx` | Project overview + demo script + team roles + competitive differentiation |

---

## License

Built for SIH 2026. See repository for details.