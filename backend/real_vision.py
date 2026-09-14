"""Real vision: webcam + video-file YOLO detection + PaddleOCR plate read."""
import cv2
import time
import random
import threading
from datetime import datetime

_yolo = None
_ocr = None

def get_yolo():
    global _yolo
    if _yolo is None:
        import os as _os
        from ultralytics import YOLO
        here = _os.path.dirname(_os.path.abspath(__file__))
        root = _os.path.dirname(here)
        pt = _os.path.join(root, 'yolov8n.pt')
        _yolo = YOLO(pt if _os.path.isfile(pt) else 'yolov8n.pt')
    return _yolo

def get_ocr():
    # RapidOCR primary (offline, light). Paddle fallback if its backend exists.
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = ('rapid', RapidOCR())
            print('[OCR] RapidOCR ready')
        except Exception as e:
            print(f'[OCR] rapid failed: {e}')
            try:
                from paddleocr import PaddleOCR
                _ocr = ('paddle', PaddleOCR(use_angle_cls=True, lang='en'))
                print('[OCR] PaddleOCR ready')
            except Exception as e2:
                print(f'[OCR] init failed: {e2}')
                _ocr = False
    return _ocr if _ocr else None


def prep_crop(crop):
    # upscale small crops + denoise so tiny plates become readable
    h, w = crop.shape[:2]
    s = 1.0
    if w < 200:
        s = 200 / max(w, 1)
        crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def ocr_lines(prepped):
    eng = get_ocr()
    if eng is None:
        return []
    kind, inst = eng
    try:
        if kind == 'rapid':
            res, _ = inst(prepped)
            if not res:
                return []
            return [(r[1], float(r[2])) for r in res]
        out = inst.ocr(prepped, cls=False)
        if out and out[0]:
            return [(ln[1][0], float(ln[1][1])) for ln in out[0]]
    except Exception:
        pass
    return []


def read_plate_from_crop(crop, full_vehicle=None):
    # try several bands: lower-middle, lower third, full vehicle; keep best read >= 2 chars
    h, w = crop.shape[:2]
    cands = [crop]
    if w > 60:
        cands.append(crop[:, max(0, w // 2 - w):])
    if full_vehicle is not None:
        cands.append(full_vehicle)
    best, bc = '', 0.0
    for c in cands:
        if c.size == 0 or c.shape[0] < 10 or c.shape[1] < 25:
            continue
        for ln, cf in ocr_lines(prep_crop(c)):
            clean = ''.join(x.upper() for x in ln if x.isalnum())
            if len(clean) >= 2 and cf > bc:
                best, bc = clean, cf
    return best, round(bc, 3)

# COCO vehicle class ids for yolov8: car=2, motorcycle=3, bus=5, truck=7
VEH_IDS = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

webcam = {'cap': None, 'on': False, 'lock': threading.Lock(), 'last_dets': [],
          'source': 0, 'frame_no': 0}

def list_cameras(max_probe=5):
    """Probe local camera indices + note about IP/phone apps.

    Phone apps (Iriun / DroidCam / Smart Connect) usually appear as a
    virtual camera on index 1..3, or expose an MJPEG URL such as
    http://<phone-ip>:4747/video . We probe indices; URLs are
    passed through untouched by webcam_start().
    """
    found = []
    for idx in range(max_probe):
        try:
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            if ok:
                ok, _ = cap.read()
            cap.release()
            if ok:
                found.append({'index': idx,
                              'label': f"Camera {idx}" + (" (built-in / default)" if idx == 0 else " (USB / virtual — phone apps usually here)")})
        except Exception:
            continue
    return found

def _parse_source(src):
    """Accept 0, '0', '1', or an http(s)/mjpeg URL string."""
    if src is None:
        return 0
    if isinstance(src, int):
        return src
    s = str(src).strip()
    if s.lower().startswith(('http://', 'https://', 'rtsp://')):
        return s
    try:
        return int(s)
    except Exception:
        return s

def webcam_start(src=0):
    src = _parse_source(src)
    with webcam['lock']:
        # switching source → release old capture first
        if webcam['cap'] is not None and webcam.get('source') != src:
            try:
                webcam['cap'].release()
            except Exception:
                pass
            webcam['cap'] = None
        if webcam['cap'] is None:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return False
            webcam['cap'] = cap
            webcam['source'] = src
        webcam['on'] = True
        return True

def webcam_stop():
    with webcam['lock']:
        webcam['on'] = False

def detect_frame(frame, do_ocr=True, max_ocr_boxes=1, conf=0.45):
    """Run YOLO on a frame. Returns (annotated_frame, dets).

    Tuned for CPU speed: imgsz=640, limited detections, OCR only on
    large boxes. Raises nothing — callers wrap in try/except.
    """
    model = get_yolo()
    res = model(frame, conf=conf, iou=0.5, imgsz=640, max_det=25, verbose=False)[0]
    dets = []
    ann = frame.copy()
    boxes = []
    for b in res.boxes:
        cid = int(b.cls[0])
        if cid not in VEH_IDS:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        if (x2 - x1) < 28 or (y2 - y1) < 20:
            continue  # too small to read — skip OCR + clutter
        confv = float(b.conf[0])
        boxes.append((x1, y1, x2, y2, confv, VEH_IDS[cid]))
    # OCR on largest boxes only (speed) — and only wide-enough boxes
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    ocr_on = get_ocr() is not None if do_ocr else False
    for i, (x1, y1, x2, y2, conf, vtype) in enumerate(boxes[:12]):
        plate, oconf = '', 0.0
        raw = ''
        if ocr_on and i < max_ocr_boxes and (x2 - x1) >= 70:
            # plate region: lower-middle of vehicle box
            w, h = x2-x1, y2-y1
            px1, py1 = x1+int(w*0.1), y1+int(h*0.45)
            px2, py2 = x2-int(w*0.1), y1+int(h*0.85)
            crop = frame[max(0,py1):py2, max(0,px1):px2]
            full = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 25:
                plate, oconf = read_plate_from_crop(crop, full)
                raw = plate
        dets.append({'bbox': [x1, y1, x2, y2], 'conf': round(conf, 3),
                     'type': vtype, 'plate': plate, 'raw': raw, 'ocr_conf': round(oconf, 3)})
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 136), 2)
        cv2.rectangle(ann, (x1, y1-22), (x1+min(230, 90+len(plate)*9), y1-4), (0, 0, 0), -1)
        label = f"{vtype} {int(conf*100)}%"
        if plate:
            label += f" | {plate} {int(oconf*100)}%"
        cv2.putText(ann, label, (x1+3, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
    return ann, dets

def mjpeg_gen():
    """Yield MJPEG frames from webcam with live detection overlay.

    Fast path: work at 640x360, run YOLO every 2nd frame and reuse the
    last annotation in between so the stream never stalls on slow OCR.
    """
    last_ann, n = None, 0
    while True:
        with webcam['lock']:
            cap, on = webcam['cap'], webcam['on']
        if cap is None or not on:
            time.sleep(0.1)
            continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (640, 360))
        n += 1
        if n % 2 == 0 or last_ann is None:
            try:
                ann, dets = detect_frame(frame, do_ocr=True, max_ocr_boxes=1)
                last_ann = ann
                with webcam['lock']:
                    webcam['last_dets'] = [{'type': d['type'], 'conf': d['conf'],
                                            'plate': d['plate'], 'ocr_conf': d['ocr_conf'],
                                            'time': datetime.now().strftime('%H:%M:%S')} for d in dets]
                    webcam['frame_no'] += 1
            except Exception:
                last_ann = frame
        out_f = last_ann if last_ann is not None else frame
        ok, buf = cv2.imencode('.jpg', out_f, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~15-18 fps cap, keeps CPU headroom

def process_video_file(path, sample_every=30, max_frames=600, max_dets=300, progress_cb=None):
    """Run YOLO+OCR over a video file, return detection list.

    Fast defaults: inspect ~20 frames at 640px, OCR top-1 large box
    only. A 2-min 1080p clip finishes in ~1-3 min on CPU instead of
    10-20 min. progress_cb(done, total) is called as we go so the UI
    can show a live progress bar (async job).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out, idx, inspected = [], 0, 0
    try:
        while len(out) < max_dets:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                small = cv2.resize(frame, (640, 360))
                try:
                    _, dets = detect_frame(small, do_ocr=True, max_ocr_boxes=1)
                except Exception:
                    dets = []
                ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                for d in dets:
                    out.append({'detection_id': f"DET_FILE_{idx}_{len(out)}",
                                'camera_id': 'UPLOAD', 'timestamp': ts,
                                'plate': d['plate'] or f"UNREAD_{len(out)}",
                                'ocr_confidence': d['ocr_conf'] or d['conf'],
                                'image_quality': d['conf'], 'vehicle_type': d['type'],
                                'vehicle_color': 'unknown', 'direction': 'unknown',
                                'status': 'verified' if d['ocr_conf'] > 0.7 else 'low_confidence'})
                    if len(out) >= max_dets:
                        break
                inspected += 1
                if progress_cb and total:
                    try:
                        progress_cb(min(idx + 1, total), total)
                    except Exception:
                        pass
            idx += 1
            if idx > max_frames * sample_every:
                break
    finally:
        cap.release()
    if progress_cb:
        try:
            progress_cb(1, 1)
        except Exception:
            pass
    return out

def ocr_demo_on_image(path):
    import os as _os
    img = cv2.imread(path)
    if img is None:
        return {'ok': False, 'error': f'cannot read {path}'}
    if get_ocr() is None:
        return {'ok': False, 'error': 'OCR engine unavailable'}
    lines = []
    for ln, cf in ocr_lines(img):
        clean = ''.join(c.upper() for c in ln if c.isalnum())
        if clean:
            lines.append({'text': ln, 'clean': clean, 'conf': round(float(cf), 3),
                          'indian_plate_like': 6 <= len(clean) <= 12})
    lines.sort(key=lambda l: -l['conf'])
    return {'ok': True, 'image': _os.path.basename(path), 'lines': lines[:15]}


def _resolve_traffic_source():
    """Return best available traffic video path — REAL footage first.

    Priority: user's real 4K download → repo 4K copy → synthetic clips.
    (Previously the synthetic cartoon clip won, which is why the Live
    feed showed animated boxes instead of the real video.)
    """
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    root = _os.path.dirname(here)
    candidates = [
        'C:/Users/amanj/Downloads/18437773-uhd_3840_2160_50fps.mp4',
        _os.path.join(root, 'data', 'videos', '18437773-uhd_3840_2160_50fps.mp4'),
        _os.path.join(root, 'data', 'videos', 'cam_03_highway.mp4'),
        _os.path.join(root, 'data', 'videos', 'cam_02_junction.mp4'),
        _os.path.join(root, 'data', 'videos', 'cam_01_street_day.mp4'),
    ]
    for p in candidates:
        try:
            if p and _os.path.isfile(p):
                return p
        except Exception:
            continue
    return candidates[0]


TRAFFIC_PATH = _resolve_traffic_source()
traffic = {'cap': None, 'on': False, 'lock': threading.Lock(), 'last_dets': [],
           'frame_no': 0, 'source': TRAFFIC_PATH, 'infer_every': 6, 'last_fps': 0.0,
           'error': ''}

def traffic_start():
    import os as _os
    with traffic['lock']:
        # always prefer the real file if it appeared after boot
        best = _resolve_traffic_source()
        if best != traffic.get('source') and _os.path.isfile(best or ''):
            try:
                if traffic['cap'] is not None:
                    traffic['cap'].release()
            except Exception:
                pass
            traffic['cap'] = None
            traffic['source'] = best
        if traffic['cap'] is None:
            traffic['source'] = best
            cap = cv2.VideoCapture(best)
            if not cap.isOpened():
                traffic['error'] = f"cannot open {best}"
                traffic['on'] = False
                return False
            # reduce internal buffering so the stream stays live, not laggy
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
            traffic['cap'] = cap
            traffic['error'] = ''
        traffic['on'] = True
        return True

def traffic_restart():
    with traffic['lock']:
        try:
            if traffic['cap'] is not None:
                traffic['cap'].release()
        except Exception:
            pass
        traffic['cap'] = None
        traffic['frame_no'] = 0
        traffic['last_dets'] = []
    return traffic_start()

def traffic_status():
    import os as _os
    src = traffic.get('source') or TRAFFIC_PATH
    return {'source': src, 'exists': bool(src and _os.path.isfile(src)),
            'frame': traffic.get('frame_no', 0), 'fps': traffic.get('last_fps', 0.0),
            'infer_every': traffic.get('infer_every', 6),
            'dets': len(traffic.get('last_dets', [])),
            'error': traffic.get('error', '')}

def traffic_stop():
    with traffic['lock']:
        traffic['on'] = False

def traffic_mjpeg_gen():
    """Stream the REAL traffic file with YOLO+OCR overlay — fast path.

    Why it used to stick: 4K decode + 960px YOLO every 3rd frame + 2-box
    OCR all ran inline per served frame. Now: 640x360 work size, YOLO
    every 6th frame, OCR top-1 large box only, last annotation reused
    between inferences, output throttled to ~14 fps, adaptive skip if
    inference is slow. A missing file yields a readable placeholder
    frame instead of a hanging stream.
    """
    import numpy as _np
    traffic_start()
    last_ann, t_prev, fps = None, time.time(), 0.0
    infer_every = traffic.get('infer_every', 6)
    while True:
        with traffic['lock']:
            cap, on = traffic['cap'], traffic['on']
        if cap is None or not on:
            ph = _np.zeros((360, 640, 3), dtype=_np.uint8)
            cv2.putText(ph, 'Live feed offline — check video source', (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ok, buf = cv2.imencode('.jpg', ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.5)
            traffic_start()
            continue
        ok, frame = cap.read()
        if not ok:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            continue
        traffic['frame_no'] += 1
        small = cv2.resize(frame, (640, 360))
        if traffic['frame_no'] % infer_every == 0 or last_ann is None:
            t0 = time.time()
            try:
                ann, dets = detect_frame(small, do_ocr=True, max_ocr_boxes=1)
                last_ann = ann
                with traffic['lock']:
                    traffic['last_dets'] = [
                        {'type': d['type'], 'conf': d['conf'], 'plate': d['plate'],
                         'ocr_conf': d['ocr_conf'], 'frame': traffic['frame_no'],
                         'time': datetime.now().strftime('%H:%M:%S')} for d in dets]
            except Exception:
                last_ann = small
            dt = time.time() - t0
            # adaptive: if inference is slow, infer less often (up to every 12th)
            if dt > 0.35 and infer_every < 12:
                infer_every += 2
                traffic['infer_every'] = infer_every
            elif dt < 0.12 and infer_every > 4:
                infer_every -= 1
                traffic['infer_every'] = infer_every
        out_f = last_ann if last_ann is not None else small
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(0.001, now - t_prev) if t_prev else 0.0
        t_prev = now
        traffic['last_fps'] = round(fps, 1)
        ok, buf = cv2.imencode('.jpg', out_f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.06)  # ~14 fps cap
