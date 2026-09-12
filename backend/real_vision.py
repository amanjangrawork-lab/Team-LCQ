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
        from ultralytics import YOLO
        _yolo = YOLO('yolov8n.pt')
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

webcam = {'cap': None, 'on': False, 'lock': threading.Lock(), 'last_dets': []}

def webcam_start(src=0):
    with webcam['lock']:
        if webcam['cap'] is None:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return False
            webcam['cap'] = cap
        webcam['on'] = True
        return True

def webcam_stop():
    with webcam['lock']:
        webcam['on'] = False

def detect_frame(frame, do_ocr=True, max_ocr_boxes=3):
    """Run YOLO on a frame. Returns (annotated_frame, dets)."""
    model = get_yolo()
    res = model(frame, conf=0.35, verbose=False)[0]
    dets = []
    ann = frame.copy()
    boxes = []
    for b in res.boxes:
        cid = int(b.cls[0])
        if cid not in VEH_IDS:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        boxes.append((x1, y1, x2, y2, conf, VEH_IDS[cid]))
    # OCR on largest boxes only (speed)
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    ocr_on = get_ocr() is not None if do_ocr else False
    for i, (x1, y1, x2, y2, conf, vtype) in enumerate(boxes):
        plate, oconf = '', 0.0
        raw = ''
        if ocr_on and i < max_ocr_boxes:
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
    """Yield MJPEG frames from webcam with live detection overlay."""
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
        frame = cv2.resize(frame, (960, 540))
        try:
            ann, dets = detect_frame(frame)
            with webcam['lock']:
                webcam['last_dets'] = [{'type': d['type'], 'conf': d['conf'],
                                        'plate': d['plate'], 'ocr_conf': d['ocr_conf'],
                                        'time': datetime.now().strftime('%H:%M:%S')} for d in dets]
        except Exception:
            ann = frame
        ok, buf = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

def process_video_file(path, sample_every=10, max_frames=400):
    """Run YOLO+OCR over a video file, return detection list."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    out, idx = [], 0
    while len(out) < 400:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            small = cv2.resize(frame, (960, 540))
            _, dets = detect_frame(small)
            ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            for d in dets:
                out.append({'detection_id': f"DET_FILE_{idx}_{len(out)}",
                            'camera_id': 'UPLOAD', 'timestamp': ts,
                            'plate': d['plate'] or f"UNREAD_{len(out)}",
                            'ocr_confidence': d['ocr_conf'] or d['conf'],
                            'image_quality': d['conf'], 'vehicle_type': d['type'],
                            'vehicle_color': 'unknown', 'direction': 'unknown',
                            'status': 'verified' if d['ocr_conf'] > 0.7 else 'low_confidence'})
        idx += 1
        if idx > max_frames * sample_every:
            break
    cap.release()
    return out

def ocr_demo_on_image(path):
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
    return {'ok': True, 'image': path.split('/')[-1], 'lines': lines[:15]}


TRAFFIC_PATH = 'C:/Users/amanj/Downloads/18437773-uhd_3840_2160_50fps.mp4'
traffic = {'cap': None, 'on': False, 'lock': threading.Lock(), 'last_dets': [], 'frame_no': 0}

def traffic_start():
    with traffic['lock']:
        if traffic['cap'] is None:
            cap = cv2.VideoCapture(TRAFFIC_PATH)
            if not cap.isOpened():
                return False
            traffic['cap'] = cap
        traffic['on'] = True
        return True

def traffic_stop():
    with traffic['lock']:
        traffic['on'] = False

def traffic_mjpeg_gen():
    """Loop the real 4K traffic file, YOLO+OCR every 3rd frame, stream MJPEG."""
    traffic_start()
    last_ann = None
    while True:
        with traffic['lock']:
            cap, on = traffic['cap'], traffic['on']
        if cap is None or not on:
            time.sleep(0.1)
            continue
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        traffic['frame_no'] += 1
        small = cv2.resize(frame, (960, 540))
        if traffic['frame_no'] % 3 == 0:
            try:
                ann, dets = detect_frame(small, do_ocr=True, max_ocr_boxes=2)
                last_ann = ann
                with traffic['lock']:
                    traffic['last_dets'] = [
                        {'type': d['type'], 'conf': d['conf'], 'plate': d['plate'],
                         'ocr_conf': d['ocr_conf'], 'frame': traffic['frame_no'],
                         'time': datetime.now().strftime('%H:%M:%S')} for d in dets]
            except Exception:
                last_ann = small
        out_f = last_ann if last_ann is not None else small
        ok, buf = cv2.imencode('.jpg', out_f, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
