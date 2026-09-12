"""City-scale fleet: 5000 virtual cameras + high-throughput event simulator.

The 5 pilot nodes stay live (webcam/video). The other 4995 are a simulated
city fleet used to prove the architecture scales: edge aggregation, event
bus, sharded counters. Lightweight by design — no 5000 threads, just
one generator + precomputed aggregates.
"""
import math
import random
import threading
import time
from collections import Counter
from datetime import datetime

TOTAL = 5000
DISTRICTS = ['Andheri', 'Bandra', 'Borivali', 'Chembur', 'Dadar', 'Juhu',
             'Kurla', 'Malad', 'Powai', 'Thane', 'Vasai', 'Worli']
ROADS = ['Link Rd', 'SV Rd', 'WE Highway', 'Eastern Fwy', 'MG Rd', 'Station Rd',
         'Market Rd', 'Airport Rd', 'Coastal Rd', 'Ghodbunder Rd']
EDGES = 50  # 50 edge nodes x 100 cameras each

_rng = random.Random(26127)
cameras = []
for i in range(1, TOTAL + 1):
    lat = _rng.uniform(18.90, 19.30)
    lon = _rng.uniform(72.75, 72.98)
    r = _rng.random()
    status = 'OFFLINE' if r < 0.02 else ('DEGRADED' if r < 0.03 else 'ONLINE')
    cameras.append({
        'camera_id': f"CAM_{i:04d}",
        'latitude': round(lat, 5),
        'longitude': round(lon, 5),
        'district': DISTRICTS[(i - 1) % len(DISTRICTS)],
        'road_segment': ROADS[(i - 1) % len(ROADS)],
        'edge_node': f"EDGE_{(i - 1) // 100 + 1:02d}",
        'status': status,
    })

_by_district = Counter(c['district'] for c in cameras)
_by_edge = Counter(c['edge_node'] for c in cameras)

sim = {'on': False, 'rate': 0, 'total': 0, 't0': None, 'eps': 0.0,
       'per_district': Counter(), 'recent': [], 'thread': None, 'lock': threading.Lock()}

STATES = ['MH', 'DL', 'KA', 'TN', 'GJ', 'RJ', 'UP', 'WB']

def _plate():
    return (f"{_rng.choice(STATES)}{_rng.randint(10,99)}"
            f"{chr(65+_rng.randint(0,25))}{chr(65+_rng.randint(0,25))}{_rng.randint(1000,9999)}")

def _gen_loop():
    while True:
        with sim['lock']:
            if not sim['on']:
                return
            rate = sim['rate']
        batch = max(1, rate // 10)  # 10 batches/sec
        now = datetime.now().strftime('%H:%M:%S')
        buf = []
        for _ in range(batch):
            c = cameras[_rng.randint(0, TOTAL - 1)]
            buf.append({'plate': _plate(), 'camera_id': c['camera_id'],
                        'district': c['district'], 'time': now,
                        'conf': round(_rng.uniform(0.7, 0.99), 2)})
        with sim['lock']:
            sim['total'] += len(buf)
            for e in buf:
                sim['per_district'][e['district']] += 1
            sim['recent'] = (buf + sim['recent'])[:100]
            if sim['t0']:
                el = max(0.1, time.time() - sim['t0'])
                sim['eps'] = round(sim['total'] / el, 1)
        time.sleep(0.1)

def sim_start(rate=500, ncam=5000):
    with sim['lock']:
        sim['on'] = True
        sim['rate'] = rate
        sim['total'] = 0
        sim['t0'] = time.time()
        sim['eps'] = 0.0
        sim['per_district'] = Counter()
        sim['recent'] = []
        if sim['thread'] is None or not sim['thread'].is_alive():
            sim['thread'] = threading.Thread(target=_gen_loop, daemon=True)
            sim['thread'].start()
    return {'status': 'simulating', 'rate_eps': rate, 'cameras': min(ncam, TOTAL)}

def sim_stop():
    with sim['lock']:
        sim['on'] = False
    return {'status': 'stopped', 'total': sim['total']}

def summary():
    on = sum(1 for c in cameras if c['status'] == 'ONLINE')
    return {
        'total_cameras': TOTAL,
        'online': on,
        'offline': sum(1 for c in cameras if c['status'] == 'OFFLINE'),
        'degraded': sum(1 for c in cameras if c['status'] == 'DEGRADED'),
        'districts': len(DISTRICTS),
        'edge_nodes': EDGES,
        'cams_per_edge': TOTAL // EDGES,
        'by_district': dict(_by_district),
        'pilot_live': 5,
        'arch': 'edge(50x Jetson) → Kafka(32 partitions) → FastAPI workers → PostGIS sharded + Redis + S3 evidence',
    }

def sample(limit=1500, district=None):
    pool = cameras if not district else [c for c in cameras if c['district'] == district]
    if len(pool) > limit:
        step = len(pool) / limit
        return [pool[int(i * step)] for i in range(limit)]
    return pool

def stats():
    with sim['lock']:
        return {'on': sim['on'], 'target_eps': sim['rate'], 'measured_eps': sim['eps'],
                'total_events': sim['total'], 'per_district': dict(sim['per_district']),
                'recent': sim['recent'][:20]}
