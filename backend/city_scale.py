"""City-scale: 5000 virtual cameras across zones with edge hierarchy."""
import random
import math

ZONES = [
    ('North Gate', 19.10, 72.86, 900),
    ('Central', 19.08, 72.88, 1400),
    ('Market', 19.085, 72.89, 800),
    ('Eastern Hwy', 19.09, 72.91, 900),
    ('Terminal', 19.07, 72.87, 1000),
]

ROADS = ['North Entry Road', 'Main Junction', 'Market Corridor', 'Eastern Highway', 'Terminal Road',
         'Ring Road', 'Airport Link', 'Port Corridor', 'Metro Line Rd', 'Coastal Rd']

_cams = None

def gen_cameras(n=5000, seed=7):
    global _cams
    if _cams is not None and len(_cams) == n:
        return _cams
    rnd = random.Random(seed)
    cams = []
    # 5 real pilot cameras first
    base = [(19.0760, 72.8777, 'City Gate North'), (19.0795, 72.8802, 'Central Junction'),
            (19.0820, 72.8850, 'Market Road'), (19.0870, 72.8910, 'Highway Exit'),
            (19.0740, 72.8720, 'Bus Terminal')]
    for i, (la, lo, nm) in enumerate(base):
        cams.append({'camera_id': f"CAM_{i+1:04d}", 'camera_name': nm,
                     'latitude': la, 'longitude': lo, 'road_segment': rnd.choice(ROADS),
                     'direction': rnd.choice(['North', 'South', 'East', 'West']),
                     'zone': 'Pilot', 'edge': 'EDGE-PILOT-1', 'status': 'ONLINE',
                     'fps': 24, 'density': round(rnd.uniform(0.5, 0.95), 2)})
    per = (n - 5) // len(ZONES)
    idx = 6
    for zname, zla, zlo, _count in ZONES:
        edge = f"EDGE-{zname.upper().replace(' ', '-')}"
        for _ in range(per):
            la = zla + rnd.gauss(0, 0.018)
            lo = zlo + rnd.gauss(0, 0.022)
            st = 'ONLINE' if rnd.random() > 0.04 else ('DEGRADED' if rnd.random() > 0.4 else 'OFFLINE')
            cams.append({'camera_id': f"CAM_{idx:04d}", 'camera_name': f"{zname} {idx}",
                         'latitude': round(la, 5), 'longitude': round(lo, 5),
                         'road_segment': rnd.choice(ROADS),
                         'direction': rnd.choice(['North', 'South', 'East', 'West']),
                         'zone': zname, 'edge': edge, 'status': st,
                         'fps': rnd.choice([15, 20, 24, 30]),
                         'density': round(rnd.uniform(0.05, 0.98), 2)})
            idx += 1
    while len(cams) < n:
        cams.append({'camera_id': f"CAM_{idx:04d}", 'camera_name': f"Overflow {idx}",
                     'latitude': round(19.08 + rnd.gauss(0, 0.03), 5),
                     'longitude': round(72.88 + rnd.gauss(0, 0.03), 5),
                     'road_segment': rnd.choice(ROADS), 'direction': 'North',
                     'zone': 'Overflow', 'edge': 'EDGE-OVERFLOW', 'status': 'ONLINE',
                     'fps': 20, 'density': 0.4})
        idx += 1
    _cams = cams[:n]
    return _cams

def stats(n=5000):
    cams = gen_cameras(n)
    on = sum(1 for c in cams if c['status'] == 'ONLINE')
    # simulated city throughput
    ev_sec = on * 0.6  # ~0.6 events/sec/camera
    raw_gbps = on * 8 / 1000  # 8 Mbps raw each
    edge_gbps = on * 0.02 / 1000 * 1000  # ~20 kbps metadata each -> in Mbps
    return {
        'total': len(cams), 'online': on,
        'degraded': sum(1 for c in cams if c['status'] == 'DEGRADED'),
        'offline': sum(1 for c in cams if c['status'] == 'OFFLINE'),
        'edge_nodes': len(set(c['edge'] for c in cams)),
        'events_per_sec': round(ev_sec, 1),
        'events_per_day_m': round(ev_sec * 86400 / 1e6, 1),
        'raw_bandwidth_gbps': round(raw_gbps, 1),
        'edge_bandwidth_gbps': round(on * 0.02 / 1000, 2),
        'bandwidth_saved_pct': 99.7,
        'storage_per_day_tb': round(ev_sec * 86400 * 2 / 1e6, 2),  # ~2KB/event
        'gpu_workers': math.ceil(on / 60),
        'p95_alert_latency_s': 3.8,
        'zones': [{'zone': z[0], 'count': sum(1 for c in cams if c['zone'] == z[0])} for z in ZONES] + [{'zone': 'Pilot', 'count': 5}],
    }

def heat_points(n=5000, limit=1500):
    cams = gen_cameras(n)
    # top-density subset for heat circles
    top = sorted(cams, key=lambda c: c['density'], reverse=True)[:limit]
    return [{'lat': c['latitude'], 'lon': c['longitude'], 'd': c['density']} for c in top]
