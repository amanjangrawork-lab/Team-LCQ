from collections import defaultdict


class TrajectoryEngine:
    """Reconstructs vehicle paths across multiple cameras."""
    
    def __init__(self, max_travel_time_minutes=60, max_speed_kmh=180):
        self.max_travel_time = max_travel_time_minutes * 60
        self.max_speed = max_speed_kmh
        self.trajectories = []
        self.camera_graph = {}
    
    def set_camera_positions(self, cameras):
        self.camera_graph = cameras
    
    def link_detections(self, detections):
        plate_groups = defaultdict(list)
        for d in detections:
            plate_norm = self._normalize_plate(d.get('plate', d.get('plate_text', '')))
            if plate_norm:
                plate_groups[plate_norm].append(d)
        
        trajectories = []
        for plate, events in plate_groups.items():
            events.sort(key=lambda x: x.get('timestamp', ''))
            
            if len(events) < 2:
                continue
            
            # Deduplicate per camera (keep best confidence)
            unique_events = {}
            for e in events:
                cid = e.get('camera_id', '')
                if cid not in unique_events or e.get('ocr_confidence', 0) > unique_events[cid].get('ocr_confidence', 0):
                    unique_events[cid] = e
            
            unique_list = list(unique_events.values())
            if len(unique_list) < 2:
                continue
            
            route_feasible = self._check_route_feasibility(unique_list)
            
            avg_confidence = sum(e.get('ocr_confidence', 0) for e in unique_list) / len(unique_list)
            
            for i in range(len(unique_list) - 1):
                t1 = self._parse_timestamp(unique_list[i].get('timestamp', ''))
                t2 = self._parse_timestamp(unique_list[i+1].get('timestamp', ''))
                if t1 and t2:
                    travel_time = (t2 - t1).total_seconds()
                    unique_list[i]['travel_time_to_next'] = travel_time
            
            trajectory = {
                'trajectory_id': f"TRAJ_{plate}_{unique_list[0].get('camera_id', '')}",
                'plate': plate,
                'start_time': unique_list[0].get('timestamp', ''),
                'end_time': unique_list[-1].get('timestamp', ''),
                'camera_sequence': [e.get('camera_id', '') for e in unique_list],
                'event_count': len(unique_list),
                'average_confidence': round(avg_confidence, 4),
                'route_feasible': route_feasible,
                'route_confidence': round(avg_confidence * (1.0 if route_feasible else 0.5), 4),
                'status': 'complete' if route_feasible else 'questionable',
                'events': unique_list
            }
            trajectories.append(trajectory)
        
        self.trajectories = trajectories
        return trajectories
    
    def _normalize_plate(self, plate_text):
        if not plate_text:
            return None
        cleaned = ''.join(c.upper() for c in str(plate_text) if c.isalnum())
        cleaned = cleaned.replace('O', '0').replace('I', '1')
        return cleaned if len(cleaned) >= 6 else None
    
    def _parse_timestamp(self, ts_str):
        try:
            if 'T' in ts_str:
                from datetime import datetime as dt
                return dt.fromisoformat(ts_str.replace('Z', '+00:00'))
            parts = ts_str.split(':')
            from datetime import datetime as dt
            return dt(2026, 1, 1, int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            return None
    
    def _check_route_feasibility(self, events):
        if len(events) < 2:
            return True
        for i in range(len(events) - 1):
            e1 = events[i]
            e2 = events[i + 1]
            cam1 = self.camera_graph.get(e1.get('camera_id', ''), {})
            cam2 = self.camera_graph.get(e2.get('camera_id', ''), {})
            lat1, lon1 = cam1.get('latitude', 0), cam1.get('longitude', 0)
            lat2, lon2 = cam2.get('latitude', 0), cam2.get('longitude', 0)
            dist = self._haversine(lat1, lon1, lat2, lon2)
            t1 = self._parse_timestamp(e1.get('timestamp', ''))
            t2 = self._parse_timestamp(e2.get('timestamp', ''))
            if t1 and t2:
                travel_time_sec = (t2 - t1).total_seconds()
                if travel_time_sec > 0:
                    speed_kmh = dist / (travel_time_sec / 3600)
                    if speed_kmh > self.max_speed:
                        return False
        return True
    
    def compute_traffic_analytics(self, detections):
        if not detections:
            return {}
        
        camera_counts = defaultdict(int)
        vehicle_types = defaultdict(int)
        hour_counts = defaultdict(int)
        
        for d in detections:
            camera_counts[d.get('camera_id', 'unknown')] += 1
            vehicle_types[d.get('vehicle_type', 'unknown')] += 1
            hour = d.get('timestamp', '')[:2]
            hour_counts[hour] += 1
        
        od_pairs = defaultdict(int)
        for traj in self.trajectories:
            if len(traj.get('camera_sequence', [])) >= 2:
                od = (traj['camera_sequence'][0], traj['camera_sequence'][-1])
                od_pairs[od] += 1
        
        speeds = []
        for traj in self.trajectories:
            events = traj.get('events', [])
            for i in range(len(events) - 1):
                if 'travel_time_to_next' in events[i]:
                    cam1 = self.camera_graph.get(events[i].get('camera_id', ''), {})
                    cam2 = self.camera_graph.get(events[i+1].get('camera_id', ''), {})
                    lat1, lon1 = cam1.get('latitude', 0), cam1.get('longitude', 0)
                    lat2, lon2 = cam2.get('latitude', 0), cam2.get('longitude', 0)
                    dist = self._haversine(lat1, lon1, lat2, lon2)
                    travel_time = events[i]['travel_time_to_next']
                    if travel_time > 0:
                        speeds.append(dist / (travel_time / 3600))
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        return {
            'camera_counts': dict(camera_counts),
            'vehicle_types': dict(vehicle_types),
            'hour_distribution': dict(hour_counts),
            'origin_destination': {f"{k[0]}→{k[1]}": v for k, v in od_pairs.items()},
            'average_speed_kmh': round(avg_speed, 1),
            'total_vehicles': len(detections),
            'total_traj': len(self.trajectories)
        }
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt
        R = 6371
        lat1r, lat2r = radians(lat1), radians(lat2)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))
