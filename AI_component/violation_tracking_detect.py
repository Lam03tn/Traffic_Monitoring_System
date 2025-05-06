import cv2
import imageio
import numpy as np
import json
import time
from collections import defaultdict, deque
import uuid
import io
import base64
from datetime import datetime
import redis
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import minio
import tempfile
import os
from supervision import Detections
from bytetrack.byte_track import ByteTrack


class ViolationHandler:
    """Base class for handling specific violation types"""
    def __init__(self, config, camera_id):
        self.config = config
        self.camera_id = camera_id
        self.roi = self.create_polygon(config.get('roi', {}))
        self.traffic_light_zone = self.create_polygon(config.get('violation_config', [{}])[0].get('traffic_light_zone', {}))
        self.violation_type = config.get('violation_type', 'unknown')
        
        # Initialize lane marking for line-crossing violations
        lane_marking = config.get('violation_config', [{}])[0].get('lane_marking', {})
        if lane_marking:
            self.lane_marking_start = (
                lane_marking.get('start_point', {}).get('x', 0),
                lane_marking.get('start_point', {}).get('y', 0)
            )
            self.lane_marking_end = (
                lane_marking.get('end_point', {}).get('x', 0),
                lane_marking.get('end_point', {}).get('y', 0)
            )
        else:
            self.lane_marking_start = (0, 0)
            self.lane_marking_end = (0, 0)

    def create_polygon(self, points):
        """Create polygon from points dictionary"""
        if not points:
            return np.array([], dtype=np.float32)
        coords = [
            [points.get('point1', {}).get('x', 0), points.get('point1', {}).get('y', 0)],
            [points.get('point2', {}).get('x', 0), points.get('point2', {}).get('y', 0)],
            [points.get('point3', {}).get('x', 0), points.get('point3', {}).get('y', 0)],
            [points.get('point4', {}).get('x', 0), points.get('point4', {}).get('y', 0)]
        ]
        return np.array(coords, dtype=np.float32)

    def is_point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        if polygon.size == 0:
            return False
        point = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def ccw(self, A, B, C):
        """Check counter-clockwise order"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A, B, C, D):
        """Check if line segments AB and CD intersect"""
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def get_direction(self, point1, point2):
        """Determine movement direction"""
        direction_str = ""
        if point1[1] > point2[1]:
            direction_str += "South"
        elif point1[1] < point2[1]:
            direction_str += "North"
        if point1[0] > point2[0]:
            direction_str += "East"
        elif point1[0] < point2[0]:
            direction_str += "West"
        return direction_str

    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        """Base method to detect violation - to be overridden"""
        return None

class TrafficLightViolationHandler(ViolationHandler):
    """Handler for traffic light violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        if traffic_light_state != 'red':
            return []
        
        violations = []
        
        # Iterate through tracked objects
        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            if cls not in [2, 3, 5, 7]:  # Vehicles
                continue

            # Get bounding box
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            track_id = tracked_detections.tracker_id[i]
            
            if track_id in data_deque and len(data_deque[track_id]) >= 2:
                # Get current and previous positions
                curr_pos = data_deque[track_id][0]
                prev_pos = data_deque[track_id][1]
                
                # Check direction and intersection with line
                direction = self.get_direction(curr_pos, prev_pos)
                
                if self.intersect(
                    curr_pos, prev_pos, 
                    self.lane_marking_start, self.lane_marking_end
                ):
                    if "North" in direction:  # Customize this based on your specific violation criteria
                        # Draw bounding box on the frame
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color=(0, 0, 255),  # Red color in BGR
                            thickness=2
                        )
                        # Optionally add track ID as text
                        cv2.putText(
                            frame,
                            f"ID: {track_id}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),  # Red color
                            2
                        )

                        # Append violation with annotated frame
                        violations.append({
                            'track_id': track_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'frame': frame.copy(),  # Include annotated frame
                            'timestamp': time.time()
                        })

        return violations


class LineCrossingViolationHandler(ViolationHandler):
    """Handler for line crossing violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        if traffic_light_state != 'red':
            return []

        violations = []
        
        # Check for lane-crossing violations
        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            if cls not in [2, 3, 5, 7]:  # Not a vehicle
                continue
                
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get track history from data_deque
            if track_id in data_deque and len(data_deque[track_id]) >= 2:
                # Get current and previous positions
                curr_pos = data_deque[track_id][0]
                prev_pos = data_deque[track_id][1]
                
                # Check direction and intersection with line
                direction = self.get_direction(curr_pos, prev_pos)
                
                if self.intersect(
                    curr_pos, prev_pos, 
                    self.lane_marking_start, self.lane_marking_end
                ):
                    if "North" in direction:  # Customize this based on your specific violation criteria
                        violations.append({
                            'track_id': track_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'timestamp': time.time(),
                            'direction': direction
                        })
        
        return violations


class WrongWayViolationHandler(ViolationHandler):
    """Handler for wrong way violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        violations = []

        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            if cls not in [2, 3, 5, 7]:  # Not a vehicle
                continue
                
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if self.is_point_in_polygon((center_x, center_y), self.roi):
                # Check movement direction if track history exists
                if track_id in data_deque and len(data_deque[track_id]) >= 2:
                    curr_pos = data_deque[track_id][0]
                    prev_pos = data_deque[track_id][1]
                    
                    # Calculate direction angle
                    direction = np.arctan2(curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0])
                    
                    # Assuming config specifies allowed direction range in degrees
                    allowed_direction = np.radians(self.config.get('allowed_direction', 0))
                    direction_tolerance = np.radians(self.config.get('direction_tolerance', 45))
                    
                    if abs(direction - allowed_direction) > direction_tolerance:
                        violations.append({
                            'track_id': track_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'timestamp': time.time()
                        })

        return violations


class EnhancedViolationDetector:
    def __init__(self, camera_id, redis_host='localhost', redis_port=6379, 
                 minio_endpoint='localhost:9000', cassandra_host='localhost'):
        self.camera_id = camera_id
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.minio_client = minio.Minio(
            minio_endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        self.cassandra_cluster = Cluster([cassandra_host])
        self.cassandra_session = self.cassandra_cluster.connect('traffic_system')
        
        self.ensure_minio_buckets()
        self.configs = self.load_configs_from_minio()
        
        if not self.configs:
            print(f"No configurations found for camera {camera_id}. Violation detection disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        print(f"Configurations loaded for camera {camera_id}. Violation detection enabled.")
        
        self.detection_channel = f'inferences_{camera_id}'
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.data_deque = defaultdict(lambda: deque(maxlen=64))
        self.frame_buffer = deque(maxlen=120)  # 4 seconds at 30 fps
        self.traffic_light_state = 'unknown'
        self.active_violations = defaultdict(dict)
        self.processed_frames = 0
        
        # Initialize ByteTrack
        self.tracker = ByteTrack(
            track_activation_threshold=0.6,   # High-confidence threshold
            lost_track_buffer=30,            # Keep tracks for 30 frames
            minimum_matching_threshold=0.8,   # IoU threshold for matching tracks
            frame_rate=10                     # Assumed frame rate
        )
        
        self.violation_handlers = self.initialize_violation_handlers()

    def ensure_minio_buckets(self):
        required_buckets = ['violation-configs', 'violation-videos', 'violation-images']
        for bucket in required_buckets:
            try:
                if not self.minio_client.bucket_exists(bucket):
                    self.minio_client.make_bucket(bucket)
                    print(f"Created bucket: {bucket}")
            except Exception as e:
                print(f"Error creating bucket {bucket}: {e}")

    def load_configs_from_minio(self):
        configs = []
        try:
            objects = self.minio_client.list_objects('violation-configs', prefix=f"{self.camera_id}_")
            for obj in objects:
                response = self.minio_client.get_object('violation-configs', obj.object_name)
                config_data = response.read().decode('utf-8')
                config = json.loads(config_data)
                configs.append(config)
        except Exception as e:
            print(f"Error loading configs from MinIO: {e}")
        return configs

    def initialize_violation_handlers(self):
        handlers = {}
        for config in self.configs:
            violation_type = config.get('violation_type')
            if violation_type == 'traffic_light':
                handlers[violation_type] = TrafficLightViolationHandler(config, self.camera_id)
            elif violation_type == 'line_crossing':
                handlers[violation_type] = LineCrossingViolationHandler(config, self.camera_id)
            elif violation_type == 'wrong_way':
                handlers[violation_type] = WrongWayViolationHandler(config, self.camera_id)
        return handlers

    def detect_traffic_light_color(self, frame):
        if not self.enabled:
            return 'unknown'
            
        traffic_light_handler = self.violation_handlers.get('traffic_light')
        if not traffic_light_handler:
            return 'unknown'
            
        traffic_light_poly = traffic_light_handler.traffic_light_zone
        if traffic_light_poly.size == 0:
            return 'unknown'
            
        x, y, w, h = cv2.boundingRect(traffic_light_poly.astype(np.int32))
        x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        traffic_light_region = frame[y:y+h, x:x+w]
        if traffic_light_region.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(traffic_light_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 150, 70])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 150, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        red_count = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count = cv2.countNonZero(mask_green)
        
        max_count = max(red_count, yellow_count, green_count)
        if max_count < 50:
            return 'unknown'
            
        if max_count == red_count:
            return 'red'
        elif max_count == yellow_count:
            return 'yellow'
        else:
            return 'green'

    def convert_to_supervision_detections(self, detection_data):
        """Convert detection data to Supervision Detections format for ByteTrack"""

        boxes = []
        scores = []
        class_ids = []
        
        for det in detection_data.get('detections', []):
            x, y, w, h = det.get('box', [0, 0, 0, 0])
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            boxes.append([x1, y1, x2, y2])
            scores.append(det.get('score', 0.0))
            class_ids.append(det.get('class_id', 0))
            
        if not boxes:
            return None
                    
        return Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            class_id=np.array(class_ids)
        )

    def process_detection(self, detection_data):
        if not self.enabled or not detection_data:
            return
            
        try:
            detection = json.loads(detection_data)
            frame_index = detection['frame_index']
            metadata = detection['metadata']
            frame_base64 = detection.get('frame_data')
            self.processed_frames += 1
                
            if frame_base64:
                frame_data = base64.b64decode(frame_base64)
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"Failed to decode frame {frame_index}")
                    return
            else:
                print(f"No frame data for frame {frame_index}")
                return
                
            timestamp = metadata.get('timestamp', time.time())
            self.frame_buffer.append((frame, frame_index, timestamp))
            
            # Detect traffic light state
            self.traffic_light_state = self.detect_traffic_light_color(frame)
            
            # Convert detections to Supervision Detections format
            detections = self.convert_to_supervision_detections(detection)
            if detections is None:
                print(f"No detections in frame {frame_index}")
                return True
                
            # Update tracker with new detections
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Update tracking history for line crossing detection
            for i in range(len(tracked_detections.tracker_id)):
                track_id = tracked_detections.tracker_id[i]
                box = tracked_detections.xyxy[i]
                
                # Calculate center point
                center_x = (box[0] + box[2]) / 2
                center_y = (box[3] + box[3]) / 2
                
                # Store point for path tracking
                self.data_deque[track_id].appendleft((center_x, center_y))
                
                # Store detailed position history
                self.track_history[track_id].append((
                    center_x, center_y, frame_index, timestamp
                ))
            
            # Process violations for each handler
            for violation_type, handler in self.violation_handlers.items():
                violations = handler.detect(frame, tracked_detections, self.traffic_light_state, self.data_deque)
                if violations:
                    for violation in violations:
                        track_id = violation['track_id']
                        if track_id not in self.active_violations[violation_type]:
                            violation_id = str(uuid.uuid4())
                            self.active_violations[violation_type][track_id] = {
                                'violation_id': violation_id,
                                'violation_type': violation_type,
                                'start_frame': frame_index,
                                'start_time': timestamp,
                                'end_frame': frame_index,
                                'end_time': timestamp,
                                'license_plate': '',
                                'frames': [],
                                'track_id': track_id,
                                'processed': False
                            }
                            print(f"New {violation_type} violation detected: ID {violation_id}, Track {track_id}")
                        
                        violation_data = self.active_violations[violation_type][track_id]
                        violation_data['end_frame'] = frame_index
                        violation_data['end_time'] = timestamp
                        violation_data['frames'].append((frame.copy(), frame_index, timestamp))
            
            # Process completed violations
            self.process_completed_violations(frame_index, timestamp)
            return True
            
        except Exception as e:
            print(f"Error processing detection: {e}")
            import traceback
            traceback.print_exc()
            return True

    def process_completed_violations(self, current_frame, current_time):
        completed_violations = []
        for violation_type, violations in list(self.active_violations.items()):
            for track_id, violation in list(violations.items()):
                if current_frame - violation['end_frame'] > 15:
                    violation['processed'] = True
                    # Collect frames for 4-second video (±2 seconds around violation)
                    violation_time = violation['start_time']
                    frames_for_video = []
                    for frame, frame_index, timestamp in self.frame_buffer:
                        if abs(timestamp - violation_time) <= 4.0:  # Within ±2 seconds
                            frames_for_video.append((frame.copy(), frame_index, timestamp))
                    violation['frames_for_video'] = sorted(frames_for_video, key=lambda x: x[2])  # Sort by timestamp
                    completed_violations.append((violation_type, track_id, violation))
        
        for violation_type, track_id, violation in completed_violations:
            try:
                self.save_violation_evidence(violation)
                del self.active_violations[violation_type][track_id]
                if not self.active_violations[violation_type]:
                    del self.active_violations[violation_type]
            except Exception as e:
                print(f"Error processing violation {violation['violation_id']}: {e}")

    def save_violation_evidence(self, violation):
        violation_id = violation['violation_id']
        violation_time = violation['start_time']
        violation_type = violation['violation_type']
        timestamp = datetime.fromtimestamp(violation_time).strftime('%Y%m%d_%H%M%S')

        try:
            frames_for_video = violation.get('frames_for_video', [])
            if len(frames_for_video) > 0:
                first_frame = frames_for_video[0][0]
                height, width = first_frame.shape[:2]
                fps = 3  # Assuming 30 fps for 4-second video

                buffer = io.BytesIO()
    
                # Write video to buffer using imageio
                with imageio.get_writer(buffer, format='mp4', fps=fps) as writer:
                    for frame_data, _, _ in frames_for_video:
                        writer.append_data(frame_data)
                
                # Upload to MinIO
                buffer.seek(0)
    
                video_path = f"{violation_type}/{self.camera_id}/{timestamp}.mp4"
                self.minio_client.put_object(
                    'violation-videos',
                    video_path,
                    data=buffer,
                    length=buffer.getbuffer().nbytes,
                    content_type='video/mp4'
                )

                # Save image from middle frame
                middle_idx = len(frames_for_video) // 2
                violation_image = frames_for_video[middle_idx][0]
                _, buffer = cv2.imencode('.jpg', violation_image)
                image_buffer = io.BytesIO(buffer.tobytes())
                
                image_path = f"{violation_type}/{self.camera_id}/{timestamp}.jpg"
                self.minio_client.put_object(
                    'violation-images',
                    image_path,
                    data=image_buffer,
                    length=len(buffer),
                    content_type='image/jpeg'
                )

                # Save metadata to Cassandra
                violation_date = datetime.fromtimestamp(violation_time).strftime('%Y-%m-%d')
                query = """
                INSERT INTO violations 
                (violation_type, violation_date, violation_time, violation_id, license_plate, 
                camera_id, processed_time, status, video_evidence_url, image_evidence_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cassandra_session.execute(query, (
                    violation_type,
                    violation_date,
                    datetime.fromtimestamp(violation_time),
                    uuid.UUID(violation_id),
                    violation.get('license_plate', ''),
                    self.camera_id,
                    datetime.now(),
                    'pending', 
                    f"minio://violation-videos/{video_path}",
                    f"minio://violation-images/{image_path}"
                ))

                print(f"Saved violation {violation_id} evidence to MinIO and Cassandra")

        except Exception as e:
            print(f"Error saving violation evidence: {e}")

    def start_processing(self):
        if not self.enabled:
            print("Violation detection is disabled. No configuration found.")
            return
            
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.detection_channel)
        
        print(f"Started violation detection for camera {self.camera_id}")
        print(f"Listening on Redis channel: {self.detection_channel}")
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    result = self.process_detection(message['data'])
                    if result is False:
                        break
        except KeyboardInterrupt:
            print("Violation detection interrupted by user")
        except Exception as e:
            print(f"Error in violation detection: {e}")
        finally:
            pubsub.unsubscribe()
            print("Violation detection stopped")

    def close(self):
        try:
            self.cassandra_cluster.shutdown()
        except:
            pass
        print("Connections closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced traffic violation detection")
    parser.add_argument("--camera", type=str, default="cam4", help="Camera ID to monitor")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--minio-endpoint", type=str, default="localhost:9000", help="MinIO endpoint")
    parser.add_argument("--cassandra-host", type=str, default="localhost", help="Cassandra host")
    
    args = parser.parse_args()
    
    detector = EnhancedViolationDetector(
        camera_id=args.camera,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        minio_endpoint=args.minio_endpoint,
        cassandra_host=args.cassandra_host
    )
    
    try:
        detector.start_processing()
    except KeyboardInterrupt:
        print("Violation detection interrupted")
    finally:
        detector.close()


if __name__ == "__main__":
    main()