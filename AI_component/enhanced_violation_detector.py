import uuid
import cv2
import imageio
import numpy as np
import json
import time
from collections import defaultdict, deque
import io
from datetime import datetime
import minio
from supervision import Detections
from cassandra.cluster import Cluster
from bytetrack.byte_track import ByteTrack

class EnhancedViolationDetector:
    def __init__(self, camera_id, minio_endpoint='localhost:9000', cassandra_host='localhost'):
        self.camera_id = camera_id
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
        
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.data_deque = defaultdict(lambda: deque(maxlen=64))
        self.frame_buffer = deque(maxlen=60)  # 4 seconds at 30 fps
        self.traffic_light_state = 'unknown'
        self.active_violations = defaultdict(dict)
        
        # Initialize ByteTrack
        self.tracker = ByteTrack(
            track_activation_threshold=0.6,   # High-confidence threshold
            lost_track_buffer=40,            # Keep tracks for 30 frames
            minimum_matching_threshold=0.8,   # IoU threshold for matching tracks
            frame_rate=3                     # Assumed frame rate
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
                from violation_tracking_detect import TrafficLightViolationHandler
                handlers[violation_type] = TrafficLightViolationHandler(config, self.camera_id)
            elif violation_type == 'line_crossing':
                from violation_tracking_detect import LineCrossingViolationHandler
                handlers[violation_type] = LineCrossingViolationHandler(config, self.camera_id)
            elif violation_type == 'wrong_way':
                from violation_tracking_detect import WrongWayViolationHandler
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
        
        if(h < 640):
            scale = 640 / h
            traffic_light_region = cv2.resize(traffic_light_region, (int(w * scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

        hsv = cv2.cvtColor(traffic_light_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 150, 70])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 150, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        red_count = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count = cv2.countNonZero(mask_green)
        
        max_count = max(red_count, yellow_count, green_count)
        if max_count < 20:
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

    def process_detection(self, detection_data, frame):
        if not self.enabled or not detection_data:
            return
            
        try:
            frame_index = detection_data['frame_index']
            metadata = detection_data['metadata']
            timestamp = time.time()
            self.frame_buffer.append((frame, frame_index, timestamp))
            
            # Detect traffic light state
            self.traffic_light_state = self.detect_traffic_light_color(frame)

            # Convert detections to Supervision Detections format
            detections = self.convert_to_supervision_detections(detection_data)
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
                        violation_id = violation['violation_id']
                        self.active_violations[violation_type][violation_id] = {
                            'violation_id': violation_id,
                            'violation_type': violation_type,
                            'timestamp': timestamp,
                            'frame_index': frame_index,
                            'license_plate': violation['license_plate'],
                            'frame_violation': violation['frame'],
                            'processed': False
                        }
                        print(f"New {violation_type} violation detected: ID {violation_id}, Track {track_id}")
                        
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
        frame_list = list(self.frame_buffer)  # Chuyển sang list để dễ truy cập theo chỉ số\

        for violation_type, violations in list(self.active_violations.items()):
            for violation_id, violation in list(violations.items()):
                if current_time - violation['timestamp'] > 2:
                    violation['processed'] = True
                    # Collect frames for 2-second video (±1 second around violation)
                    violation_time = violation['timestamp']
                    frames_for_video = []
                    # Tìm index của frame chính
                    main_index = next(
                        (i for i, (_, _, ts) in enumerate(frame_list) if ts == violation_time),
                        None
                    )

                    if main_index is not None:
                        # Lấy 6 frame trước và sau (nếu có đủ)
                        start_index = max(0, main_index - 6)
                        end_index = min(len(frame_list), main_index + 7)  # +7 để bao gồm frame chính + 6 sau

                        frames_for_video = []
                        for i in range(start_index, end_index):
                            frame, frame_index, timestamp = frame_list[i]
                            # Frame chính giữ nguyên object, còn lại copy
                            if i == main_index:
                                frames_for_video.append((violation['frame_violation'], frame_index, timestamp))
                            else:
                                frames_for_video.append((frame.copy(), frame_index, timestamp))

                        # Sắp xếp (nếu cần thiết, nhưng thường không cần nếu deque đã sắp theo thời gian)
                        violation['frames_for_video'] = sorted(frames_for_video, key=lambda x: x[2])
                        completed_violations.append((violation_type, violation_id, violation))
                            
        for violation_type, violation_id, violation in completed_violations:
            try:
                self.save_violation_evidence(violation)
                del self.active_violations[violation_type][violation_id]
                if not self.active_violations[violation_type]:
                    del self.active_violations[violation_type]
            except Exception as e:
                print(f"Error processing violation {violation['violation_id']}: {e}")

    def save_violation_evidence(self, violation):
        violation_id = violation['violation_id']
        violation_time = violation['timestamp']
        violation_type = violation['violation_type']
        timestamp = datetime.fromtimestamp(violation_time).strftime('%Y%m%d_%H%M%S')

        try:
            frames_for_video = violation.get('frames_for_video', [])
            if len(frames_for_video) > 0:
                first_frame = frames_for_video[0][0]
                height, width = first_frame.shape[:2]
                fps = 5  # Assuming 3 fame detect per second

                buffer = io.BytesIO()
    
                # Write video to buffer using imageio
                with imageio.get_writer(buffer, format='mp4', fps=fps) as writer:
                    for frame_data, _, _ in frames_for_video:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame_rgb)
                
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

                # Save image from violation frames (with bounding box)
                violation_image = violation.get('frame_violation', [])
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
                    f"http://localhost:9000://violation-videos/{video_path}",
                    f"http://localhost:9000://violation-images/{image_path}"          
                ))

                print(f"Saved violation {violation_id} evidence to MinIO and Cassandra")

        except Exception as e:
            print(f"Error saving violation evidence: {e}")

    def close(self):
        try:
            self.cassandra_cluster.shutdown()
        except:
            pass
        print("Connections closed")