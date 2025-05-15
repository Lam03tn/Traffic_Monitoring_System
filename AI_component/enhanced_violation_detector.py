import uuid
import cv2
import imageio
from matplotlib import pyplot as plt
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
    def __init__(self, camera_id, minio_endpoint='localhost:9000', cassandra_host='localhost', config_check_interval=60):
        self.camera_id = camera_id
        self.minio_client = minio.Minio(
            minio_endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        self.cassandra_cluster = Cluster([cassandra_host])
        self.cassandra_session = self.cassandra_cluster.connect('traffic_system')
        
        self.config_check_interval = config_check_interval  # Interval in seconds to check for config updates
        self.last_config_check = 0  # Timestamp of last config check
        self.config_etags = {}  # Store ETags or last_modified timestamps for configs
        
        self.ensure_minio_buckets()
        self.configs = self.load_configs_from_minio()
        
        self.enabled = bool(self.configs)  # Enable only if configs are present
        if not self.enabled:
            print(f"No configurations found for camera {camera_id}. Violation detection disabled.")
        else:
            print(f"Configurations loaded for camera {camera_id}. Violation detection enabled.")
        
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.data_deque = defaultdict(lambda: deque(maxlen=64))
        self.frame_buffer = deque(maxlen=60)  # 4 seconds at 30 fps
        self.traffic_light_state = 'unknown'
        self.active_violations = defaultdict(dict)
        
        # Initialize ByteTrack
        self.tracker = ByteTrack(
            track_activation_threshold=0.6,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=3
        )
        
        self.violation_handlers = self.initialize_violation_handlers()
        
        # Extract ROIs from all violation handlers for filtering detections
        self.all_rois = self.extract_all_rois()

    def extract_all_rois(self):
        """Extract all ROIs from violation handlers for filtering detections"""
        all_rois = []
        if not self.enabled:
            return all_rois
        
        for _, handler in self.violation_handlers.items():
            if handler.roi.size > 0:  # Check if ROI exists and is not empty
                all_rois.append(handler.roi)
        return all_rois

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
        new_etags = {}
        try:
            objects = self.minio_client.list_objects('violation-configs', prefix=f"{self.camera_id}_", recursive=True)
            for obj in objects:
                response = self.minio_client.get_object('violation-configs', obj.object_name)
                config_data = response.read().decode('utf-8')
                config = json.loads(config_data)
                configs.append(config)
                # Store ETag or last_modified to detect changes
                new_etags[obj.object_name] = obj.last_modified or obj.etag
                response.close()
                response.release_conn()
        except Exception as e:
            print(f"Error loading configs from MinIO: {e}")
        self.config_etags = new_etags  # Update stored ETags
        return configs

    def check_and_reload_configs(self, current_time):
        """Check for config updates and reload if necessary."""
        if current_time - self.last_config_check < self.config_check_interval:
            return False  # Not time to check yet

        self.last_config_check = current_time
        try:
            # List objects and compare ETags or last_modified timestamps
            new_etags = {}
            objects = self.minio_client.list_objects('violation-configs', prefix=f"{self.camera_id}_", recursive=True)
            configs_changed = False
            temp_configs = []

            for obj in objects:
                new_etags[obj.object_name] = obj.last_modified or obj.etag
                if obj.object_name not in self.config_etags or self.config_etags[obj.object_name] != new_etags[obj.object_name]:
                    configs_changed = True
                response = self.minio_client.get_object('violation-configs', obj.object_name)
                config_data = response.read().decode('utf-8')
                config = json.loads(config_data)
                temp_configs.append(config)
                response.close()
                response.release_conn()

            # Check for deleted configs or new configs when previously empty
            if set(self.config_etags.keys()) != set(new_etags.keys()) or (not self.enabled and temp_configs):
                configs_changed = True

            if configs_changed:
                was_disabled = not self.enabled
                self.configs = temp_configs
                self.config_etags = new_etags
                self.enabled = bool(self.configs)
                self.violation_handlers = self.initialize_violation_handlers()
                # Update ROIs after reloading configs
                self.all_rois = self.extract_all_rois()
                
                if was_disabled and self.enabled:
                    print(f"Configurations detected for camera {self.camera_id}. Enabling violation detection.")
                elif self.enabled:
                    print(f"Config changes detected for camera {self.camera_id}. Reloaded {len(self.configs)} configurations.")
                else:
                    print(f"No configurations found after reload for camera {self.camera_id}. Violation detection disabled.")
                return True
            return False
        except Exception as e:
            print(f"Error checking configs: {e}")
            return False

    def initialize_violation_handlers(self):
        handlers = {}
        if not self.enabled:
            return handlers  # Return empty handlers if disabled
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
        
        if h < 640:
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
    
    def is_inside_any_roi(self, point):
        """Check if a point is inside any of the ROIs"""
        if not self.all_rois:  # If no ROIs defined, accept all points
            return True
            
        for roi in self.all_rois:
            if cv2.pointPolygonTest(roi.astype(np.int32), point, False) >= 0:
                return True
        return False

    def process_detection(self, detection_data, frame):
        try:
            current_time = time.time()
            # Always check configs, even if disabled, to detect new configs
            self.check_and_reload_configs(current_time)
            
            if not self.enabled or not detection_data:
                return True  # Continue processing to allow future config checks
                
            frame_index = detection_data['frame_index']
            metadata = detection_data['metadata']
            timestamp = current_time
            self.frame_buffer.append((frame, frame_index, timestamp))
            
            # Detect traffic light state
            self.traffic_light_state = self.detect_traffic_light_color(frame)

            # Convert detections to Supervision Detections format and filter by ROI
            detections = self.convert_to_supervision_detections(detection_data)
            if detections is None:
                print(f"No detections in ROI for frame {frame_index}")
                return True
                
            # Update tracker with new detections (only those in ROI)
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Update tracking history for line crossing detection
            for i in range(len(tracked_detections.tracker_id)):
                track_id = tracked_detections.tracker_id[i]
                box = tracked_detections.xyxy[i]
                
                # Calculate center point
                center_x = (box[0] + box[2]) / 2
                center_y = (box[3] + box[3]) / 2  # Fixed: Use box[1] and box[3] for y-coordinates

                if not self.is_inside_any_roi((center_x, center_y)):
                    continue  # Skip detections outside all ROIs
                
                # Store point for path tracking
                self.data_deque[track_id].appendleft((center_x, center_y))
                
            # Process violations for each handler
            for violation_type, handler in self.violation_handlers.items():
                violations = handler.detect(
                    frame, 
                    tracked_detections, 
                    self.traffic_light_state, 
                    self.data_deque, 
                    frame_buffer=self.frame_buffer
                )
                if violations:
                    for violation in violations:
                        violation_id = violation.get('violation_id', str(uuid.uuid4()))
                        if 'violation_id' not in violation:
                            violation['violation_id'] = violation_id
                            
                        self.active_violations[violation_type][violation_id] = {
                            'violation_id': violation_id,
                            'violation_type': violation_type,
                            'timestamp': timestamp,
                            'frame_index': frame_index,
                            'license_plate': violation.get('license_plate', ''),
                            'frame_violation': violation.get('frame', [frame.copy(), frame.copy(), None]),
                            'processed': False
                        }
                        track_id = violation.get('track_id', 'unknown')
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
        frame_list = list(self.frame_buffer)

        for violation_type, violations in list(self.active_violations.items()):
            for violation_id, violation in list(violations.items()):
                if current_time - violation['timestamp'] > 2:
                    violation['processed'] = True
                    violation_time = violation['timestamp']
                    frames_for_video = []
                    main_index = next(
                        (i for i, (_, _, ts) in enumerate(frame_list) if ts == violation_time),
                        None
                    )

                    if main_index is not None:
                        start_index = max(0, main_index - 9)
                        end_index = min(len(frame_list), main_index + 10)
                        for i in range(start_index, end_index):
                            frame, frame_index, timestamp = frame_list[i]
                            if i == main_index:
                                frames_for_video.append((violation['frame_violation'][0], frame_index, timestamp))
                            elif i == main_index - 1:
                                frames_for_video.append((violation['frame_violation'][1], frame_index, timestamp))
                            else:
                                frames_for_video.append((frame.copy(), frame_index, timestamp))
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
                fps = 5

                buffer = io.BytesIO()
                with imageio.get_writer(buffer, format='mp4', fps=fps) as writer:
                    for frame_data, _, _ in frames_for_video:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame_rgb)
                
                buffer.seek(0)
                video_path = f"{violation_type}/{self.camera_id}/{timestamp}.mp4"
                self.minio_client.put_object(
                    'violation-videos',
                    video_path,
                    data=buffer,
                    length=buffer.getbuffer().nbytes,
                    content_type='video/mp4'
                )

                # Save frame_violation_after as _1.jpg
                frame_violation_after = violation.get('frame_violation', [])[0]
                _, buffer_after = cv2.imencode('.jpg', frame_violation_after)
                image_buffer_after = io.BytesIO(buffer_after.tobytes())
                image_path_after = f"{violation_type}/{self.camera_id}/{timestamp}_1.jpg"
                self.minio_client.put_object(
                    'violation-images',
                    image_path_after,
                    data=image_buffer_after,
                    length=len(buffer_after),
                    content_type='image/jpeg'
                )

                # Save frame_violation_before as _2.jpg
                frame_violation_before = violation.get('frame_violation', [])[1]
                _, buffer_before = cv2.imencode('.jpg', frame_violation_before)
                image_buffer_before = io.BytesIO(buffer_before.tobytes())
                image_path_before = f"{violation_type}/{self.camera_id}/{timestamp}_2.jpg"
                self.minio_client.put_object(
                    'violation-images',
                    image_path_before,
                    data=image_buffer_before,
                    length=len(buffer_before),
                    content_type='image/jpeg'
                )

                # Save plate_img as _3.jpg if available
                plate_imgs = violation.get('frame_violation', [])[2]
                if plate_imgs is not None:
                    _, buffer_plate = cv2.imencode('.jpg', plate_imgs)
                    plate_image_buffer = io.BytesIO(buffer_plate.tobytes())
                    plate_image_path = f"{violation_type}/{self.camera_id}/{timestamp}_3.jpg"
                    self.minio_client.put_object(
                        'violation-images',
                        plate_image_path,
                        data=plate_image_buffer,
                        length=len(buffer_plate),
                        content_type='image/jpeg'
                    )

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
                    f"http://localhost:9000/violation-videos/{video_path}",
                    f"http://localhost:9000/violation-images/{image_path_after}"
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