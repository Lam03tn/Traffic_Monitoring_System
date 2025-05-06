import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import json
import time
from collections import defaultdict, deque
import os

class RedLightViolationDetector:
    def __init__(self, config_path, model_path, video_source, output_path=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize YOLOv8 model
        self.model = YOLO(model_path, task='detect')
        
        # Video source
        self.cap = cv2.VideoCapture(video_source)
        
        # Get original video dimensions
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.model_input_size = (640, 640)
        
        # Scaling factors
        self.scale_x = self.model_input_size[1] / self.original_width
        self.scale_y = self.model_input_size[0] / self.original_height
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0
        
        # Video output setup
        self.output_path = output_path
        if output_path:
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                     (self.original_width, self.original_height))
        
        # Extract configuration parameters
        self.roi = self.config['violation_config'][0]['roi']
        self.traffic_light_zone = self.config['violation_config'][0]['traffic_light_zone']
        self.lane_marking = self.config['violation_config'][0]['lane_marking']
        
        # Traffic light state
        self.traffic_light_state = 'unknown'
        
        # Tracking history
        self.track_history = defaultdict(lambda: [])
        self.data_deque = defaultdict(lambda: deque(maxlen=64))
        
        # Lane marking for line-crossing detection
        start_point = self.scale_coordinates(self.lane_marking['start_point']['x'], 
                                           self.lane_marking['start_point']['y'])
        end_point = self.scale_coordinates(self.lane_marking['end_point']['x'], 
                                         self.lane_marking['end_point']['y'])
        self.line_scaled = (start_point, end_point)
        
        # Violation counters and folder
        self.object_counter = defaultdict(int)
        self.num_faults = 0
        self.folder_path = 'fault_vehicles'
        os.makedirs(self.folder_path, exist_ok=True)

    def scale_coordinates(self, x, y):
        """Scale coordinates from original to model input size"""
        return x * self.scale_x, y * self.scale_y

    def unscale_coordinates(self, x, y):
        """Unscale coordinates from model input to original size"""
        return x / self.scale_x, y / self.scale_y

    def create_polygon(self, points, scale=True):
        """Create polygon from points dictionary"""
        coords = [
            [points['point1']['x'], points['point1']['y']],
            [points['point2']['x'], points['point2']['y']],
            [points['point3']['x'], points['point3']['y']],
            [points['point4']['x'], points['point4']['y']]
        ]
        if scale:
            coords = [self.scale_coordinates(x, y) for x, y in coords]
        return np.array(coords, dtype=np.float32)

    def is_point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        point = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def detect_traffic_light_colors(self, frame, traffic_light_poly):
        """Detect traffic light color in the specified polygon region"""
        x, y, w, h = cv2.boundingRect(traffic_light_poly.astype(np.int32))
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
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
        if max_count < 30:
            return 'unknown'
        if max_count == red_count:
            return 'red'
        elif max_count == yellow_count:
            return 'yellow'
        else:
            return 'green'

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

    def compute_color_for_labels(self, label):
        """Generate color for labels"""
        palette = np.array([[255, 128, 0], [0, 255, 128], [128, 0, 255], 
                           [255, 255, 0], [0, 128, 255]])
        return tuple(int(x) for x in palette[label % len(palette)])

    def UI_box(self, box, img, label="", color=(0, 255, 0), line_thickness=2):
        """Draw bounding box and label"""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        if label:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - t_size[1] - 5), 
                         (x1 + t_size[0] + 5, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)

    def draw_boxes(self, img, bbox, names, object_id, identities=None, offset=(0, 0), mode='off'):
        """Draw bounding boxes and handle violation visualization"""
        height, width, _ = img.shape
        img_copy = img.copy()
        
        # Draw lane marking on original frame
        start_point = (int(self.lane_marking['start_point']['x']), 
                      int(self.lane_marking['start_point']['y']))
        end_point = (int(self.lane_marking['end_point']['x']), 
                    int(self.lane_marking['end_point']['y']))
        lane_color = (0, 0, 255) if mode == 'on' else (0, 255, 0)  # Red when active, green when inactive
        cv2.line(img, start_point, end_point, lane_color, 3)

        # Clean up data_deque
        if identities is not None:
            for key in list(self.data_deque):
                if key not in identities:
                    self.data_deque.pop(key)

        violations = []
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            # Convert back to original coordinates for display
            x1, y1 = self.unscale_coordinates(x1, y1)
            x2, y2 = self.unscale_coordinates(x2, y2)
            center = (int((x1 + x2) / 2), int((y2 + y2) / 2))
            id = int(identities[i]) if identities is not None else 0

            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=64)
            # Store scaled coordinates for intersection check
            scaled_center = self.scale_coordinates(center[0], center[1])
            self.data_deque[id].appendleft(scaled_center)

            color = self.compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]
            label = f"{id}:{obj_name}"

            # Check for lane-crossing violation
            if mode == 'on' and len(self.data_deque[id]) >= 2:
                direction = self.get_direction(self.data_deque[id][0], self.data_deque[id][1])
                if self.intersect(self.data_deque[id][0], self.data_deque[id][1], 
                                self.line_scaled[0], self.line_scaled[1]):
                    if "North" in direction:
                        # Highlight violation
                        cv2.line(img, start_point, end_point, (255, 255, 255), 3)
                        # Draw movement path
                        unscaled_prev = self.unscale_coordinates(*self.data_deque[id][1])
                        unscaled_curr = self.unscale_coordinates(*self.data_deque[id][0])
                        cv2.line(img, (int(unscaled_curr[0]), int(unscaled_curr[1])), 
                                (int(unscaled_prev[0]), int(unscaled_prev[1])), (0, 0, 0), 3)
                        self.object_counter[obj_name] += 1
                        self.UI_box([x1, y1, x2, y2], img_copy, label=label, color=(255, 0, 0), line_thickness=2)
                        cv2.line(img_copy, start_point, end_point, (255, 255, 255), 3)
                        output_file = f"{self.folder_path}/fault_vehicle_{self.num_faults}.png"
                        cv2.imwrite(output_file, img_copy)
                        self.num_faults += 1
                        violations.append({
                            'track_id': id,
                            'timestamp': time.time(),
                            'cam_id': self.config['cam_id'],
                            'violation_type': 'lane_crossing'
                        })

            self.UI_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=2)

        return img, violations

    def process_frame(self, frame):
        # Resize frame for inference
        inference_frame = cv2.resize(frame, self.model_input_size, interpolation=cv2.INTER_LINEAR)
        
        # Create polygons with scaled coordinates
        roi_poly = self.create_polygon(self.roi, scale=True)
        traffic_light_poly = self.create_polygon(self.traffic_light_zone, scale=True)
        
        # Detect traffic light color
        self.traffic_light_state = self.detect_traffic_light_colors(inference_frame, traffic_light_poly)
        
        # Track objects
        results = self.model.track(inference_frame, persist=True, imgsz=self.model_input_size)[0]
        
        violations = []
        processed_frame = frame.copy()  # Work on original frame for display
        
        if results.boxes and results.boxes.id is not None:
            # Convert xywh to xyxy
            boxes_xywh = results.boxes.xywh.cpu()
            boxes_xyxy = []
            for box in boxes_xywh:
                x, y, w, h = box
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes_xyxy.append([x1, y1, x2, y2])
            
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            names = self.model.names
            
            # Draw boxes and check for violations
            mode = 'on' if self.traffic_light_state == 'red' else 'off'
            processed_frame, line_violations = self.draw_boxes(
                processed_frame, boxes_xyxy, names, classes, track_ids, mode=mode
            )
            violations.extend(line_violations)
            
            # Check for red light violations
            for box, track_id, cls in zip(boxes_xywh, track_ids, classes):
                if cls in [2, 3, 5, 7]:  # Vehicle classes
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    
                    if self.traffic_light_state == 'red' and self.is_point_in_polygon((x, y), traffic_light_poly):
                        # Convert to original coordinates for display
                        x_orig, y_orig = self.unscale_coordinates(x, y)
                        w_orig, h_orig = w / self.scale_x, h / self.scale_y
                        violations.append({
                            'track_id': track_id,
                            'timestamp': time.time(),
                            'cam_id': self.config['cam_id'],
                            'violation_type': 'red_light'
                        })
                        cv2.putText(processed_frame, 'VIOLATION', 
                                  (int(x_orig - w_orig/2), int(y_orig - h_orig/2 - 30)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw ROI and traffic light zone
        original_roi_poly = self.create_polygon(self.roi, scale=False)
        original_traffic_light_poly = self.create_polygon(self.traffic_light_zone, scale=False)
        cv2.polylines(processed_frame, [original_roi_poly.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(processed_frame, [original_traffic_light_poly.astype(np.int32)], True, (0, 0, 255), 2)
        
        # Draw lane marking
        original_start_point = (int(self.lane_marking['start_point']['x']), 
                              int(self.lane_marking['start_point']['y']))
        original_end_point = (int(self.lane_marking['end_point']['x']), 
                            int(self.lane_marking['end_point']['y']))
        cv2.line(processed_frame, original_start_point, original_end_point, 
                (255, 255, 0) if self.traffic_light_state != 'red' else (255, 0, 0), 2)
        
        return processed_frame, violations

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            # Process every 3rd frame to reduce load
            if self.frame_count % 3 != 0:
                continue
            
            processed_frame, violations = self.process_frame(frame)

            # Display traffic light state
            text = f'Traffic Light: {self.traffic_light_state}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = self.original_width - text_size[0] - 10
            cv2.putText(processed_frame, text, 
                       (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            for violation in violations:
                print(f"Violation detected: {violation}")

            if self.output_path:
                self.out.write(processed_frame)

            cv2.imshow('Red Light Violation Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.output_path:
            self.out.release()
        cv2.destroyAllWindows()

detector = RedLightViolationDetector(
    config_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\test\cam4_traffic_light.json',
    model_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\model\choose_model\vehicle_detection.onnx',
    video_source=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\debug_segment_0.mp4',
    output_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\tracked_output_testing_0.mp4'
)
# Run detection
detector.run()