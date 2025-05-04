import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from collections import defaultdict

class RedLightViolationDetector:
    def __init__(self, config_path, model_path, video_source, output_path=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize YOLOv8 model with explicit task
        self.model = YOLO(model_path, task='detect')
        
        # Video source
        self.cap = cv2.VideoCapture(video_source)
        
        # Get original video dimensions
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1280
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 960
        self.model_input_size = (640, 640)  # Fixed input size for ONNX model
        
        # Scaling factors for coordinate transformation
        self.scale_x = self.model_input_size[1] / self.original_width  # 640/1280
        self.scale_y = self.model_input_size[0] / self.original_height  # 640/960
        
        # Video output setup
        self.output_path = output_path
        if output_path:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                     (self.original_width, self.original_height))
        
        # Extract configuration parameters (assuming 4 points for ROI and traffic light, 2 points for lane marking)
        self.roi = self.config['violation_config'][0]['roi']
        self.traffic_light_zone = self.config['violation_config'][0]['traffic_light_zone']
        self.lane_marking = self.config['violation_config'][0]['lane_marking']
        
        # Traffic light state (initially unknown)
        self.traffic_light_state = 'unknown'  # Will be updated by detect_traffic_light_colors
        
        # Tracking history
        self.track_history = defaultdict(lambda: [])
        
    def scale_coordinates(self, x, y):
        """Scale coordinates from original size to model input size"""
        return x * self.scale_x, y * self.scale_y

    def create_polygon(self, points, scale=True):
        """Create polygon from points dictionary (4 points expected)"""
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
        # Convert polygon to bounding box for cropping
        x, y, w, h = cv2.boundingRect(traffic_light_poly.astype(np.int32))
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Crop the traffic light region
        traffic_light_region = frame[y:y+h, x:x+w]
        if traffic_light_region.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(traffic_light_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (HSV)
        # RED - Chỉ giữ giá trị rất thấp hoặc rất cao của hue
        lower_red1 = np.array([0, 150, 70])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 150, 70])
        upper_red2 = np.array([180, 255, 255])

        # YELLOW/ORANGE - Mở rộng cho cả cam đậm
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count non-zero pixels in each mask
        red_count = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count = cv2.countNonZero(mask_green)
        print("red_count: ", red_count," yellow_count: ", yellow_count, " green_count: ")
        
        # Determine dominant color
        max_count = max(red_count, yellow_count, green_count)
        if max_count < 50:  # Threshold to avoid false positives
            return 'unknown'
        
        if max_count == red_count:
            return 'red'
        elif max_count == yellow_count:
            return 'yellow'
        else:
            return 'green'

    def process_frame(self, frame):
        # Resize frame to model input size (640x640) for inference
        inference_frame = cv2.resize(frame, self.model_input_size, interpolation=cv2.INTER_LINEAR)
        
        # Create polygons with scaled coordinates for inference
        roi_poly = self.create_polygon(self.roi, scale=True)
        traffic_light_poly = self.create_polygon(self.traffic_light_zone, scale=True)
        
        # Scale lane marking points
        start_point = self.scale_coordinates(self.lane_marking['start_point']['x'], 
                                           self.lane_marking['start_point']['y'])
        end_point = self.scale_coordinates(self.lane_marking['end_point']['x'], 
                                         self.lane_marking['end_point']['y'])
        
        # Detect traffic light color
        self.traffic_light_state = self.detect_traffic_light_colors(inference_frame, traffic_light_poly)
        
        # Track objects using YOLOv8
        results = self.model.track(inference_frame, persist=True, imgsz=self.model_input_size)[0]
        
        violations = []
        if results.boxes and results.boxes.id is not None:
            boxes = results.boxes.xywh.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            
            # Draw detection results
            inference_frame = results.plot()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                # Only process vehicles (assuming classes 2, 3, 5, 7 are vehicles)
                if cls in [2, 3, 5, 7]:
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    
                    # Check if vehicle is in ROI
                    if self.is_point_in_polygon((x, y), roi_poly):
                        # Check for red light violation
                        if self.traffic_light_state == 'red':
                            # Check if vehicle crosses traffic light zone
                            if self.is_point_in_polygon((x, y), traffic_light_poly):
                                violations.append({
                                    'track_id': track_id,
                                    'timestamp': time.time(),
                                    'cam_id': self.config['cam_id'],
                                    'violation_type': 'red_light'
                                })
                                # Draw violation indicator
                                cv2.putText(inference_frame, 'VIOLATION', (int(x - w/2), int(y - h/2 - 30)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Resize inference frame back to original size
        processed_frame = cv2.resize(inference_frame, (self.original_width, self.original_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Draw ROI, traffic light zone, and lane marking on original-sized frame
        original_roi_poly = self.create_polygon(self.roi, scale=False)
        original_traffic_light_poly = self.create_polygon(self.traffic_light_zone, scale=False)
        cv2.polylines(processed_frame, [original_roi_poly.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(processed_frame, [original_traffic_light_poly.astype(np.int32)], True, (0, 0, 255), 2)
        
        original_start_point = (int(self.lane_marking['start_point']['x']), 
                              int(self.lane_marking['start_point']['y']))
        original_end_point = (int(self.lane_marking['end_point']['x']), 
                            int(self.lane_marking['end_point']['y']))
        cv2.line(processed_frame, original_start_point, original_end_point, (255, 255, 0), 2)
        
        return processed_frame, violations

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, violations = self.process_frame(frame)
            
            # Display traffic light state
            cv2.putText(processed_frame, f'Traffic Light: {self.traffic_light_state}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Log violations
            for violation in violations:
                print(f"Violation detected: {violation}")
            
            # Write to output video if specified
            if self.output_path:
                self.out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Red Light Violation Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        if self.output_path:
            self.out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize detector
    detector = RedLightViolationDetector(
        config_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\test\cam_config_violation_testing_cam4.json',
        model_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\model\choose_model\vehicle_detection.onnx',
        video_source=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\debug_segment_3.mp4',
        output_path=r'C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\tracked_output_testing.mp4'
    )
    
    # Run detection
    detector.run()