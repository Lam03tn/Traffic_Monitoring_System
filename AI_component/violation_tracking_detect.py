import cv2
import numpy as np
import json
import time
from collections import defaultdict, deque
import uuid
import io
import re
import tritonclient.grpc as triton_grpc
from license_plate_module import postprocess, preprocess_frame, process_license_plate, is_box_inside, triton_infer

class ViolationHandler:
    """Base class for handling specific violation types"""
    def __init__(self, config, camera_id, triton_server_url="localhost:8001"):
        self.config = config
        self.camera_id = camera_id
        self.roi = self.create_polygon(config.get('roi', {}))
        self.traffic_light_zone = self.create_polygon(config.get('violation_config', [{}])[0].get('traffic_light_zone', {}))
        self.violation_type = config.get('violation_type', 'unknown')
        self.triton_client = triton_grpc.InferenceServerClient(url=triton_server_url)
        
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
    
    def detect_license_plate(self, frame, vehicle_box, track_id):
        """
        Enhanced license plate detection and recognition function
        
        Args:
            frame: Full camera frame
            vehicle_box: Bounding box of the vehicle (x1, y1, x2, y2)
            track_id: Tracking ID of the vehicle
            
        Returns:
            tuple: (license_plate_text, confidence_score, annotated_plate_image)
        """
        x1, y1, x2, y2 = vehicle_box
        
        # Step 1: Crop the vehicle from the frame
        vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if vehicle_crop.size == 0:
            return "unknown", 0.0, None
        
        try:
            # Step 2: Preprocess the vehicle crop for plate detection
            _, _, frame_normalized, original_size = preprocess_frame(vehicle_crop)
            
            # Step 3: Run plate detection using Triton
            plate_output = triton_infer(self.triton_client, "plate_detection", frame_normalized)
            if plate_output is None:
                return "unknown", 0.0, None
            
            # Step 4: Postprocess plate detection results
            num_detections, plate_boxes, plate_scores, plate_class_ids = postprocess(
                plate_output, original_size, score_threshold=0.35
            )
            
            if num_detections == 0 or len(plate_boxes) == 0:
                return "unknown", 0.0, None
                
            # Step 5: Process each potential license plate and select the best one
            license_plate_text = "unknown"
            annotated_plate = None
            best_conf_score = 0.0
            
            for plate_idx, (plate_box, plate_score) in enumerate(zip(plate_boxes, plate_scores)):
                # Skip low confidence plates
                if plate_score < 0.5:
                    continue
                    
                px1, py1, px2, py2 = plate_box
                
                # Ensure plate coordinates are within vehicle crop boundaries
                px1 = max(0, min(px1, vehicle_crop.shape[1]-1))
                py1 = max(0, min(py1, vehicle_crop.shape[0]-1))
                px2 = max(0, min(px2, vehicle_crop.shape[1]-1))
                py2 = max(0, min(py2, vehicle_crop.shape[0]-1))
                
                # Skip if plate dimensions are too small
                if (px2 - px1 < 10) or (py2 - py1 < 5):
                    continue
                    
                # Crop the license plate
                plate_crop = vehicle_crop[int(py1):int(py2), int(px1):int(px2)]
                if plate_crop.size == 0:
                    continue
                    
                # Process license plate to extract text
                lp_text, char_result, processed_plate = process_license_plate(
                    self.triton_client, plate_crop
                )
                
                # Calculate confidence score
                if lp_text != "unknown" and char_result and "scores" in char_result:
                    conf_scores = char_result["scores"]
                    if len(conf_scores) > 0:
                        total_conf_score = conf_scores.sum() / len(conf_scores)
                        
                        # Filter out extremely short license plates (likely errors)
                        if len(lp_text) >= 4 and total_conf_score > best_conf_score:
                            best_conf_score = total_conf_score
                            license_plate_text = lp_text
                            annotated_plate = processed_plate
                            
                            # Draw the plate location on the vehicle crop for visualization
                            cv2.rectangle(
                                vehicle_crop,
                                (int(px1), int(py1)),
                                (int(px2), int(py2)),
                                color=(0, 255, 0),  # Green
                                thickness=2
                            )
            
            # Step 6: Validate license plate format using regex (customize for your region)
            if license_plate_text != "unknown" and self.validate_license_plate_format(license_plate_text):
                # Store in vehicle tracking history for consistency
                if track_id in self.license_plate_history:
                    # Update the history with the most confident detection
                    history = self.license_plate_history[track_id]
                    if best_conf_score > history["best_confidence"]:
                        history["plates"][license_plate_text] = history["plates"].get(license_plate_text, 0) + 1
                        history["best_confidence"] = best_conf_score
                        history["best_plate"] = license_plate_text
                else:
                    # Create new history entry
                    self.license_plate_history[track_id] = {
                        "plates": {license_plate_text: 1},
                        "best_confidence": best_conf_score,
                        "best_plate": license_plate_text,
                        "frames_tracked": 1
                    }
                
                # Use the most consistent plate detection over time
                if track_id in self.license_plate_history:
                    history = self.license_plate_history[track_id]
                    if history["frames_tracked"] > 5:  # Require multiple detections for stability
                        # Find the most common plate in history
                        most_common_plate = max(
                            history["plates"].items(), 
                            key=lambda x: x[1]
                        )[0]
                        license_plate_text = most_common_plate
            
            return license_plate_text, best_conf_score, annotated_plate
        
        except Exception as e:
            print(f"Error in license plate detection: {str(e)}")
            return "unknown", 0.0, None
        
    def validate_license_plate_format(self, plate_text):
        """
        Validate license plate format using regular expressions
        Customize this method based on your region's plate format
        
        Args:
            plate_text: Detected license plate text
            
        Returns:
            bool: True if the format is valid, False otherwise
        """
        # Example for Vietnamese license plates (customize as needed)
        # Common formats: 
        # - 59A-123.45 (old format)
        # - 59F-123.45 (new format for cars)
        # - 59K1-123.45 (motorcycles)
        
        # Basic pattern - adjust for your specific region
        # This is an example pattern that accepts formats like:
        # - 2 digits + 1-2 letters + (optional dash) + 5-6 digits (with optional dot)
        pattern = r'^\d{2}[A-Z]{1,2}[-]?\d{3}[.]?\d{2}$'
        
        return bool(re.match(pattern, plate_text))

class TrafficLightViolationHandler(ViolationHandler):
    """Enhanced handler for traffic light violations with improved license plate detection"""
    def __init__(self, config, camera_id, triton_server_url="localhost:8001"):
        super().__init__(config, camera_id, triton_server_url)
        # License plate tracking history
        self.license_plate_history = {}
        # Minimum confidence threshold for license plate detection
        self.license_plate_confidence_threshold = 0.6
        
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        """
        Detect traffic light violations
        
        Args:
            frame: Current video frame
            tracked_detections: Object tracking results
            traffic_light_state: Current traffic light state ('red', 'yellow', 'green')
            data_deque: History of tracked object positions
            
        Returns:
            list: List of detected violations
        """
        # Only detect violations during red light
        if traffic_light_state != 'red':
            return []
        
        # Define color map for visualization
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }
        
        violations = []
        frame_violation = frame.copy()
        
        # Iterate through tracked objects
        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            
            # Skip if not a vehicle class (customize based on your model)
            if cls not in [2, 3, 5, 7]:  # Common vehicle classes in COCO
                continue

            # Get bounding box
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            track_id = tracked_detections.tracker_id[i]
            
            # Update tracking history
            if track_id in self.license_plate_history:
                self.license_plate_history[track_id]["frames_tracked"] += 1
            
            # Check for line crossing violation
            if track_id in data_deque and len(data_deque[track_id]) >= 2:
                # Get current and previous positions
                curr_pos = data_deque[track_id][0]
                prev_pos = data_deque[track_id][1]
                
                # Check direction and intersection with lane marking line
                direction = self.get_direction(curr_pos, prev_pos)
                
                if self.intersect(
                    curr_pos, prev_pos, 
                    self.lane_marking_start, self.lane_marking_end
                ):
                    # Check if movement direction matches violation criteria
                    # Customize based on your camera setup (North, South, East, West)
                    if "North" in direction:
                        # Vehicle is crossing the line in the monitored direction during red light
                        
                        # Get vehicle bounding box
                        vehicle_box = (x1, y1, x2, y2)
                        
                        # Detect license plate
                        license_plate_text, confidence, annotated_plate = self.detect_license_plate(
                            frame, vehicle_box, track_id
                        )
                        
                        # Draw bounding box on the frame
                        cv2.rectangle(
                            frame_violation,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color=(0, 0, 255),  # Red color in BGR
                            thickness=2
                        )
                        
                        # Add track ID as text
                        cv2.putText(
                            frame_violation,
                            f"ID: {track_id}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),  # Red color
                            2
                        )
                        
                        # Add license plate info
                        if license_plate_text != "unknown":
                            cv2.putText(
                                frame_violation,
                                f"LP: {license_plate_text} ({confidence:.2f})",
                                (int(x1), int(y1) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),  # Red color
                                2
                            )
                        
                        # Display traffic light status
                        status_color = color_map.get(traffic_light_state, (255, 255, 255))
                        
                        # Status display location (top right corner)
                        box_top_right = (frame.shape[1] - 200, 20)
                        box_bottom_right = (frame.shape[1] - 170, 50)
                        
                        # Draw colored box to indicate light status
                        cv2.rectangle(
                            frame_violation,
                            box_top_right,
                            box_bottom_right,
                            status_color,
                            thickness=-1  # Filled rectangle
                        )
                        
                        # Add text label
                        cv2.putText(
                            frame_violation,
                            f"Traffic Light: {traffic_light_state.upper()}",
                            (box_bottom_right[0] + 10, box_bottom_right[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            status_color,
                            2
                        )
                        
                        # Generate unique violation ID
                        violation_id = str(uuid.uuid4())
                        
                        # Create violation record
                        violations.append({
                            'violation_id': violation_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'frame': frame_violation,
                            'license_plate': license_plate_text,
                            'license_plate_confidence': confidence,
                            'violation_timestamp': time.time(),
                            'violation_type': 'traffic_light_violation',
                            'traffic_light_state': traffic_light_state,
                            'vehicle_direction': direction,
                            'camera_id': self.camera_id
                        })
                        
                        # If we have a license plate with good confidence, include the annotated plate image
                        if license_plate_text != "unknown" and confidence > self.license_plate_confidence_threshold and annotated_plate is not None:
                            violations[-1]['license_plate_image'] = annotated_plate

        return violations
    
    def cleanup_tracking_history(self):
        """Remove old tracking records to prevent memory leaks"""
        current_time = time.time()
        old_tracks = []
        
        for track_id, history in self.license_plate_history.items():
            # Remove tracks not seen in the last 60 seconds
            if "last_seen" in history and (current_time - history["last_seen"]) > 60:
                old_tracks.append(track_id)
        
        # Remove old tracks
        for track_id in old_tracks:
            del self.license_plate_history[track_id]

class LineCrossingViolationHandler(ViolationHandler):
    """Handler for line crossing violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque):
        if traffic_light_state != 'red':
            return []

        violations = []
        
        # Check for lane-crossing violations
        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
                
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