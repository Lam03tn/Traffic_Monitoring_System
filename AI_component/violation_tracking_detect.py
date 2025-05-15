import os
import cv2
from matplotlib import pyplot as plt
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
        self.roi = self.create_polygon(config.get('violation_config', [{}])[0].get('roi', {}))
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
    
    def is_in_roi(self, center_point):
        """Check if the center point is inside the ROI"""
        return self.is_point_in_polygon(center_point, self.roi)
    
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

    def detect(self, frame, tracked_detections, traffic_light_state, data_deque, frame_buffer=None):
        """Base method to detect violation - to be overridden"""
        return None
    
    def detect_license_plate(self, frame, vehicle_box):
        """
        License plate detection and recognition function for a single vehicle crop
        
        Args:
            frame: Full camera frame
            vehicle_box: Bounding box of the vehicle (x1, y1, x2, y2)
            
        Returns:
            tuple: (license_plate_text, confidence_score, annotated_plate_image)
        """
        x1, y1, x2, y2 = vehicle_box
        
        try:
            # Step 1: Crop the vehicle region from the frame
            vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if vehicle_crop.size == 0:
                return "unknown", 0.0, None
                
            # Step 2: Preprocess the vehicle crop directly for plate detection
            _, _, frame_normalized, original_size = preprocess_frame(frame)
            
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
                
            # Step 5: Process each potential license plate
            license_plate_text = "unknown"
            annotated_plate = None
            best_conf_score = 0.0
            
            for plate_idx, (plate_box, plate_score) in enumerate(zip(plate_boxes, plate_scores)):
                px, py, pw, ph = plate_box
                px1, px2, py1, py2 = px - pw/2, px + pw/2, py - ph/2, py + ph/2

                if not(px1 >= x1 and px2 <=x2 and py1 >= y1 and py2 <= y2 ):
                    continue

                # Crop the license plate from the vehicle crop
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
                if plate_crop.size == 0:
                    continue
                # Process license plate to extract text
                lp_text, char_result, processed_plate = process_license_plate(
                    self.triton_client, plate_crop
                )
                annotated_plate = processed_plate
                # Calculate confidence score
                total_conf_score = 0.0
                if lp_text != "unknown" and char_result and len(char_result[1]) > 0:
                    total_conf_score = sum(char_result[1]) / len(char_result[1])
                
                # Update if this detection is better
                if lp_text != "unknown" and total_conf_score > best_conf_score:
                    best_conf_score = total_conf_score
                    license_plate_text = lp_text
            
            return license_plate_text, best_conf_score, annotated_plate
        
        except Exception as e:
            print(f"Error in license plate detection: {str(e)}")
            return "unknown", 0.0, None

class TrafficLightViolationHandler(ViolationHandler):
    """Handler for traffic light violations with optimized license plate detection"""
    def __init__(self, config, camera_id, triton_server_url="localhost:8001"):
        super().__init__(config, camera_id, triton_server_url)
        self.previous_traffic_light_state = 'unknown'
        self.violation_history = {}
        self.cooldown_period = 10.0

    def cleanup_violation_history(self, max_age_seconds=3600):  # Remove entries older than 1 hour
        current_time = time.time()
        expired_keys = [track_id for track_id, timestamp in self.violation_history.items()
                        if current_time - timestamp > max_age_seconds]
        for track_id in expired_keys:
            del self.violation_history[track_id]

    def detect(self, frame, tracked_detections, traffic_light_state, data_deque, frame_buffer=None):
        """
        Detect traffic light violations by checking current and previous traffic light states
        """
        self.cleanup_violation_history()
        if not (self.previous_traffic_light_state == 'red' and traffic_light_state == 'red'):
            self.previous_traffic_light_state = traffic_light_state
            return []
        
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }
        
        violations = []
        frame_violation_after = frame.copy()
        _frame, _, _ = frame_buffer[-2]
        frame_violation_before = _frame.copy()

        current_time = time.time()

        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y2 + y2) / 2
            track_id = tracked_detections.tracker_id[i]
            
            # Check if vehicle center is inside ROI - only process vehicles in ROI
            if not self.is_in_roi((center_x, center_y)):
                continue
            
            if track_id in self.violation_history:
                last_violation_time = self.violation_history[track_id]
                if current_time - last_violation_time < self.cooldown_period:
                    continue

            if track_id in data_deque and len(data_deque[track_id]) >= 2:
                curr_pos = data_deque[track_id][0]
                prev_pos = data_deque[track_id][1]
                direction = self.get_direction(curr_pos, prev_pos)
                
                if self.intersect(
                    curr_pos, prev_pos, 
                    self.lane_marking_start, self.lane_marking_end
                ):
                    if "North" in direction:
                        prev_x1, prev_x2, prev_y1, prev_y2 = prev_pos[0] - (x2-x1)//2, prev_pos[0] + (x2-x1)//2, prev_pos[1] - (y2-y1), prev_pos[1]
                        
                        vehicle_box = (x1, y1, x2, y2)
                        lp_text = ''
                        plate_img = None
                        lp_text_after, conf_after, plate_img_after = self.detect_license_plate(
                            frame_violation_after, vehicle_box
                        )
                        lp_text_before, conf_before, plate_img_before = self.detect_license_plate(
                            frame_violation_before, vehicle_box
                        )
                        if conf_before > conf_after:
                            lp_text = lp_text_before
                            plate_img = plate_img_before
                        else:
                            lp_text = lp_text_after
                            plate_img = plate_img_after

                        # Vẽ hộp giới hạn
                        cv2.rectangle(
                            frame_violation_after,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color=(0, 0, 255),
                            thickness=2
                        )
                        # Thêm track_id trên khung hình
                        cv2.putText(
                            frame_violation_after,
                            f"Track ID: {track_id}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                        cv2.rectangle(
                            frame_violation_before,
                            (int(prev_x1), int(prev_y1)),
                            (int(prev_x2), int(prev_y2)),
                            color=(0, 0, 255),
                            thickness=2
                        )
                        cv2.putText(
                            frame_violation_before,
                            f"Track ID: {track_id}",
                            (int(prev_x1), int(prev_y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        
                        # Hiển thị trạng thái đèn giao thông
                        status_color = color_map.get(traffic_light_state, (255, 255, 255))
                        box_top_right = (frame.shape[1] - frame.shape[1]//6, 20)
                        box_bottom_right = (frame.shape[1] - frame.shape[1]//6, 50)
                        
                        cv2.rectangle(
                            frame_violation_after,
                            box_top_right,
                            box_bottom_right,
                            status_color,
                            thickness=-1
                        )
                        cv2.putText(
                            frame_violation_after,
                            f"Traffic Light: {traffic_light_state.upper()}",
                            (box_bottom_right[0] + 10, box_bottom_right[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            status_color,
                            2
                        )

                        cv2.rectangle(
                            frame_violation_before,
                            box_top_right,
                            box_bottom_right,
                            status_color,
                            thickness=-1
                        )
                        cv2.putText(
                            frame_violation_before,
                            f"Traffic Light: {traffic_light_state.upper()}",
                            (box_bottom_right[0] + 10, box_bottom_right[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            status_color,
                            2
                        )
                        cv2.line(frame_violation_before, 
                                (int(self.lane_marking_start[0]), int(self.lane_marking_start[1])),
                                (int(self.lane_marking_end[0]), int(self.lane_marking_end[1])),
                                (0, 255, 0), 5)
                        cv2.line(frame_violation_after, 
                                (int(self.lane_marking_start[0]), int(self.lane_marking_start[1])),
                                (int(self.lane_marking_end[0]), int(self.lane_marking_end[1])),
                                (0, 255, 0), 5)
                        
                        # Draw ROI polygon on frames
                        if self.roi.size > 0:
                            cv2.polylines(
                                frame_violation_before, 
                                [self.roi.astype(np.int32)],
                                isClosed=True,
                                color=(255, 0, 0),
                                thickness=2
                            )
                            cv2.polylines(
                                frame_violation_after, 
                                [self.roi.astype(np.int32)],
                                isClosed=True,
                                color=(255, 0, 0),
                                thickness=2
                            )
                        
                        # Tạo ID vi phạm
                        violation_id = str(uuid.uuid4())
                        
                        # Tạo bản ghi vi phạm
                        violation = {
                            'violation_id': violation_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'license_plate': lp_text,
                            'violation_timestamp': current_time,
                            'camera_id': self.camera_id,
                            'track_id': track_id
                        }

                        # Bao gồm hình ảnh biển số xe nếu có
                        if plate_img is not None:
                            violation['frame'] = [frame_violation_after, frame_violation_before, plate_img]
                        else:
                            violation['frame'] = [frame_violation_after, frame_violation_before, None]

                        violations.append(violation)
                        
                        # Cập nhật lịch sử vi phạm
                        self.violation_history[track_id] = current_time

        self.previous_traffic_light_state = traffic_light_state
        return violations

class LineCrossingViolationHandler(ViolationHandler):
    """Handler for line crossing violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque, frame_buffer=None):
        if traffic_light_state != 'red':
            return []

        violations = []
        
        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check if vehicle center is inside ROI - only process vehicles in ROI
            if not self.is_in_roi((center_x, center_y)):
                continue
            
            if track_id in data_deque and len(data_deque[track_id]) >= 2:
                curr_pos = data_deque[track_id][0]
                prev_pos = data_deque[track_id][1]
                direction = self.get_direction(curr_pos, prev_pos)
                
                if self.intersect(
                    curr_pos, prev_pos, 
                    self.lane_marking_start, self.lane_marking_end
                ):
                    if "North" in direction:
                        violation_frame = None
                        if frame_buffer:
                            violation_frame = frame.copy()
                            # Draw ROI polygon on frame
                            if self.roi.size > 0:
                                cv2.polylines(
                                    violation_frame, 
                                    [self.roi.astype(np.int32)],
                                    isClosed=True,
                                    color=(255, 0, 0),
                                    thickness=2
                                )
                            # Draw line marking
                            cv2.line(
                                violation_frame,
                                (int(self.lane_marking_start[0]), int(self.lane_marking_start[1])),
                                (int(self.lane_marking_end[0]), int(self.lane_marking_end[1])),
                                (0, 255, 0), 5
                            )
                            # Draw vehicle bounding box
                            cv2.rectangle(
                                violation_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color=(0, 0, 255),
                                thickness=2
                            )

                        violations.append({
                            'track_id': track_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'timestamp': time.time(),
                            'direction': direction,
                            'frame': [violation_frame, violation_frame, None] if violation_frame is not None else None
                        })
        
        return violations

class WrongWayViolationHandler(ViolationHandler):
    """Handler for wrong way violations"""
    def detect(self, frame, tracked_detections, traffic_light_state, data_deque, frame_buffer=None):
        violations = []

        for i in range(len(tracked_detections.xyxy)):
            cls = tracked_detections.class_id[i]
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check if vehicle center is inside ROI
            if self.is_in_roi((center_x, center_y)):
                if track_id in data_deque and len(data_deque[track_id]) >= 2:
                    curr_pos = data_deque[track_id][0]
                    prev_pos = data_deque[track_id][1]
                    direction = np.arctan2(curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0])
                    allowed_direction = np.radians(self.config.get('allowed_direction', 0))
                    direction_tolerance = np.radians(self.config.get('direction_tolerance', 45))
                    
                    if abs(direction - allowed_direction) > direction_tolerance:
                        violation_frame = None
                        if frame_buffer:
                            violation_frame = frame.copy()
                            # Draw ROI polygon on frame
                            if self.roi.size > 0:
                                cv2.polylines(
                                    violation_frame, 
                                    [self.roi.astype(np.int32)],
                                    isClosed=True,
                                    color=(255, 0, 0),
                                    thickness=2
                                )
                            # Draw vehicle bounding box
                            cv2.rectangle(
                                violation_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color=(0, 0, 255),
                                thickness=2
                            )
                            # Draw movement direction
                            cv2.arrowedLine(
                                violation_frame,
                                (int(prev_pos[0]), int(prev_pos[1])),
                                (int(curr_pos[0]), int(curr_pos[1])),
                                color=(0, 0, 255),
                                thickness=2
                            )
                            
                        violations.append({
                            'track_id': track_id,
                            'center': (center_x, center_y),
                            'box': (x1, y1, x2, y2),
                            'timestamp': time.time(),
                            'frame': [violation_frame, violation_frame, None] if violation_frame is not None else None
                        })

        return violations