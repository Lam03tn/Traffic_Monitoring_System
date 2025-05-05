from matplotlib import pyplot as plt
import redis
import json
import numpy as np
import cv2
from bytetrack.byte_track import ByteTrack
from queue import Queue
from supervision import Detections
import base64
import os
import time


# Redis settings
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
CAMERA_ID = 'cam4'
INFERENCE_CHANNEL = f'inferences_{CAMERA_ID}'

# Tracking settings
MAX_FRAMES_TRACKING = 40
FRAME_RATE_TARGET = 3

# Video output settings
OUTPUT_DIR = 'output_videos'
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f'tracking_{CAMERA_ID}_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
VIDEO_CODEC = 'mp4v'  # Use 'avc1' for H.264 if available on your system
VIDEO_FPS = FRAME_RATE_TARGET

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
pubsub = redis_client.pubsub()

# Tracking worker
def tracking_worker():
    # Create output directory if it doesn't exist
    written_frame_count = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tracker = ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=40,
        minimum_matching_threshold=0.8,
        frame_rate=FRAME_RATE_TARGET
    )
    
    frame_queue = Queue(maxsize=MAX_FRAMES_TRACKING)
    tracking_results = []
    
    # Initialize video writer
    video_writer = None
    
    pubsub.subscribe(INFERENCE_CHANNEL)
    
    print(f"[✓] ByteTrack worker started, subscribed to {INFERENCE_CHANNEL}")
    print(f"[✓] Output video will be saved to: {OUTPUT_VIDEO_PATH}")
    
    for message in pubsub.listen():
        if message['type'] != 'message':
            continue
            
        try:
            inference_result = json.loads(message['data'])
            detections = inference_result['detections']
            frame_index = inference_result['frame_index']
            metadata = inference_result['metadata']
            frame_base64 = inference_result.get('frame_data')
            
            # Get frame dimensions from metadata
            frame_shape = metadata['frame_shape']
            frame_height, frame_width = frame_shape[0], frame_shape[1]
            
            # Decode base64 frame
            if frame_base64:
                try:
                    frame_data = base64.b64decode(frame_base64)
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Failed to decode frame")
                except Exception as e:
                    print(f"[!] Error decoding frame {frame_index}: {e}")
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                print(f"[!] No frame data for frame {frame_index}, using blank frame")
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Initialize video writer if not already done
            if video_writer is None and frame is not None:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                    video_writer = cv2.VideoWriter(
                        OUTPUT_VIDEO_PATH,
                        fourcc,
                        VIDEO_FPS,
                        (frame_width, frame_height)
                    )
                    
                    # Check if video writer is initialized properly
                    if not video_writer.isOpened():
                        raise ValueError(f"Failed to open video writer with codec {VIDEO_CODEC}")
                    
                    print(f"[✓] Video writer initialized: {OUTPUT_VIDEO_PATH}")
                except Exception as e:
                    print(f"[!] Error initializing video writer: {e}")
                    # Fallback to MJPG codec if the specified one fails
                    try:
                        print("[*] Trying fallback codec MJPG...")
                        video_writer = cv2.VideoWriter(
                            OUTPUT_VIDEO_PATH,
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            VIDEO_FPS,
                            (frame_width, frame_height)
                        )
                        if video_writer.isOpened():
                            print("[✓] Video writer initialized with fallback codec")
                        else:
                            print("[!] Failed to initialize video writer with fallback codec")
                    except Exception as fallback_error:
                        print(f"[!] Error with fallback codec: {fallback_error}")
            
            # Prepare detections for Supervision Detections format
            xyxy = []
            confidence = []
            class_id = []
            for det in detections:
                x, y, w, h = det['box']
                conf = det['score']
                cls = det['class_id']
                # Convert from xywh to xyxy format
                xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])
                confidence.append(conf)
                class_id.append(cls)
            
            # Convert to numpy arrays
            xyxy = np.array(xyxy) if xyxy else np.empty((0, 4))
            confidence = np.array(confidence) if confidence else np.empty((0,))
            class_id = np.array(class_id) if class_id else np.empty((0,), dtype=np.int32)
            
            # Create Supervision Detections object
            supervision_detections = Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            
            # Update tracker with Supervision Detections
            tracked_detections = tracker.update_with_detections(supervision_detections)
            
            # Collect tracking results and draw on frame
            tracks = []
            for i in range(len(tracked_detections)):
                tlwh = tracked_detections.xyxy[i]
                tid = tracked_detections.tracker_id[i]
                score = tracked_detections.confidence[i]
                cls = tracked_detections.class_id[i]
                
                # Convert to tlwh format for tracking result
                tracks.append({
                    'track_id': int(tid),
                    'box': [float(tlwh[0]), float(tlwh[1]), float(tlwh[2] - tlwh[0]), float(tlwh[3] - tlwh[1])],
                    'score': float(score),
                    'class_id': int(cls)
                })
                
                # Draw bounding box and track ID
                x1, y1, x2, y2 = map(int, tlwh)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Write frame to video
            if video_writer is not None and video_writer.isOpened() and written_frame_count < MAX_FRAMES_TRACKING:
                try:
                    video_writer.write(frame)
                    written_frame_count += 1
                    if written_frame_count >= MAX_FRAMES_TRACKING:
                        print("[✓] Reached maximum frame count, stopping tracking worker.")
                        break
                except Exception as e:
                    print(f"[!] Error writing frame {frame_index} to video: {e}")
            elif written_frame_count >= MAX_FRAMES_TRACKING:
                print(f"[*] Reached max frame limit ({MAX_FRAMES_TRACKING}), skipping frame {frame_index}")
            else:
                print(f"[!] Video writer not available for frame {frame_index}")
            
            # Collect tracking result
            tracking_result = {
                'frame_index': frame_index,
                'tracks': tracks,
                'metadata': metadata
            }
            
            # Maintain frame queue
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(tracking_result)
            
            # Update tracking results
            tracking_results.append(tracking_result)
            if len(tracking_results) > MAX_FRAMES_TRACKING:
                tracking_results.pop(0)
            
            print(f"[✓] Processed tracking for frame {frame_index}, {len(tracks)} tracks")
        
        except Exception as e:
            print(f"[!] Error processing tracking for frame: {e}")
    
    # Cleanup
    if video_writer is not None:
        try:
            video_writer.release()
            print(f"[✓] Video writer released: {OUTPUT_VIDEO_PATH}")
        except Exception as e:
            print(f"[!] Error releasing video writer: {e}")
    
    pubsub.close()
    print("[✓] Tracking worker stopped")

def main():
    try:
        tracking_worker()
    except KeyboardInterrupt:
        print("[*] Tracking worker interrupted by user")
    except Exception as e:
        print(f"[!] Unexpected error in tracking worker: {e}")

if __name__ == "__main__":
    main()