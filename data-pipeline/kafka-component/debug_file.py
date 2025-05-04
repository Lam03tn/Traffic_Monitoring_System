import cv2
import numpy as np
import json
import boto3
from botocore.client import Config
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import av
import io
import os
import uuid
from datetime import datetime

# Configuration variables
CAMERA_ID = "cam5"  # Replace with desired camera ID
TIME_BUCKET = "2025-05-04"  # Replace with desired time bucket (YYYY-MM-DD)
TIMESTAMP = datetime.strptime("2025-05-04 01:25:39.816000+0000", "%Y-%m-%d %H:%M:%S.%f%z")  # Parse timestamp
VIDEO_ID = uuid.UUID("4105c1c4-4390-4ae2-8057-d1fc4b1b69fd")  # Replace with desired video ID (UUID)
OUTPUT_DIR = "./debug_videos"  # Replace with desired output directory
FRAME_RATE_TARGET = 3

# MinIO settings
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "traffic-videos"

# Cassandra settings
CASSANDRA_HOSTS = ['localhost']
CASSANDRA_PORT = 9042
KEYSPACE = 'traffic_system'
TABLE = 'camera_videos_bucketed'

# Video processing settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Colors for different classes

# Initialize MinIO S3 client
s3 = boto3.client(
    's3',
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Initialize Cassandra connection
cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
session = cluster.connect(KEYSPACE)

def fetch_video_and_metadata(camera_id, time_bucket, timestamp, video_id):
    """Fetch video from MinIO and metadata from Cassandra."""
    try:
        # Fetch video from MinIO
        object_key = f"{camera_id}/{time_bucket}/{video_id}.mp4"
        response = s3.get_object(Bucket=MINIO_BUCKET, Key=object_key)
        video_bytes = response['Body'].read()

        # Fetch metadata from Cassandra
        query = SimpleStatement(f"""
            SELECT inferences FROM {TABLE}
            WHERE camera_id = %s AND time_bucket = %s AND timestamp = %s AND video_id = %s
        """)
        rows = session.execute(query, (camera_id, time_bucket, timestamp, video_id))
        if not rows:
            raise ValueError("No metadata found in Cassandra for the given video")
        
        inferences = json.loads(rows[0].inferences)
        return video_bytes, inferences
    except Exception as e:
        print(f"[!] Error fetching video/metadata: {e}")
        return None, None

def draw_bboxes(frame, detections, frame_width, frame_height, scale = 640):
    """Draw bounding boxes on the frame."""
    for detection in detections:
        box = detection['box']
        score = detection['score']
        class_id = detection['class_id']

        # Convert normalized coordinates to pixel coordinates
        x = int(box[0] * frame_width / scale)
        y = int(box[1] * frame_height / scale)
        w = int(box[2] * frame_width / scale)
        h = int(box[3] * frame_height / scale)

        # Calculate top-left and bottom-right corners
        x1, y1 = x, y
        x2, y2 = x + w , y + h 

        # Draw rectangle
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), FONT, FONT_SCALE, color, FONT_THICKNESS)
    
    return frame

def process_video(video_bytes, inferences, output_path):
    """Process video, draw bboxes, and save to output, accounting for 3 FPS inference rate."""
    try:
        # Open video from bytes
        container = av.open(io.BytesIO(video_bytes), 'r', format='mp4')
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or 30

        # Calculate frame interval for 3 FPS inference
        frame_interval = int(fps / FRAME_RATE_TARGET) if fps >= FRAME_RATE_TARGET else 1

        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            int(fps),
            (video_stream.width, video_stream.height)
        )

        # Create a lookup for inferences by frame index
        inference_dict = {inf['frame_index']: inf['detections'] for inf in inferences}
        frame_count = 0
        last_detections = []  # Store the most recent detections for non-inferred frames
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                # Convert frame to OpenCV format
                img = frame.to_ndarray(format='bgr24')

                if frame_count in inference_dict:
                    last_detections = inference_dict[frame_count]
                # Draw the most recent detections (if any) on the current frame
                if last_detections:
                    img = draw_bboxes(img, last_detections, img.shape[1], img.shape[0])

                # Write frame to output video
                output_writer.write(img)
                frame_count += 1

        container.close()
        output_writer.release()
        print(f"[✓] Saved debug video to {output_path}")
    except Exception as e:
        print(f"[!] Error processing video: {e}")

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch video and metadata
    video_bytes, inferences = fetch_video_and_metadata(CAMERA_ID, TIME_BUCKET, TIMESTAMP, VIDEO_ID)

    if video_bytes and inferences:
        # Generate output path
        output_path = os.path.join(
            OUTPUT_DIR,
            f"debug_{CAMERA_ID}_{TIME_BUCKET}_{VIDEO_ID}.mp4"
        )
        process_video(video_bytes, inferences, output_path)
    else:
        print("[!] Failed to retrieve video or metadata")

if __name__ == "__main__":
    main()