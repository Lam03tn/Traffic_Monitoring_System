import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import json
import time
import uuid
from datetime import datetime
from kafka import KafkaConsumer
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from collections import defaultdict
import io
import av
import boto3
from botocore.client import Config
from enhanced_violation_detector import EnhancedViolationDetector
import json
import os

# Load JSON config file
CONFIG_FILE = os.environ.get("CONFIG_FILE", "/app/config.json")
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found")

CAMERA_ID = os.getenv('CAMERA_ID', 'cam1')  
TOPIC_META = f"video-{CAMERA_ID}-meta"
TOPIC_VIDEO = f"video-{CAMERA_ID}-raw"

KAFKA_BOOTSTRAP_SERVERS = config.get("kafka", {}).get("bootstrap_servers", "localhost:29092")
GROUP_ID = config.get("kafka", {}).get("group_id", "video_metadata_consumer_group")
CASSANDRA_HOSTS = config.get("cassandra", {}).get("hosts", "localhost")
CASSANDRA_PORT = config.get("cassandra", {}).get("port", 9042)
KEYSPACE = config.get("cassandra", {}).get("keyspace", "traffic_system")
TABLE = config.get("cassandra", {}).get("table", "camera_videos_bucketed")
MINIO_ENDPOINT = config.get("minio", {}).get("endpoint", "localhost:9000")
MINIO_ACCESS_KEY = config.get("minio", {}).get("access_key", "minioadmin")
MINIO_SECRET_KEY = config.get("minio", {}).get("secret_key", "minioadmin")
MINIO_BUCKET = config.get("minio", {}).get("bucket", "traffic-videos")
TRITON_URL = config.get("triton", {}).get("url", "localhost:8001")
MODEL_NAME = config.get("triton", {}).get("vehicle_detection_model", "vehicle_detection")
MODEL_VERSION = config.get("triton", {}).get("model_version", "1")
FRAME_RATE_TARGET = config.get("processing", {}).get("frame_rate_target", 3)
SEGMENT_TIMEOUT_SECONDS = config.get("processing", {}).get("segment_timeout_seconds", 60)
BATCH_SIZE = config.get("processing", {}).get("batch_size", 16)

# MinIO S3 client
s3 = boto3.client(
    's3',
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Create bucket if it doesn't exist
try:
    s3.head_bucket(Bucket=MINIO_BUCKET)
except:
    s3.create_bucket(Bucket=MINIO_BUCKET)

# Temporary storage for video chunks
video_chunks = defaultdict(lambda: {
    'chunks': {},
    'expected_chunks': None,
    'metadata_received': False,
    'metadata': None,
    'last_update': time.time()
})

# Cassandra connection
cluster = Cluster([CASSANDRA_HOSTS], port=CASSANDRA_PORT)
session = cluster.connect(KEYSPACE)

# Initialize Violation Detector
violation_detector = EnhancedViolationDetector(
    camera_id=CAMERA_ID,
    minio_endpoint=MINIO_ENDPOINT,
    cassandra_host=CASSANDRA_HOSTS
)

# KafkaConsumer
consumer = KafkaConsumer(
    TOPIC_META,
    TOPIC_VIDEO,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    group_id=GROUP_ID,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda v: v,
    key_deserializer=lambda k: k.decode('utf-8') if k else None
)

# Preprocess frame
def preprocess_frame(frame, input_size=(640, 640)):
    original_height, original_width = frame.shape[:2]
    # Resize frame
    frame_resized = cv2.resize(frame, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_normalized = frame_normalized.transpose((2, 0, 1))
    return frame, frame_resized, frame_normalized, (original_width, original_height)

# Postprocess inference results
def postprocess(model_output, original_size, score_threshold=0.20, nms_threshold=0.45):
    original_width, original_height = original_size
    input_size = 640  # Kích thước đầu vào của mô hình (640x640)

    # Tính tỷ lệ scale
    scale_x = original_width / input_size
    scale_y = original_height / input_size

    outputs = np.array([cv2.transpose(model_output[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= score_threshold:
            # Original box: [x_left, y_left, w, h]
            x = outputs[0][i][0]
            y = outputs[0][i][1]
            w = outputs[0][i][2]
            h = outputs[0][i][3]

            # Scale bbox về kích thước gốc
            box = [
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold, 0.5)

    num_detections = 0
    output_boxes = []
    output_scores = []
    output_classids = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        output_boxes.append(box)
        output_scores.append(scores[index])
        output_classids.append(class_ids[index])
        num_detections += 1

    num_detections = np.array(num_detections)
    return num_detections, output_boxes, output_scores, output_classids

# Perform inference with Triton for batch frames
def perform_inference(frames, triton_client, camera_id, segment_id, frame_indices):
    processed_frames = []
    original_frames = []
    resized_frames = []
    original_sizes = []
    
    for frame in frames:
        # Lấy frame gốc, frame đã resize, frame đã xử lý và kích thước gốc
        original_frame, frame_resized, frame_processed, original_size = preprocess_frame(frame)
        processed_frames.append(frame_processed)
        original_frames.append(original_frame)
        resized_frames.append(frame_resized)
        original_sizes.append(original_size)
    
    input_tensor = np.stack(processed_frames, axis=0)

    metadatas = [
        json.dumps({
            'camera_id': camera_id,
            'segment_id': segment_id,
            'frame_index': frame_index,
            'timestamp': time.time()
        }) for frame_index in frame_indices
    ]

    inputs = [grpcclient.InferInput("images", input_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_tensor)
    outputs = [grpcclient.InferRequestedOutput("output0")]

    results = triton_client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=inputs,
        outputs=outputs,
    )
    
    output_data = results.as_numpy("output0")
    
    batch_results = []
    for i in range(len(frames)):
        frame_output = output_data[i:i+1]
        # Truyền kích thước gốc vào postprocess
        result = postprocess(frame_output, original_sizes[i])
        batch_results.append((result, metadatas[i], original_frames[i]))

    return batch_results

# Process complete segment
def process_complete_segment(segment_id, video_bytes, metadata):
    try:
        timestamp = datetime.fromtimestamp(metadata['timestamp']) \
            if isinstance(metadata['timestamp'], (int, float)) \
            else datetime.fromisoformat(metadata['timestamp'])
        camera_id = metadata['camera_id']
        segment_index = metadata['segment_index']

        # Upload video to MinIO
        video_id = uuid.uuid4()
        time_bucket = timestamp.strftime("%Y-%m-%d")
        object_key = f"{camera_id}/{time_bucket}/{video_id}.mp4"

        s3.put_object(
            Bucket=MINIO_BUCKET,
            Key=object_key,
            Body=video_bytes,
            ContentType='video/mp4'
        )

        video_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_key}"

        # Open video for inference
        container = av.open(io.BytesIO(video_bytes), 'r', format='mp4')
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or 5
        frame_interval = int(fps / FRAME_RATE_TARGET) if fps >= FRAME_RATE_TARGET else 1

        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        inferences = []
        current_batch = []
        current_indices = []
        frame_count = 0 
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame_count % frame_interval == 0:
                    img = frame.to_ndarray(format='bgr24')
                    current_batch.append(img)
                    current_indices.append(time.time())
                    
                    if len(current_batch) == BATCH_SIZE:
                        batch_results = perform_inference(
                            current_batch, triton_client, camera_id, segment_id, current_indices
                        )
                        
                        for (result, returned_metadata, original_frame), frame_index in zip(batch_results, current_indices):
                            num_detections, boxes, scores, class_ids = result
                            frame_inferences = {
                                'frame_index': frame_index,
                                'detections': [
                                    {
                                        'box': [float(coord) for coord in boxes[j]],
                                        'score': float(scores[j]),
                                        'class_id': int(class_ids[j])
                                    } for j in range(int(num_detections))
                                ],
                                'metadata': json.loads(returned_metadata)
                            }
                            inferences.append(frame_inferences)
                            
                            # Gửi frame gốc cho violation_detector
                            violation_detector.process_detection(frame_inferences, original_frame)
                        
                        current_batch = []
                        current_indices = []
                
                frame_count += 1

        # Process remaining frames
        if current_batch:
            batch_results = perform_inference(
                current_batch, triton_client, camera_id, segment_id, current_indices
            )
            for (result, returned_metadata, original_frame), frame_index in zip(batch_results, current_indices):
                num_detections, boxes, scores, class_ids = result
                frame_inferences = {
                    'frame_index': frame_index,
                    'detections': [
                        {
                            'box': [float(coord) for coord in boxes[j]],
                            'score': float(scores[j]),
                            'class_id': int(class_ids[j])
                        } for j in range(int(num_detections))
                    ],
                    'metadata': json.loads(returned_metadata)
                }
                inferences.append(frame_inferences)
                
                # Gửi frame gốc cho violation_detector
                violation_detector.process_detection(frame_inferences, original_frame)

        container.close()

        # Store in Cassandra
        inferences_json = json.dumps(inferences)
        query = SimpleStatement(f"""
            INSERT INTO {TABLE} (camera_id, time_bucket, timestamp, video_id, video_url, inferences)
            VALUES (%s, %s, %s, %s, %s, %s)
            USING TTL 86400
        """)
        session.execute(query, (
            camera_id, time_bucket, timestamp, video_id, video_url, inferences_json
        ))

        print(f"[✓] Processed and saved segment {segment_id} to MinIO, Cassandra, and processed violations with {len(inferences)} frames inferred")
    except Exception as e:
        print(f"[!] Failed to process segment {segment_id}: {e}")

# Assemble segment
def try_assemble_segment(segment_id):
    data = video_chunks[segment_id]
    chunks = data['chunks']
    expected = data['expected_chunks']

    if expected is not None and len(chunks) == expected and data['metadata_received']:
        ordered_data = b''.join(chunks[i] for i in sorted(chunks))
        process_complete_segment(segment_id, ordered_data, data['metadata'])
        del video_chunks[segment_id]

# Main loop
def main():
    print("[✓] Unified Kafka Consumer with Triton Batch Inference, Cassandra, and Direct Violation Processing started")
    try:
        while True:
            raw_msgs = consumer.poll(timeout_ms=1000)
            now = time.time()

            for tp, messages in raw_msgs.items():
                topic = tp.topic
                for msg in messages:
                    segment_id = msg.key
                    try:
                        if topic == TOPIC_META:
                            metadata = json.loads(msg.value.decode('utf-8'))
                            video_chunks[segment_id]['metadata'] = metadata
                            video_chunks[segment_id]['metadata_received'] = True
                            video_chunks[segment_id]['last_update'] = now
                        elif topic == TOPIC_VIDEO:
                            header_raw, chunk_data = msg.value.split(b'||', 1)
                            header = json.loads(header_raw.decode('utf-8'))
                            idx = header['chunk_index']
                            video_chunks[segment_id]['chunks'][idx] = chunk_data
                            video_chunks[segment_id]['last_update'] = now
                            if header['is_last_chunk']:
                                video_chunks[segment_id]['expected_chunks'] = idx + 1
                        try_assemble_segment(segment_id)
                    except Exception as e:
                        print(f"[!] Error in topic {topic}, segment {segment_id}: {e}")

            # Clean up expired segments
            expired = [
                sid for sid, data in video_chunks.items()
                if now - data['last_update'] > SEGMENT_TIMEOUT_SECONDS
            ]
            for sid in expired:
                print(f"[!] Timeout: Dropping incomplete segment {sid}")
                del video_chunks[sid]
    except KeyboardInterrupt:
        print("Consumer interrupted")
    finally:
        violation_detector.close()
        cluster.shutdown()

if __name__ == "__main__":
    main()