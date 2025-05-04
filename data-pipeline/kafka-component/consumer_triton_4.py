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

# Kafka settings
CAMERA_ID = 'cam4'
TOPIC_META = f'video-{CAMERA_ID}-meta'
TOPIC_VIDEO = f'video-{CAMERA_ID}-raw'
KAFKA_BOOTSTRAP_SERVERS = ['localhost:29092']
GROUP_ID = 'video_metadata_consumer_group'

# Cassandra settings
CASSANDRA_HOSTS = 'localhost'
CASSANDRA_PORT = 9042
KEYSPACE = 'traffic_system'
TABLE = 'camera_videos_bucketed'

# MinIO settings
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "traffic-videos"

# Triton settings
TRITON_URL = "localhost:8001"
MODEL_NAME = "vehicle_detection"
MODEL_VERSION = "1"

# Video processing settings
FRAME_RATE_TARGET = 3  # 3 frames per second for inference
SEGMENT_TIMEOUT_SECONDS = 60
BATCH_SIZE = 8  # Batch size for inference

# MinIO S3 client
s3 = boto3.client(
    's3',
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Tạo bucket nếu chưa tồn tại
try:
    s3.head_bucket(Bucket=MINIO_BUCKET)
except:
    s3.create_bucket(Bucket=MINIO_BUCKET)

# Bộ nhớ tạm lưu video chunks
video_chunks = defaultdict(lambda: {
    'chunks': {},
    'expected_chunks': None,
    'metadata_received': False,
    'metadata': None,
    'last_update': time.time()
})

# Kết nối Cassandra
cluster = Cluster([CASSANDRA_HOSTS], port=CASSANDRA_PORT)
session = cluster.connect(KEYSPACE)

# Tạo KafkaConsumer
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

# Hàm tiền xử lý frame
def preprocess_frame(frame, input_size=(640, 640)):
    frame_resized = cv2.resize(frame, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_normalized = frame_normalized.transpose((2, 0, 1))
    return frame_resized, frame_normalized

# Hàm hậu xử lý kết quả suy luận
def postprocess(model_output, score_threshold=0.35, nms_threshold=0.45):
    outputs = np.array([cv2.transpose(model_output[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= score_threshold:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3]
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

# Thực hiện inference với Triton cho batch frames
def perform_inference(frames, triton_client, camera_id, segment_id, frame_indices):
    # Preprocess all frames in the batch
    processed_frames = []
    for frame in frames:
        _, frame_processed = preprocess_frame(frame)
        processed_frames.append(frame_processed)
    
    # Stack frames into a batch tensor
    input_tensor = np.stack(processed_frames, axis=0)  # Shape: [batch_size, C, H, W]

    # Chuẩn bị metadata cho từng frame
    metadatas = [
        json.dumps({
            'camera_id': camera_id,
            'segment_id': segment_id,
            'frame_index': frame_index
        }) for frame_index in frame_indices
    ]

    # Chuẩn bị input cho Triton
    inputs = [
        grpcclient.InferInput("images", input_tensor.shape, "FP32"),
    ]
    
    inputs[0].set_data_from_numpy(input_tensor)
    
    # Chuẩn bị output
    outputs = [
        grpcclient.InferRequestedOutput("output0"),
    ]

    # Gửi yêu cầu suy luận
    results = triton_client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=inputs,
        outputs=outputs,
    )
    
    # Lấy kết quả
    output_data = results.as_numpy("output0")
    
    # Post-process results for each frame in the batch
    batch_results = []
    for i in range(len(frames)):
        # Extract output for the i-th frame
        frame_output = output_data[i:i+1]  # Shape: [1, ...]
        result = postprocess(frame_output)
        batch_results.append((result, metadatas[i]))

    return batch_results

# Xử lý segment hoàn chỉnh
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

        # Mở video từ bytes để inference
        container = av.open(io.BytesIO(video_bytes), 'r', format='mp4')
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or 30
        frame_interval = int(fps / FRAME_RATE_TARGET) if fps >= FRAME_RATE_TARGET else 1

        print("Frame rate: ", frame_interval)

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
                    current_indices.append(frame_count)
                    
                    # Khi đủ batch size thì gửi đi inference
                    if len(current_batch) == BATCH_SIZE:
                        batch_results = perform_inference(
                            current_batch, triton_client, camera_id, segment_id, current_indices
                        )
                        
                        for (result, returned_metadata), frame_index in zip(batch_results, current_indices):
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
                            }
                            inferences.append(frame_inferences)
                        
                        current_batch = []
                        current_indices = []
                        
                frame_count += 1

        # Xử lý các frame còn lại trong batch chưa đủ size
        if current_batch:
            batch_results = perform_inference(
                current_batch, triton_client, camera_id, segment_id, current_indices
            )
            for (result, returned_metadata), frame_index in zip(batch_results, current_indices):
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
                }
                inferences.append(frame_inferences)

        container.close()

        # Ghi vào Cassandra
        inferences_json = json.dumps(inferences)
        query = SimpleStatement(f"""
            INSERT INTO {TABLE} (camera_id, time_bucket, timestamp, video_id, video_url, inferences)
            VALUES (%s, %s, %s, %s, %s, %s)
            USING TTL 86400
        """)
        session.execute(query, (
            camera_id, time_bucket, timestamp, video_id, video_url, inferences_json
        ))

        print(f"[✓] Processed and saved segment {segment_id} to MinIO and Cassandra with {len(inferences)} frames inferred")
    except Exception as e:
        print(f"[!] Failed to process segment {segment_id}: {e}")

# Ghép segment
def try_assemble_segment(segment_id):
    data = video_chunks[segment_id]
    chunks = data['chunks']
    expected = data['expected_chunks']

    if expected is not None and len(chunks) == expected and data['metadata_received']:
        ordered_data = b''.join(chunks[i] for i in sorted(chunks))
        process_complete_segment(segment_id, ordered_data, data['metadata'])
        del video_chunks[segment_id]

# Vòng lặp chính
def main():
    print("[✓] Unified Kafka Consumer with Triton Batch Inference and MinIO started")
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

        # Xóa segment quá hạn
        expired = [
            sid for sid, data in video_chunks.items()
            if now - data['last_update'] > SEGMENT_TIMEOUT_SECONDS
        ]
        for sid in expired:
            print(f"[!] Timeout: Dropping incomplete segment {sid}")
            del video_chunks[sid]

if __name__ == "__main__":
    main()