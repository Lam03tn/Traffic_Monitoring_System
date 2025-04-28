import json
import time
import uuid
import base64
from datetime import datetime
from kafka import KafkaConsumer
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import boto3
from botocore.client import Config
from collections import defaultdict
import io


# Kafka settings
TOPIC_META = 'video-cam1-meta'
TOPIC_VIDEO = 'video-cam1-raw'
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

# Bộ nhớ tạm lưu metadata và video
pending_segments = defaultdict(dict)

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

SEGMENT_TIMEOUT_SECONDS = 60
# Kết nối Cassandra
cluster = Cluster([CASSANDRA_HOSTS], port=CASSANDRA_PORT)
session = cluster.connect(KEYSPACE)

# Tạo KafkaConsumer cho cả hai topic
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

print("[✓] Unified Kafka Consumer started")
video_chunks = defaultdict(lambda: {
    'chunks': {},
    'expected_chunks': None,
    'metadata_received': False,
    'metadata': None,
    'last_update': time.time()
})

def try_assemble_segment(segment_id):
    data = video_chunks[segment_id]
    chunks = data['chunks']
    expected = data['expected_chunks']

    if expected is not None and len(chunks) == expected and data['metadata_received']:
        # Ghép lại video bytes
        ordered_data = b''.join(chunks[i] for i in sorted(chunks))
        process_complete_segment(segment_id, ordered_data, data['metadata'])
        del video_chunks[segment_id]

def process_complete_segment(segment_id, video_bytes, metadata):
    try:
        timestamp = datetime.fromtimestamp(metadata['timestamp']) \
            if isinstance(metadata['timestamp'], (int, float)) \
            else datetime.fromisoformat(metadata['timestamp'])

        camera_id = metadata['camera_id']
        segment_index = metadata['segment_index']
        # duration = metadata['duration']

        debug_filename = f"debug_{segment_id}.mp4"

        # with open(debug_filename, 'wb') as f:
        #     f.write(video_bytes)
        # print(f"[✓] Debug video saved to {debug_filename}")

        # Upload MinIO
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
        inferences = ''  # hoặc json.dumps(metadata.get('inferences', ''))

        # Ghi vào Cassandra
        query = SimpleStatement(f"""
            INSERT INTO {TABLE} (camera_id, time_bucket, timestamp, video_id, video_url, inferences)
            VALUES (%s, %s, %s, %s, %s, %s)
        """)
        session.execute(query, (
            camera_id, time_bucket, timestamp, video_id, video_url, inferences
        ))

        print(f"[✓] Saved segment {segment_id}")
    except Exception as e:
        print(f"[!] Failed to save segment {segment_id}: {e}")

# Vòng lặp
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

    # Xoá segment quá hạn
    expired = [
        sid for sid, data in video_chunks.items()
        if now - data['last_update'] > SEGMENT_TIMEOUT_SECONDS
    ]
    for sid in expired:
        print(f"[!] Timeout: Dropping incomplete segment {sid}")
        del video_chunks[sid]