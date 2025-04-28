from fractions import Fraction
import io
import subprocess
import threading
import time
import json
import requests
import cv2
import numpy as np
import av
import ffmpeg
from kafka import KafkaProducer

# ========== CONFIG ==========
CAMERA_ID = "cam5"  # Path name in MediaMTX

MEDIAMTX_CONFIG = [
    {"server_name": "localhost", "api_port": 9997, "rtmp_port": 1936},
    {"server_name": "localhost", "api_port": 9998, "rtmp_port": 1937},
    {"server_name": "localhost", "api_port": 9999, "rtmp_port": 1938}
]

KAFKA_BOOTSTRAP_SERVERS = 'localhost:29092'
TOPIC_META = f"video-{CAMERA_ID}-meta"
TOPIC_VIDEO = f"video-{CAMERA_ID}-raw"
SEGMENT_DURATION = 10  # seconds
CHUNK_SIZE = 512 * 1024  # 512KB
DEBUG = False

# ========== FUNCTIONS ==========

def initialize_kafka_producers():
    producer_meta = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    )
    producer_video = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode('utf-8'),
        max_request_size=10485760  # 10MB
    )
    return producer_meta, producer_video

def find_rtsp_url(cam_id):
    for server in MEDIAMTX_CONFIG:
        try:
            # MediaMTX uses a different API endpoint
            api_url = f"http://{server['server_name']}:{server['api_port']}/v3/paths/list"
            response = requests.get(api_url, timeout=2)
            data = response.json()
            
            # Check if the camera path exists in MediaMTX
            for item in data.get("items", []):
                if item["name"] == f"stream/{cam_id}":
                    rtsp_url = f"rtmp://{server['server_name']}:{server['rtmp_port']}/stream/{cam_id}"
                    print(f"[✓] Found RTSP URL: {rtsp_url}")
                    return rtsp_url
        except Exception as e:
            print(f"[✗] Could not query {api_url}: {e}")
    return None

def get_video_resolution(url):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json', url
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    info = json.loads(result.stdout)
    w = info['streams'][0]['width']
    h = info['streams'][0]['height']
    print(f"[✓] Stream resolution: {w}x{h}")
    return w, h

def get_resolution_with_pyav(rtmp_url):
    container = av.open(rtmp_url)
    video_stream = next(s for s in container.streams if s.type == 'video')
    width = video_stream.codec_context.width
    height = video_stream.codec_context.height
    container.close()
    return width, height

def encode_frames_to_mp4(frames, width, height, fps):
    buffer = io.BytesIO()
    output = av.open(buffer, 'w', format='mp4')
    stream = output.add_stream('h264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17', 'preset': 'fast'}
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
        for packet in stream.encode(av_frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    return buffer.getvalue()

def encode_av_frames_to_mp4(av_frames, width, height, fps):
    buffer = io.BytesIO()
    output = av.open(buffer, 'w', format='mp4')
    stream = output.add_stream('h264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17', 'preset': 'fast'}

    for i, av_frame in enumerate(av_frames):
        # Đảm bảo format phù hợp với encoder
        if av_frame.format.name != 'yuv420p':
            av_frame = av_frame.reformat(width, height, format='yuv420p')
    
        for packet in stream.encode(av_frame):
            output.mux(packet)

    for packet in stream.encode():
        output.mux(packet)

    output.close()
    return buffer.getvalue()

def process_and_send_segment(frames, start_time, segment_index, width, height, producer_meta, producer_video):
    try:
        fps = len(frames) / SEGMENT_DURATION if len(frames) > 0 else 10
        # video_bytes = encode_av_frames_to_mp4(frames, width, height, fps)
        video_bytes = encode_frames_to_mp4(frames, width, height, fps)

        if DEBUG:
            with open(f"debug_segment_{segment_index}.mp4", "wb") as f:
                f.write(video_bytes)

        segment_id = f"{CAMERA_ID}_{int(start_time)}"
        metadata = {
            'segment_id': segment_id,
            'timestamp': start_time,
            'camera_id': CAMERA_ID,
            'segment_index': segment_index,
        }
        producer_meta.send(TOPIC_META, key=segment_id, value=metadata)

        num_chunks = (len(video_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(num_chunks):
            chunk = video_bytes[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
            header = json.dumps({
                'segment_id': segment_id,
                'chunk_index': i,
                'is_last_chunk': (i == num_chunks - 1)
            }).encode('utf-8') + b'||'
            producer_video.send(TOPIC_VIDEO, key=segment_id, value=header + chunk)

        producer_meta.flush()
        producer_video.flush()
        print(f"[✓] Segment {segment_id} sent with {len(frames)} frames")

    except Exception as e:
        print(f"[✗] Failed to send segment: {e}")

def process_video_stream(rtsp_url, producer_meta, producer_video, width, height):
    try:
        container = av.open(rtsp_url)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        frames = []
        segment_index = 0
        start_time = time.time()

        print("[✓] Connected to RTMP stream via PyAV")

        for packet in container.demux(stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format='bgr24')
                frames.append(img)

                if time.time() - start_time >= SEGMENT_DURATION:
                    if frames:
                        # start_cpu = time.process_time()
                        threading.Thread(
                            target=process_and_send_segment,
                            args=(frames, start_time, segment_index, width, height, producer_meta, producer_video)
                        ).start()

                        # end_cpu = time.process_time()
                        # print(f"CPU time: {end_cpu - start_cpu:.4f} giây")
                        segment_index += 1
                        frames = []
                        start_time = time.time()
    except Exception as e:
        print(f"[✗] Error in stream processing: {e}")
    finally:
        producer_meta.close()
        producer_video.close()

# ========== MAIN ==========
if __name__ == "__main__":
    while True:
        rtsp_url = find_rtsp_url(CAMERA_ID)
        if not rtsp_url:
            print("[…] Waiting for camera stream to become available...")
            time.sleep(3)
            continue
        try:
            # width, height = get_video_resolution(rtsp_url)
            width, height = get_resolution_with_pyav(rtsp_url)
            producer_meta, producer_video = initialize_kafka_producers()
            process_video_stream(rtsp_url, producer_meta, producer_video, width, height)
        except Exception as e:
            print(f"[✗] Top-level error: {e}")
            time.sleep(3)