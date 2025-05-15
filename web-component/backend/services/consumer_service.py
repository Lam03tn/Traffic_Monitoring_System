from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from kafka import KafkaConsumer, TopicPartition
import asyncio
import json
import base64
import time
from typing import Optional
from typing import Dict, Set, List, Optional

KAFKA_BOOTSTRAP_SERVERS = ['localhost:29092']
SEGMENT_TIMEOUT_SECONDS = 60

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}  # websocket: current_camera_id
        self.camera_consumers: Dict[str, KafkaConsumer] = {}  # camera_id: consumer
        self.consumer_tasks: Dict[str, asyncio.Task] = {}  # camera_id: task
        self.video_processors: Dict[str, VideoProcessor] = {}  # camera_id: processor

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        self.active_connections[websocket] = camera_id

        # Start consumer if not already running for this camera
        if camera_id not in self.consumer_tasks or self.consumer_tasks[camera_id].done():
            self.video_processors[camera_id] = VideoProcessor(camera_id, self)
            self.consumer_tasks[camera_id] = asyncio.create_task(
                self.run_camera_consumer(camera_id)
            )

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            camera_id = self.active_connections[websocket]
            del self.active_connections[websocket]
            
            try:
                await websocket.close()
            except Exception:
                pass
                        
            # Clean up if no more connections for this camera
            await self._cleanup_if_no_connections(camera_id)

    async def switch_camera(self, websocket: WebSocket, old_camera_id: str, new_camera_id: str) -> bool:
        """Switch a WebSocket connection to a different camera and clean up old camera resources"""
        if websocket not in self.active_connections:
            return False

        # Update the connection mapping
        self.active_connections[websocket] = new_camera_id

        # Explicitly clean up old camera resources
        if old_camera_id != new_camera_id:  # Only cleanup if switching to a different camera
            if old_camera_id in self.consumer_tasks:
                self.consumer_tasks[old_camera_id].cancel()
                try:
                    await self.consumer_tasks[old_camera_id]
                except (asyncio.CancelledError, Exception):
                    print(f"[i] Consumer task for camera {old_camera_id} was cancelled")
                del self.consumer_tasks[old_camera_id]

            if old_camera_id in self.camera_consumers:
                try:
                    self.camera_consumers[old_camera_id].close()
                except Exception as e:
                    print(f"[!] Error closing consumer: {e}")
                del self.camera_consumers[old_camera_id]

            if old_camera_id in self.video_processors:
                del self.video_processors[old_camera_id]

        # Start consumer for new camera if not already running
        if new_camera_id not in self.consumer_tasks or self.consumer_tasks[new_camera_id].done():
            self.video_processors[new_camera_id] = VideoProcessor(new_camera_id, self)
            self.consumer_tasks[new_camera_id] = asyncio.create_task(
                self.run_camera_consumer(new_camera_id)
            )

        return True

    async def _cleanup_if_no_connections(self, camera_id: str):
        """Clean up resources for a camera if no clients are connected to it"""
        if not any(cid == camera_id for cid in self.active_connections.values()):
            if camera_id in self.consumer_tasks:
                self.consumer_tasks[camera_id].cancel()
                try:
                    await self.consumer_tasks[camera_id]
                except (asyncio.CancelledError, Exception):
                    print(f"[i] Consumer task for camera {camera_id} was cancelled")
                del self.consumer_tasks[camera_id]
            
            if camera_id in self.camera_consumers:
                try:
                    self.camera_consumers[camera_id].close()
                except Exception as e:
                    print(f"[!] Error closing consumer: {e}")
                del self.camera_consumers[camera_id]
            
            if camera_id in self.video_processors:
                del self.video_processors[camera_id]

    async def send_frame(self, websocket: WebSocket, frame_data: bytes, timestamp: str, camera_id: str):
        if websocket not in self.active_connections or self.active_connections[websocket] != camera_id:
            return
            
        encoded_frame = base64.b64encode(frame_data).decode('utf-8')
        message = {
            "frame": encoded_frame,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "type": "video_frame"
        }

        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"[!] Error sending frame: {e}")
            await self.disconnect(websocket)

    async def run_camera_consumer(self, camera_id: str):
        topic_video = f'video-{camera_id}-raw'
        
        try:
            consumer = KafkaConsumer(
                topic_video,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                enable_auto_commit=False,  # Tắt auto-commit để tránh lưu offset cũ
                value_deserializer=lambda v: v,
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                consumer_timeout_ms=1000
            )
            self.camera_consumers[camera_id] = consumer

            # Reset offset to the latest
            partitions = consumer.partitions_for_topic(topic_video)
            if partitions:
                for partition in partitions:
                    tp = TopicPartition(topic_video, partition)
                    consumer.seek_to_end(tp)

            try:
                while camera_id in self.consumer_tasks:  # Keep running while task exists
                    # Check if there are any active connections for this camera
                    if not any(cid == camera_id for cid in self.active_connections.values()):
                        break
                    
                    raw_msgs = consumer.poll(timeout_ms=100)
                    
                    for tp, messages in raw_msgs.items():
                        for msg in messages:
                            segment_id = msg.key
                            await self.video_processors[camera_id].process_message(segment_id, msg.value)
                    
                    # Send frames to all connected clients for this camera
                    if camera_id in self.video_processors:
                        complete_segments = self.video_processors[camera_id].get_complete_segments()
                        for segment_id, frame_data in complete_segments.items():
                            timestamp = datetime.now().isoformat()
                            # Send to all websockets watching this camera
                            for ws, cid in list(self.active_connections.items()):
                                if cid == camera_id:
                                    await self.send_frame(ws, frame_data, timestamp, camera_id)
                    
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                print(f"[i] Consumer task for camera {camera_id} was cancelled")
            except Exception as e:
                print(f"[!] Error in consumer task for camera {camera_id}: {e}")
            finally:
                try:
                    consumer.close()
                    if camera_id in self.camera_consumers:
                        del self.camera_consumers[camera_id]
                except Exception:
                    pass

        except Exception as e:
            print(f"[!] Failed to create consumer for camera {camera_id}: {e}")

class VideoProcessor:
    def __init__(self, camera_id: str, connection_manager: ConnectionManager):
        self.camera_id = camera_id
        self.connection_manager = connection_manager
        self.video_chunks: dict[int, bytes] = {}  # Stores chunks of the latest segment
        self.complete_segments: dict[str, bytes] = {}  # Stores complete segments
        self.latest_segment_id: Optional[str] = None
        self.latest_timestamp: Optional[int] = None
        self.expected_chunks: Optional[int] = None
        self.last_update: float = time.time()

    def get_complete_segments(self) -> dict[str, bytes]:
        """Returns and clears the complete segments buffer"""
        # Only return segments if there are active connections for this camera
        if not any(cid == self.camera_id for cid in self.connection_manager.active_connections.values()):
            self.complete_segments.clear()
            return {}
        segments = self.complete_segments.copy()
        self.complete_segments.clear()
        return segments

    def _extract_timestamp(self, segment_id: str) -> int:
        """Extracts timestamp from segment_id (CAMERA_ID_timestamp)"""
        try:
            return int(segment_id.split('_')[1])
        except (IndexError, ValueError):
            return 0

    async def process_message(self, segment_id: str, value: bytes):
        # Skip processing if no active connections for this camera
        if not any(cid == self.camera_id for cid in self.connection_manager.active_connections.values()):
            return

        now = time.time()
        try:
            # Split header and chunk_data
            header_raw, chunk_data = value.split(b'||', 1)
            header = json.loads(header_raw.decode('utf-8'))

            # Extract timestamp from segment_id
            current_timestamp = self._extract_timestamp(segment_id)

            # Check if this is a newer segment
            if (self.latest_timestamp is None or 
                current_timestamp > self.latest_timestamp):
                # Reset if we encounter a newer segment
                self.video_chunks = {}
                self.expected_chunks = None
                self.latest_segment_id = segment_id
                self.latest_timestamp = current_timestamp
                self.last_update = now

            # Only process chunks for the latest segment
            if segment_id == self.latest_segment_id:
                chunk_index = header['chunk_index']
                self.video_chunks[chunk_index] = chunk_data

                # Update expected_chunks when receiving last chunk
                if header['is_last_chunk']:
                    self.expected_chunks = header['chunk_index'] + 1

                # If we've received all chunks, assemble and store
                if self.expected_chunks is not None and len(self.video_chunks) == self.expected_chunks:
                    await self.assemble_segment(segment_id, self.expected_chunks)

            self.last_update = now
        except Exception as e:
            print(f"[!] Error processing message for camera {self.camera_id}, segment {segment_id}: {e}")

    async def assemble_segment(self, segment_id: str, expected_chunks: int):
        try:
            # Concatenate chunks in order
            ordered_data = b''.join(self.video_chunks[i] for i in sorted(self.video_chunks))
            self.complete_segments[segment_id] = ordered_data  # Store complete segment
            # Reset after processing
            self.video_chunks = {}
            self.expected_chunks = None
            self.latest_segment_id = None
            self.latest_timestamp = None
        except Exception as e:
            print(f"[!] Failed to assemble segment {segment_id} for camera {self.camera_id}: {e}")

    def cleanup_expired_segments(self):
        """Clean up segments if they timeout"""
        now = time.time()
        if now - self.last_update > SEGMENT_TIMEOUT_SECONDS and self.video_chunks:
            self.video_chunks = {}
            self.expected_chunks = None
            self.latest_segment_id = None
            self.latest_timestamp = None
        self.last_update = now