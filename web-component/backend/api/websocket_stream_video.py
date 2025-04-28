from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer
import asyncio
import json
import base64
import time
from collections import defaultdict
from typing import Dict, Set, List, Optional
from ..services.consumer_service import ConnectionManager

# Create FastAPI app
router = APIRouter(prefix="/camera-stream", tags=["camera-stream"])

# Create connection manager instance
manager = ConnectionManager()

@router.websocket("/ws/stream/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for streaming video from a specific camera"""
    await manager.connect(websocket, camera_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "switch_camera" and "camera_id" in message:
                    new_camera_id = message["camera_id"]
                    success = await manager.switch_camera(websocket, camera_id, new_camera_id)
                    if success:
                        await websocket.send_json({
                            "type": "camera_switched", 
                            "camera_id": new_camera_id,
                            "success": True
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to switch camera"
                        })
                        
            except json.JSONDecodeError:
                print(f"[!] Received invalid JSON: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
        await manager.disconnect(websocket)