import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime, date
from typing import List
from ..models.schemas import VideoInfo, ViolationInfo, UpdateViolationStatus
from ..services.cassandra_service import CassandraService
from ..services.minio_service import MinioService
from pydantic import BaseModel

router = APIRouter(prefix="/query", tags=["queries"])
cassandra_service = CassandraService()
minio_service = MinioService()

@router.get("/videos/by-date/{query_date}", response_model=List[VideoInfo])
async def get_videos_by_date(query_date: str):
    try:
        videos = cassandra_service.get_videos_by_date(query_date)
        return videos
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/violations/by-date/{query_date}", response_model=List[ViolationInfo])
async def get_violations_by_date(query_date: str):
    try:
        violations = cassandra_service.get_violations_by_date(query_date)
        return violations
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/violations/by-status/{status}", response_model=List[ViolationInfo])
async def get_violations_by_status(status: str):
    try:
        violations = cassandra_service.get_violations_by_status(status)
        return violations
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/evidence/video/{violation_type}/{camera_id}/{timestamp}")
async def get_violation_video(violation_type: str, camera_id: str, timestamp: str):
    try:
        video_data = minio_service.get_video(camera_id, timestamp, violation_type)
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={"Content-Disposition": f"inline; filename={timestamp}.mp4"}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch video: {str(exc)}")

@router.get("/evidence/image/{violation_type}/{camera_id}/{timestamp}")
async def get_violation_image(violation_type: str, camera_id: str, timestamp: str):
    try:
        image_data = minio_service.get_image(camera_id, timestamp, violation_type)
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={timestamp}.jpg"}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(exc)}")

@router.post("/violations/update-status")
async def update_violation_status(update: UpdateViolationStatus):
    try:
        updated_violation = cassandra_service.update_violation_status(update.violation.model_dump(), update.new_status)
        return updated_violation
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update violation status: {str(exc)}")