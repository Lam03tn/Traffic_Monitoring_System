from fastapi import APIRouter, FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error
import json
from pydantic import BaseModel
from typing import List
from ..services.minio_service import MinioService

router = APIRouter(prefix="/camera-config", tags=["camera-config"])
minio_service = MinioService()

# Pydantic model for Camera
class Camera(BaseModel):
    id: str
    name: str
    location: str
    status: str

@router.get("/all-cameras", response_model=List[dict])
async def get_all_cameras():
    """
    Retrieve all cameras basic information from camera-config/cameras.json in MinIO
    """
    try:
        cameras_data = minio_service.get_all_cameras()
        return cameras_data
    except S3Error as e:
        if e.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Camera configuration file not found")
        raise HTTPException(status_code=500, detail="Error retrieving camera configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))