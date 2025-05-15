from fastapi import APIRouter, HTTPException, Body
from datetime import datetime
from ..models.schemas import ViolationConfigCreate, ViolationConfigResponse
from ..services.minio_service import MinioService

router = APIRouter(prefix="/violation-camera-config", tags=["violation-camera-config"])
minio_service = MinioService()

@router.post("/", response_model=ViolationConfigResponse)
async def create_camera_config(config: ViolationConfigCreate):
    config_dict = config.model_dump()
    config_dict["created_at"] = datetime.now().isoformat()
    
    try:
        file_path = minio_service.create_camera_config(config_dict)
        return config_dict
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/{cam_id}/{violation_type}", response_model=ViolationConfigResponse)
async def get_camera_config(cam_id: str, violation_type: str):
    try:
        config_data = minio_service.get_camera_config(cam_id, violation_type)
        return config_data
    except Exception as exc:
        if getattr(exc, "code", None) == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Camera config not found")
        raise HTTPException(status_code=500, detail=str(exc))
    
@router.put("/{cam_id}", response_model=ViolationConfigResponse)
async def update_camera_config(cam_id: str, updated_config: ViolationConfigCreate = Body(...)):
    try:
        config_dict = updated_config.model_dump()
        config_dict["cam_id"] = cam_id
        config_dict["created_at"] = datetime.now().isoformat()
        
        file_path = minio_service.create_camera_config(config_dict)
        return config_dict
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
@router.delete("/{cam_id}/{violation_type}")
async def delete_camera_config(cam_id: str, violation_type: str):
    try:
        minio_service.delete_camera_config(cam_id, violation_type)
        return {"message": f"Camera config for '{cam_id}' has been deleted."}
    except Exception as exc:
        if getattr(exc, "code", None) == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Camera config not found")
        raise HTTPException(status_code=500, detail=str(exc))
