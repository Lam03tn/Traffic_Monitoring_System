from fastapi import APIRouter, HTTPException
from datetime import datetime, date
from typing import List
from ..models.schemas import VideoInfo, ViolationInfo
from ..services.cassandra_service import CassandraService

router = APIRouter(prefix="/query", tags=["queries"])
cassandra_service = CassandraService()

@router.get("/videos/by-date/{query_date}", response_model=List[VideoInfo])
async def get_videos_by_date(query_date: str):
    try:
        videos = cassandra_service.get_videos_by_date(query_date)
        return videos
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cassandra_service.close()

@router.get("/violations/by-date/{query_date}", response_model=List[ViolationInfo])
async def get_violations_by_date(query_date: str):
    try:
        violations = cassandra_service.get_violations_by_date(query_date)
        return violations
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cassandra_service.close()

@router.get("/violations/by-status/{status}", response_model=List[ViolationInfo])
async def get_violations_by_status(status: str):
    try:
        violations = cassandra_service.get_violations_by_status(status)
        return violations
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cassandra_service.close()

# @router.put("/{violation_id}/status", response_model=Dict)
# async def update_violation_status(violation_id: str, status_update: StatusUpdate = Body(...)):
#     try:
#         # Validate status value
#         valid_statuses = ["pending", "processed", "false_positive", "verified"]
#         if status_update.status not in valid_statuses:
#             raise HTTPException(status_code=400, detail=f"Invalid status value. Must be one of: {', '.join(valid_statuses)}")
        
#         # Update in database
#         success = cassandra_service.update_violation_status(violation_id, status_update.status)
        
#         if not success:
#             raise HTTPException(status_code=404, detail=f"Violation with ID {violation_id} not found")
        
#         return {"message": "Status updated successfully", "violation_id": violation_id, "new_status": status_update.status}
#     except Exception as exc:
#         if isinstance(exc, HTTPException):
#             raise exc
#         raise HTTPException(status_code=500, detail=str(exc))
#     finally:
#         cassandra_service.close()