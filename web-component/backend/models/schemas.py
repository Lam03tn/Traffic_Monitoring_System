from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class Point(BaseModel):
    x: float
    y: float

class ROI(BaseModel):
    point1: Point
    point2: Point
    point3: Point
    point4: Point

class TrafficLightZone(BaseModel):
    point1: Point
    point2: Point
    point3: Point
    point4: Point

class LaneMarking(BaseModel):
    start_point: Point
    end_point: Point

class ViolationTypeConfig(BaseModel):
    roi: ROI
    traffic_light_zone: TrafficLightZone | None = None  # Tùy loại vi phạm
    lane_marking: LaneMarking | None = None             # Tùy loại vi phạm

class ViolationConfigCreate(BaseModel):
    cam_id: str
    violation_type: str
    violation_config: list[ViolationTypeConfig]

class ViolationConfigResponse(ViolationConfigCreate):
    created_at: datetime

# Add to schemas.py
class VideoInfo(BaseModel):
    camera_id: str
    timestamp: datetime
    video_id: UUID
    video_url: str
    inferences: dict  # or List[dict] depending on your data structure

class ViolationInfo(BaseModel):
    violation_id: UUID
    violation_type: str
    violation_date: str
    violation_time: datetime
    license_plate: str
    camera_id: str
    processed_time: datetime
    status: str
    video_evidence_url: str
    image_evidence_url: str

class UpdateViolationStatus(BaseModel):
    violation: ViolationInfo
    new_status: str