from fastapi import FastAPI
from .api.cam_violation_config import router as camera_config_router
from .api.video_violation_query import router as query_router  # Add this line
from .api.cam_config import router as config_router
from .api.websocket_stream_video import router as websocket_stream
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Camera Configuration API",
    description="API for managing camera violation configurations in MinIO",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "*"],  # Thay bằng origin của frontend
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức (bao gồm WebSocket)
    allow_headers=["*"],  # Cho phép tất cả header
)

app.include_router(camera_config_router)
app.include_router(query_router)
app.include_router(config_router)
app.include_router(websocket_stream)

@app.get("/")
async def root():
    return {"message": "Camera Configuration API"}