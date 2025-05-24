from minio import Minio
from minio.error import S3Error
import json
import io

# Khởi tạo client MinIO
client = Minio(
    endpoint="minio:9000",  # Thay đổi nếu cần (vd: "minio-server:9000")
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # Đặt True nếu dùng HTTPS
)

# Tên bucket và đối tượng
bucket_name = "camera-configs"
object_name = "cameras.json"

# Dữ liệu camera
camera_data = [
    {
        "id": "cam1",
        "name": "Camera Lê Duẩn - Nguyễn Thái Học",
        "location": "Lê Duẩn - Nguyễn Thái Học",
        "status": "online"
    },
    {
        "id": "cam2",
        "name": "Camera Phố Huế - Trần Khát Chân",
        "location": "Phố Huế - Trần Khát Chân",
        "status": "online"
    },
    {
        "id": "cam3",
        "name": "Camera Láng Hạ - Thái Hà",
        "location": "Láng Hạ - Thái Hà",
        "status": "online"
    },

    {
        "id": "cam4",
        "name": "Camera Lý Thường Kiệt - Hàng Bài",
        "location": "Lý Thường Kiệt - Hàng Bài",
        "status": "online"
    },

    {
        "id": "cam5",
        "name": "Camera Cửa Nam - Điện Biên Phủ",
        "location": "Cửa Nam - Điện Biên Phủ",
        "status": "online"
    },

    {
        "id": "cam6",
        "name": "Camera Lý Thường Kiệt - Bà Triệu",
        "location": "Lý Thường Kiệt - Bà Triệu",
        "status": "online"
    },
]

# Chuyển dữ liệu thành chuỗi JSON
json_data = json.dumps(camera_data, ensure_ascii=False, indent=4).encode('utf-8')
json_buffer = io.BytesIO(json_data)

# Tạo bucket nếu chưa có
found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)
    print(f"✅ Đã tạo bucket: {bucket_name}")
else:
    print(f"📦 Bucket đã tồn tại: {bucket_name}")

# Upload file JSON
client.put_object(
    bucket_name=bucket_name,
    object_name=object_name,
    data=json_buffer,
    length=len(json_data),
    content_type="application/json"
)

print(f"✅ Đã upload {object_name} vào bucket {bucket_name}")
