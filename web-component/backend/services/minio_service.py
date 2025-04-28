import io
from minio import Minio
from minio.error import S3Error
import json
from ..config import MinIO_settings

class MinioService:
    def __init__(self):
        self.client = Minio(
            MinIO_settings.minio_endpoint,
            access_key=MinIO_settings.minio_access_key,
            secret_key=MinIO_settings.minio_secret_key,
            secure=MinIO_settings.minio_secure
        )
        self.bucket = MinIO_settings.minio_bucket
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as exc:
            print("Error creating bucket:", exc)

    def create_camera_config(self, config_data: dict):
        violation_type = config_data['violation_type']
        file_path = f"{violation_type}/violation-configs-camera/{config_data['cam_id']}.json"
        json_data = json.dumps(config_data, indent=2)
        
        try:
            self.client.put_object(
                self.bucket,
                file_path,
                data=io.BytesIO(json_data.encode('utf-8')),
                length=len(json_data.encode('utf-8')),
                content_type="application/json"
            )
            return file_path
        except S3Error as exc:
            raise exc

    def get_camera_config(self, cam_id: str, violation_type: str):
        file_path = f"{violation_type}/violation-configs-camera/{cam_id}.json"
        
        try:
            response = self.client.get_object(self.bucket, file_path)
            try:
                data = response.read()
                config_data = json.loads(data.decode('utf-8'))
                return config_data
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            raise exc
            
    def delete_camera_config(self, cam_id: str, violation_type: str):
        file_path = f"{violation_type}/violation-configs-camera/{cam_id}.json"
        
        try:
            self.client.remove_object(self.bucket, file_path)
        except S3Error as exc:
            raise exc
        
    def get_all_cameras(self):
        try:
            response = self.client.get_object("camera-configs", "cameras.json")
            try:
                data = response.read()
                cameras_data = json.loads(data.decode('utf-8'))
                return cameras_data
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            raise exc