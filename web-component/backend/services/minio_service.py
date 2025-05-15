import io
from minio import Minio
from minio.error import S3Error
import json
from ..config import MinIO_settings
import time
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinioService:
    def __init__(self, max_retries=10, backoff_factor=2):
        self.client = None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.camera_config_bucket = MinIO_settings.camera_config_bucket
        self.violation_config_bucket = MinIO_settings.violation_config_bucket
        self.violation_video_bucket = MinIO_settings.violation_video_bucket
        self.violation_image_bucket = MinIO_settings.violation_image_bucket
        self._connect()

    def _connect(self):
        """Thử kết nối tới MinIO với cơ chế retry."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.client = Minio(
                    MinIO_settings.minio_endpoint,
                    access_key=MinIO_settings.minio_access_key,
                    secret_key=MinIO_settings.minio_secret_key,
                    secure=MinIO_settings.minio_secure
                )
                self._ensure_buckets_exist()
                logger.info("Successfully connected to MinIO")
                return
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries:
                    logger.error("Failed to connect to MinIO after %d attempts: %s", self.max_retries, str(exc))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("MinIO connection attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
            except Exception as e:
                logger.error("Unexpected error connecting to MinIO: %s", str(e))
                raise

    def _ensure_buckets_exist(self):
        """Đảm bảo các bucket tồn tại."""
        for bucket in [self.camera_config_bucket, self.violation_config_bucket, self.violation_video_bucket, self.violation_image_bucket]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info("Created bucket: %s", bucket)
            except S3Error as exc:
                logger.error("Error creating bucket %s: %s", bucket, str(exc))
                raise

    def create_camera_config(self, config_data: dict):
        attempt = 0
        while attempt < self.max_retries:
            try:
                cam_id = config_data['cam_id']
                violation_type = config_data.get('violation_type', 'default')
                file_path = f"{cam_id}_{violation_type}.json"
                json_data = json.dumps(config_data, indent=2)
                self.client.put_object(
                    self.violation_config_bucket,
                    file_path,
                    data=io.BytesIO(json_data.encode('utf-8')),
                    length=len(json_data.encode('utf-8')),
                    content_type="application/json"
                )
                return file_path
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries:
                    logger.error("Failed to create camera config after %d attempts: %s", self.max_retries, str(exc))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Create camera config attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()  # Thử kết nối lại

    def get_camera_config(self, cam_id: str, violation_type: str):
        attempt = 0
        while attempt < self.max_retries:
            try:
                file_path = f"{cam_id}_{violation_type}.json"
                response = self.client.get_object(self.violation_config_bucket, file_path)
                try:
                    data = response.read()
                    config_data = json.loads(data.decode('utf-8'))
                    return config_data
                finally:
                    response.close()
                    response.release_conn()
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries or exc.code == "NoSuchKey":
                    logger.error("Failed to get camera config: %s", str(exc))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Get camera config attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()

    def delete_camera_config(self, cam_id: str, violation_type: str):
        attempt = 0
        while attempt < self.max_retries:
            try:
                file_path = f"{cam_id}_{violation_type}.json"
                self.client.remove_object(self.violation_config_bucket, file_path)
                return
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries or exc.code == "NoSuchKey":
                    logger.error("Failed to delete camera config: %s", str(exc))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Delete camera config attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()

    def get_all_cameras(self):
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.client.get_object(self.camera_config_bucket, "cameras.json")
                try:
                    data = response.read()
                    cameras_data = json.loads(data.decode('utf-8'))
                    return cameras_data
                finally:
                    response.close()
                    response.release_conn()
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries or exc.code == "NoSuchKey":
                    logger.error("Failed to get all cameras: %s", str(exc))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Get all cameras attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()

    def get_video(self, camera_id: str, timestamp: str, violation_type: str):
        attempt = 0
        while attempt < self.max_retries:
            try:
                file_path = f"{violation_type}/{camera_id}/{timestamp}.mp4"
                response = self.client.get_object(self.violation_video_bucket, file_path)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries or exc.code in ["NoSuchKey", "NoSuchBucket"]:
                    logger.error("Failed to get video: %s", str(exc))
                    raise Exception(f"Error fetching video {file_path}: {str(exc)}")
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Get video attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()

    def get_image(self, camera_id: str, timestamp: str, violation_type: str):
        attempt = 0
        while attempt < self.max_retries:
            try:
                file_path = f"{violation_type}/{camera_id}/{timestamp}.jpg"
                response = self.client.get_object(self.violation_image_bucket, file_path)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()
            except S3Error as exc:
                attempt += 1
                if attempt == self.max_retries or exc.code in ["NoSuchKey", "NoSuchBucket"]:
                    logger.error("Failed to get image: %s", str(exc))
                    raise Exception(f"Error fetching image {file_path}: {str(exc)}")
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Get image attempt %d failed: %s. Retrying in %d seconds...", attempt, str(exc), sleep_time)
                time.sleep(sleep_time)
                self._connect()