# from pydantic import BaseSettings
from pydantic_settings import BaseSettings

class MinIOSettings(BaseSettings):
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    camera_config_bucket: str = "camera-configs"
    violation_config_bucket: str = "violation-configs"
    violation_video_bucket: str = "violation-videos"
    violation_image_bucket: str = "violation-images"
    minio_secure: bool = False

class CassandraSettings(BaseSettings):
    cassandra_host: str = "cassandra"
    cassandra_keyspace: str = "traffic_system"
    cassandra_port: int = 9042

Cassandra_settings = CassandraSettings()
MinIO_settings = MinIOSettings()