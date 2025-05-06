# from pydantic import BaseSettings
from pydantic_settings import BaseSettings

class MinIOSettings(BaseSettings):
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "violation-configs"
    minio_secure: bool = False
    
class CassandraSettings(BaseSettings):
    cassandra_host: str = "localhost"
    cassandra_keyspace: str = "traffic_system"
    cassandra_port: int = 9042

Cassandra_settings = CassandraSettings()
MinIO_settings = MinIOSettings()