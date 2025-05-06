from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import time

def wait_for_cassandra(host, port=9042, timeout=60):
    """Đợi cho Cassandra sẵn sàng"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            cluster = Cluster([host], port=port)
            session = cluster.connect()
            session.execute("SELECT now() FROM system.local")
            cluster.shutdown()
            return True
        except Exception:
            time.sleep(1)
    return False

def create_traffic_system_schema():
    # Cassandra Docker thường chạy ở localhost, port 9042
    cassandra_host = 'localhost'
    
    print(f"Đang kết nối với Cassandra tại {cassandra_host}:9042...")
    
    if not wait_for_cassandra(cassandra_host):
        print("Không thể kết nối với Cassandra sau thời gian chờ")
        return

    # Tạo kết nối (Cassandra Docker mặc định không có xác thực)
    cluster = Cluster([cassandra_host], port=9042)
    session = cluster.connect()

    try:
        # Tạo keyspace
        create_keyspace = """
        CREATE KEYSPACE IF NOT EXISTS traffic_system
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
        """
        session.execute(create_keyspace)
        print("Đã tạo keyspace: traffic_system")

        # Sử dụng keyspace
        session.set_keyspace('traffic_system')
        print("Đang sử dụng keyspace: traffic_system")

        # Tạo bảng camera_videos_bucketed
        create_camera_videos = """
        CREATE TABLE IF NOT EXISTS camera_videos_bucketed (
            camera_id text,
            time_bucket text,
            timestamp timestamp,
            video_id uuid,
            video_url text,
            inferences text,
            PRIMARY KEY ((camera_id, time_bucket), timestamp, video_id)
        ) WITH CLUSTERING ORDER BY (timestamp DESC);
        """
        session.execute(create_camera_videos)
        print("Đã tạo bảng: camera_videos_bucketed")

        # Tạo bảng violations
        create_violations = """
        CREATE TABLE IF NOT EXISTS violations (
            violation_type text,
            violation_date text,
            violation_time timestamp,
            violation_id uuid,
            license_plate text,
            camera_id text,
            processed_time timestamp,
            status text,
            video_evidence_url text,
            image_evidence_url text,
            PRIMARY KEY ((status, violation_date, violation_type), violation_time, violation_id)
        ) WITH CLUSTERING ORDER BY (violation_time DESC);
        """
        session.execute(create_violations)
        print("Đã tạo bảng: violations")

    except Exception as e:
        print(f"Lỗi xảy ra: {e}")
    finally:
        cluster.shutdown()
        print("Đã đóng kết nối")

if __name__ == "__main__":
    create_traffic_system_schema()