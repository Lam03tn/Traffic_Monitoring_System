from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

KAFKA_BOOTSTRAP_SERVERS = 'localhost:29092'
# KAFKA_BOOTSTRAP_SERVERS = 'kafka:9092'

admin_client = KafkaAdminClient(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    client_id='camera-topic-creator'
)

camera_configs = {
    "cam1": "rtsp://rtsp-server:8554/cam1",
    "cam2": "rtsp://rtsp-server:8554/cam2",
    "cam3": "rtsp://rtsp-server:8554/cam3"
}

topics = []
for cam_id in camera_configs.keys():
    topic_name = f"video-{cam_id}"
    topics.append(NewTopic(name=topic_name, num_partitions=1, replication_factor=1))

try:
    admin_client.create_topics(new_topics=topics, validate_only=False)
    print("Created topics:", [t.name for t in topics])
except TopicAlreadyExistsError as e:
    print("Some topics already exist.")
finally:
    admin_client.close()
