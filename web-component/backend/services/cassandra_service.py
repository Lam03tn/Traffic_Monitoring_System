from uuid import UUID
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from datetime import datetime, date
from typing import List
from ..config import Cassandra_settings, MinIO_settings
import json
import time
import logging
from cassandra.cluster import NoHostAvailable

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CassandraService:
    def __init__(self, max_retries=10, backoff_factor=2):
        self.cluster = None
        self.session = None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.minio_endpoint = MinIO_settings.minio_endpoint
        self.violation_config_bucket = MinIO_settings.violation_config_bucket
        self.violation_video_bucket = MinIO_settings.violation_video_bucket
        self.violation_image_bucket = MinIO_settings.violation_image_bucket
        self._connect()

    def _connect(self):
        """Thử kết nối tới Cassandra với cơ chế retry."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.cluster = Cluster([Cassandra_settings.cassandra_host], port=Cassandra_settings.cassandra_port)
                self.session = self.cluster.connect(Cassandra_settings.cassandra_keyspace)
                self.session.row_factory = dict_factory
                logger.info("Successfully connected to Cassandra")
                return
            except NoHostAvailable as e:
                attempt += 1
                if attempt == self.max_retries:
                    logger.error("Failed to connect to Cassandra after %d attempts: %s", self.max_retries, str(e))
                    raise
                sleep_time = self.backoff_factor ** attempt
                logger.warning("Cassandra connection attempt %d failed: %s. Retrying in %d seconds...", attempt, str(e), sleep_time)
                time.sleep(sleep_time)
            except Exception as e:
                logger.error("Unexpected error connecting to Cassandra: %s", str(e))
                raise

    def _reconnect(self):
        """Thử kết nối lại nếu session bị mất."""
        logger.info("Attempting to reconnect to Cassandra...")
        self.close()
        self._connect()

    def get_videos_by_date(self, date: str) -> List[dict]:
        try:
            query = """
                SELECT * FROM camera_videos_bucketed 
                WHERE time_bucket = %s ALLOW FILTERING
            """
            time_bucket = date[:7]  # Extract 'YYYY-MM' from date
            result = self.session.execute(query, [time_bucket])
            videos = [dict(row) for row in result]
            for video in videos:
                timestamp = video['timestamp'].strftime('%Y%m%d%H%M%S')
                video['video_url'] = f"{self.minio_endpoint}/{self.violation_video_bucket}/{video['camera_id']}/{timestamp}.mp4"
            return videos
        except Exception as e:
            logger.error("Error querying videos: %s", str(e))
            self._reconnect()
            raise

    def get_violations_by_date(self, date: str) -> List[dict]:
        try:
            query = """
                SELECT * FROM violations 
                WHERE violation_date = %s ALLOW FILTERING
            """
            result = self.session.execute(query, [date])
            violations = [dict(row) for row in result]
            for violation in violations:
                timestamp = violation['violation_time'].strftime('%Y%m%d_%H%M%S')
                violation_type = violation['violation_type']
                violation['video_evidence_url'] = f"{self.minio_endpoint}/{self.violation_video_bucket}/{violation_type}/{violation['camera_id']}/{timestamp}.mp4"
                violation['image_evidence_url'] = f"{self.minio_endpoint}/{self.violation_image_bucket}/{violation_type}/{violation['camera_id']}/{timestamp}.jpg"
            return violations
        except Exception as e:
            logger.error("Error querying violations by date: %s", str(e))
            self._reconnect()
            raise

    def get_violations_by_status(self, status: str) -> List[dict]:
        try:
            query = """
                SELECT * FROM violations 
                WHERE status = %s ALLOW FILTERING
            """
            result = self.session.execute(query, [status])
            violations = [dict(row) for row in result]
            for violation in violations:
                timestamp = violation['violation_time'].strftime('%Y%m%d_%H%M%S')
                violation_type = violation['violation_type']
                violation['video_evidence_url'] = f"{self.minio_endpoint}/{self.violation_video_bucket}/{violation_type}/{violation['camera_id']}/{timestamp}.mp4"
                violation['image_evidence_url'] = f"{self.minio_endpoint}/{self.violation_image_bucket}/{violation_type}/{violation['camera_id']}/{timestamp}.jpg"
            return violations
        except Exception as e:
            logger.error("Error querying violations by status: %s", str(e))
            self._reconnect()
            raise

    def update_violation_status(self, violation: dict, new_status: str) -> dict:
        try:

            # Extract and validate fields
            old_status = violation['status']
            violation_date = violation['violation_date']
            violation_type = violation['violation_type']
            violation_time = violation['violation_time']
            violation_id = violation['violation_id']  # Convert string to UUID
            # Delete the old record using the primary key
            delete_query = """
                DELETE FROM violations 
                WHERE status = %s AND violation_date = %s AND violation_type = %s 
                AND violation_time = %s AND violation_id = %s
            """
            self.session.execute(delete_query, [
                old_status,
                violation_date,
                violation_type,
                violation_time,
                violation_id
            ])
            logger.info("Deleted old violation record")

            # Insert new record with updated status
            insert_query = """
                INSERT INTO violations (
                    violation_type, violation_date, violation_time, violation_id,
                    license_plate, camera_id, processed_time, status,
                    video_evidence_url, image_evidence_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            processed_time = (violation['processed_time']
                             if violation['processed_time'] else None)
            updated_violation = {
                **violation,
                'violation_time': violation_time,
                'processed_time': processed_time,
                'status': new_status
            }

            self.session.execute(insert_query, [
                updated_violation['violation_type'],
                updated_violation['violation_date'],
                updated_violation['violation_time'],
                violation_id,  # Use UUID object
                updated_violation['license_plate'],
                updated_violation['camera_id'],
                datetime.now() if updated_violation['status'] != 'pending' else updated_violation['processed_time'],
                updated_violation['status'],
                updated_violation['video_evidence_url'],
                updated_violation['image_evidence_url']
            ])
            logger.info("Inserted new violation record")

            return updated_violation
        except Exception as e:
            logger.error("Error updating violation status: %s, violation=%s", str(e), violation)
            self._reconnect()
            raise

    def close(self):
        """Đóng kết nối Cassandra."""
        if self.cluster:
            try:
                self.cluster.shutdown()
                logger.info("Cassandra connection closed")
            except Exception as e:
                logger.error("Error closing Cassandra connection: %s", str(e))
            finally:
                self.cluster = None
                self.session = None