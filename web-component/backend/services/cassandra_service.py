from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from datetime import datetime, date
from typing import List
from ..config import Cassandra_settings  # You'll need to add Cassandra settings to config.py
import json

class CassandraService:
    def __init__(self):
        # Add CassandraSettings to your config.py similar to MinIOSettings
        # self.cluster = Cluster(['localhost'], port=9042)
        self.cluster = Cluster([Cassandra_settings.cassandra_host], port=Cassandra_settings.cassandra_port)
        self.session = self.cluster.connect(Cassandra_settings.cassandra_keyspace)
        # self.session = self.cluster.connect('traffic-system')
        self.session.row_factory = dict_factory

    def get_videos_by_date(self, date: str) -> List[dict]:
        # Assuming date format is 'YYYY-MM-DD'
        query = """
            SELECT * FROM camera_videos_bucketed 
            WHERE time_bucket = %s ALLOW FILTERING
        """
        time_bucket = date[:7]  # Extract 'YYYY-MM' from date
        result = self.session.execute(query, [time_bucket])
        return [dict(row) for row in result]

    def get_violations_by_date(self, date: str) -> List[dict]:
        query = """
            SELECT * FROM violations 
            WHERE violation_date = %s ALLOW FILTERING
        """
        result = self.session.execute(query, [date])
        return [dict(row) for row in result]

    def get_violations_by_status(self, status: str) -> List[dict]:
        query = """
            SELECT * FROM violations 
            WHERE status = %s ALLOW FILTERING
        """
        result = self.session.execute(query, [status])
        return [dict(row) for row in result]

    def close(self):
        self.cluster.shutdown()