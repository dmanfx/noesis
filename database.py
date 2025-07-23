"""
database.py
Handles database operations for the tracking system.
Provides thread-safe connection management and CRUD operations.
"""
import sqlite3
import os
import threading
import time
from datetime import datetime, timedelta
from contextlib import contextmanager

# Database file path
DB_FILENAME = "tracking.db"

# Thread-local storage for database connections
local_storage = threading.local()

def get_db():
    """Gets a database connection for the current thread or creates one."""
    if not hasattr(local_storage, 'db_conn'):
        local_storage.db_conn = sqlite3.connect(DB_FILENAME, check_same_thread=False)
        # Set WAL mode for better concurrency
        try:
            local_storage.db_conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as e:
            print(f"[Thread {threading.get_ident()}] Warning: Could not set WAL mode - {e}")
    return local_storage.db_conn

def initialize_database(db_path=DB_FILENAME):
    """Initializes the SQLite database schema if it doesn't exist."""
    print(f"Initializing database schema in '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tracks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        camera TEXT NOT NULL,
        last_seen REAL NOT NULL,
        current_zone TEXT,
        data TEXT NOT NULL
    );
    ''')
    
    # Create zone_transitions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS zone_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        track_id INTEGER NOT NULL,
        camera TEXT NOT NULL,
        from_zone TEXT,
        to_zone TEXT NOT NULL
    );
    ''')
    
    # Create heatmap_data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS heatmap_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        date TEXT NOT NULL
    );
    ''')
    
    # Add Indexes for Performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_lookup ON tracks (track_id, camera);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_last_seen ON tracks (last_seen);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON zone_transitions (timestamp);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_heatmap_date_coords ON heatmap_data (date, x, y);')

    conn.commit()
    conn.close()

class DatabaseManager:
    """Handles database operations with improved connection management."""
    
    def __init__(self, db_path=DB_FILENAME):
        self.db_path = db_path
        initialize_database(db_path)
    
    def store_track(self, track, track_data_json):
        """Store or update track data in the database."""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE tracks SET 
                last_seen = ?,
                current_zone = ?,
                data = ?
            WHERE track_id = ? AND camera = ?
            ''', (track.last_seen, track.current_zone, track_data_json, 
                  track.track_id, track.camera))
            
            if cursor.rowcount == 0:
                cursor.execute('''
                INSERT INTO tracks (track_id, camera, last_seen, current_zone, data)
                VALUES (?, ?, ?, ?, ?)
                ''', (track.track_id, track.camera, track.last_seen, 
                      track.current_zone, track_data_json))
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"[DB Error] Error storing track {track.camera}_{track.track_id}: {e}")
    
    def store_transition(self, transition):
        """Store zone transition event in the database."""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO zone_transitions (timestamp, track_id, camera, from_zone, to_zone)
            VALUES (?, ?, ?, ?, ?)
            ''', (transition.timestamp, transition.track_id, transition.camera, 
                  transition.from_zone, transition.to_zone))
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"[DB Error] Error storing transition: {e}")
    
    def update_heatmap(self, points_to_update):
        """Update heatmap data in the database."""
        try:
            if not points_to_update:
                return
                
            conn = get_db()
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            
            updates = 0
            inserts = 0
            for x, y in points_to_update:
                cursor.execute('''
                UPDATE heatmap_data SET count = count + 1
                WHERE x = ? AND y = ? AND date = ?
                ''', (x, y, today))
                
                if cursor.rowcount == 0:
                    cursor.execute('''
                    INSERT INTO heatmap_data (x, y, count, date)
                    VALUES (?, ?, 1, ?)
                    ''', (x, y, today))
                    inserts += 1
                else:
                    updates += 1
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"[DB Error] Error updating heatmap data: {e}")
    
    def get_heatmap_data(self, date_filter=None):
        """Get aggregated heatmap data for visualization, optionally filtered by date."""
        heatmap_data = []
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            query = ''
            params = []

            if date_filter == 'week':
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
                
                query = '''
                SELECT x, y, SUM(count) as total_count 
                FROM heatmap_data
                WHERE date BETWEEN ? AND ?
                GROUP BY x, y
                HAVING total_count > 0
                '''
                params = (start_date, end_date)
                
            elif date_filter == 'yesterday':
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                query = '''
                SELECT x, y, count FROM heatmap_data
                WHERE date = ? AND count > 0
                '''
                params = (yesterday,)
                
            else: 
                target_date = date_filter
                if not date_filter or date_filter == 'today':
                    target_date = datetime.now().strftime("%Y-%m-%d")

                query = '''
                SELECT x, y, count FROM heatmap_data
                WHERE date = ? AND count > 0
                '''
                params = (target_date,)
            
            cursor.execute(query, params)
            heatmap_data = [{'x': x, 'y': y, 'value': count} for x, y, count in cursor.fetchall()]
            
            return heatmap_data
        
        except sqlite3.Error as e:
            print(f"[DB Error] Error retrieving heatmap data (filter: {date_filter}): {e}")
            return []
    
    def get_recent_transitions(self, limit=50):
        """Get recent zone transitions from the database."""
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
            SELECT timestamp, track_id, camera, from_zone, to_zone 
            FROM zone_transitions
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            transitions = [
                {"timestamp": ts, "track_id": tid, "camera": cam, "from_zone": fz, "to_zone": tz}
                for ts, tid, cam, fz, tz in cursor.fetchall()
            ]
            return transitions
        except sqlite3.Error as e:
            print(f"[DB Error] Error retrieving recent transitions: {e}")
            return []
    
    def get_historical_data(self, start_time=None, end_time=None, limit=100):
        """Get historical tracking data from database."""
        if start_time is None:
            start_time = time.time() - 24*60*60  # Last 24 hours
        if end_time is None:
            end_time = time.time()
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT track_id, camera, last_seen, current_zone, data
            FROM tracks
            WHERE last_seen BETWEEN ? AND ?
            ORDER BY last_seen DESC
            LIMIT ?
            ''', (start_time, end_time, limit))
            
            tracks = []
            for row in cursor.fetchall():
                track_id, camera, last_seen, current_zone, data = row
                try:
                    import json
                    track_data = json.loads(data)
                    tracks.append({
                        "track_id": track_id,
                        "camera": camera,
                        "last_seen": last_seen,
                        "current_zone": current_zone,
                        "data": track_data
                    })
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            return tracks
        except Exception as e:
            print(f"Error retrieving historical data: {e}")
            return []
    
    def cleanup_old_data(self, days=7):
        """Remove track history and heatmap data older than specified days."""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # Cleanup Tracks
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            cursor.execute('DELETE FROM tracks WHERE last_seen < ?', (cutoff_time,))
            deleted_tracks = cursor.rowcount
            
            # Cleanup Heatmap Data
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            cursor.execute('DELETE FROM heatmap_data WHERE date < ?', (cutoff_date,))
            deleted_heatmap = cursor.rowcount
            
            conn.commit()
            
            return deleted_tracks, deleted_heatmap
        except sqlite3.Error as e:
            print(f"[DB Error] Error cleaning up old data: {e}")
            return 0, 0 