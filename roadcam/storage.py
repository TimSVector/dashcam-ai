"""SQLite adapters: parameterized queries and normalized exact plate matching."""
import sqlite3
from .models import KnownVehicle, DetectionEvent
from .plates import normalize_plate

SCHEMA = """
CREATE TABLE IF NOT EXISTS known_vehicles (
    id INTEGER PRIMARY KEY,
    plate TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS detection_events (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    detected_plate TEXT NOT NULL,
    ocr_confidence REAL NOT NULL,
    vehicle_type TEXT NOT NULL,
    known_vehicle_id INTEGER REFERENCES known_vehicles(id),
    source TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS events_plate_idx ON detection_events(detected_plate);
"""


class SQLiteVehicleDatabase:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.executescript(SCHEMA)
        except Exception:
            self.connection.close()
            raise

    def add(self, plate: str, description: str, notes: str = "") -> KnownVehicle:
        plate = normalize_plate(plate)
        if not plate:
            raise ValueError("Plate must contain letters or digits")
        if not description.strip():
            raise ValueError("Vehicle description cannot be empty")
        with self.connection:
            self.connection.execute(
                "INSERT INTO known_vehicles (plate, description, notes) VALUES (?, ?, ?) "
                "ON CONFLICT(plate) DO UPDATE SET description=excluded.description, notes=excluded.notes",
                (plate, description, notes),
            )
        return self.lookup(plate)

    def lookup(self, plate: str) -> KnownVehicle | None:
        row = self.connection.execute(
            "SELECT id, plate, description, notes FROM known_vehicles WHERE plate = ?",
            (normalize_plate(plate),),
        ).fetchone()
        return KnownVehicle(*row) if row else None

    def list(self) -> list[KnownVehicle]:
        return [KnownVehicle(*row) for row in self.connection.execute(
            "SELECT id, plate, description, notes FROM known_vehicles ORDER BY plate"
        )]

    def close(self):
        self.connection.close()


class SQLiteEventLogger:
    def __init__(self, path):
        # Reuse schema setup, but own a separate connection and lifetime.
        self.database = SQLiteVehicleDatabase(path)

    def log(self, event: DetectionEvent):
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO detection_events "
                "(timestamp, detected_plate, ocr_confidence, vehicle_type, known_vehicle_id, source, frame_index) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.timestamp, normalize_plate(event.detected_plate), event.ocr_confidence,
                 event.vehicle_type, event.known_vehicle_id, event.source, event.frame_index),
            )

    def close(self):
        self.database.close()
