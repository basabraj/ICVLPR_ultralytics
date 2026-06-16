"""
CSV Logger
==========
Appends every detection (valid and invalid) to detections.csv.
Called from anpr_server.py after each db_insert().

CSV columns (in order):
  id, detected_at, plate, confidence,
  state_code, state, rto_code, last_4_digits,
  office_location, jurisdiction_area, annotation,
  old_code, rto_district, zone,
  plate_verified, correct_plate, validation_error,
  gross_weight, net_weight, modbus_sent, webhook_sent
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path

CSV_FILE = "/home/pi/Desktop/ultralytics/detections.csv"

FIELDNAMES = [
    "id",
    "detected_at",
    "plate",
    "confidence",
    "state_code",
    "state",
    "rto_code",
    "last_4_digits",
    "office_location",
    "jurisdiction_area",
    "annotation",
    "old_code",
    "rto_district",
    "zone",
    "plate_verified",
    "correct_plate",
    "validation_error",
    "gross_weight",
    "net_weight",
    "modbus_sent",
    "webhook_sent",
]

_lock = threading.Lock()


def _ensure_header():
    """Write header row if the file does not exist or is empty."""
    path = Path(CSV_FILE)
    if not path.exists() or path.stat().st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def _extract_state_code(plate: str) -> str:
    """Return first 2 chars of cleaned plate (state code)."""
    clean = plate.replace(" ", "").upper()
    return clean[:2] if len(clean) >= 2 else ""


def _extract_last_4(plate: str) -> str:
    """Return last 4 chars of cleaned plate if they are all digits."""
    clean = plate.replace(" ", "").upper()
    last4 = clean[-4:] if len(clean) >= 4 else ""
    return last4 if last4.isdigit() else ""


def log_detection(det: dict):
    """
    Append one detection row to the CSV file.
    Safe to call from multiple threads.

    `det` is the same dict passed to db_insert():
        id, detected_at, plate, confidence,
        state, rto_code, office_location, jurisdiction_area,
        annotation, old_code, rto_district, zone,
        plate_verified, correct_plate, validation_error,
        gross_weight, net_weight, modbus_sent, webhook_sent
    """
    plate = det.get("plate", "")

    row = {
        "id":                det.get("id", ""),
        "detected_at":       det.get("detected_at", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        "plate":             plate,
        "confidence":        det.get("confidence", ""),
        "state_code":        _extract_state_code(plate),
        "state":             det.get("state", ""),
        "rto_code":          det.get("rto_code", ""),
        "last_4_digits":     _extract_last_4(plate),
        "office_location":   det.get("office_location",   "NIL"),
        "jurisdiction_area": det.get("jurisdiction_area", "NIL"),
        "annotation":        det.get("annotation",        "NIL"),
        "old_code":          det.get("old_code",          "NIL"),
        "rto_district":      det.get("rto_district",      "NIL"),
        "zone":              det.get("zone",              "NIL"),
        "plate_verified":    "YES" if det.get("plate_verified") else "NO",
        "correct_plate":     det.get("correct_plate", ""),
        "validation_error":  det.get("validation_error", ""),
        "gross_weight":      det.get("gross_weight", ""),
        "net_weight":        det.get("net_weight",   ""),
        "modbus_sent":       "YES" if det.get("modbus_sent")  else "NO",
        "webhook_sent":      "YES" if det.get("webhook_sent") else "NO",
    }

    with _lock:
        _ensure_header()
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)


def get_csv_path() -> str:
    return CSV_FILE
