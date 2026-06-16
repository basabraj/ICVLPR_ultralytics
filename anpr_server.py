"""
ANPR Weighbridge Server
RTSP (H.265) → YOLO (exp.pt) → EasyOCR → FastAPI dashboard

Three independent threads:
  1. capture    — reads RTSP frames as fast as possible, stores latest raw frame
  2. inference  — reads latest raw frame, runs YOLO + EasyOCR, caches detected boxes
  3. mjpeg pump — reads raw frame + overlays cached boxes at fixed FPS (smooth video)
"""

import asyncio
import json
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import easyocr, torch
from ultralytics import YOLO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

from train import engine as suggestion_engine
from plate_validator import clean_plate, validate as validate_plate, get_rto_info
from csv_logger import log_detection, get_csv_path

# ✅ FIX 1: import এখন file-এর top-এ
from rapidfuzz.distance import Levenshtein

# ── Config ─────────────────────────────────────────────────────────────────────
RTSP_URL   = "rtsp://192.168.96.31:554/rtsp/streaming?channel=01&subtype=A"
MODEL_PATH           = "models/exp.pt"
FINETUNED_MODEL_PATH = "models/finetuned_model.pth"
MJPEG_FPS            = 15
YOLO_CONF            = 0.15
OCR_CONF             = 0.30
COOLDOWN_S           = 30
PAD                  = 20
DB_PATH              = "anpr.db"

BOX_COLOR      = (108, 27, 255)
TXT_COLOR      = (255, 255, 255)
MJPEG_INTERVAL = 1.0 / MJPEG_FPS

# ✅ FIX 2: PLATE_RE এবং _score module level-এ — loop-এর বাইরে
PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

def _score(txt: str) -> int:
    """
    Indian plate format score:
      3 → valid format (SS DD LL NNNN) — best
      2 → valid length (8-10 chars)
      1 → partial read (6-7 chars)
      0 → too short or too long
    """
    t = txt.replace(" ", "")
    if PLATE_RE.match(t):
        return 3
    if 8 <= len(t) <= 10:
        return 2
    if 6 <= len(t) < 8:
        return 1
    return 0


# ── Shared state ───────────────────────────────────────────────────────────────
_raw_lock    = threading.Lock()
_raw_frame   = None

_boxes_lock  = threading.Lock()
_last_boxes: list = []

_mjpeg_clients: list[asyncio.Queue] = []
_ws_clients: set[WebSocket]         = set()
_cooldown: dict[str, float]         = {}
_event_loop: asyncio.AbstractEventLoop | None = None
_inference_active                   = False


# ── Database ───────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            plate            TEXT    NOT NULL,
            confidence       REAL,
            gross_weight     REAL,
            net_weight       REAL,
            modbus_sent      INTEGER DEFAULT 0,
            webhook_sent     INTEGER DEFAULT 0,
            detected_at      TEXT    NOT NULL,
            plate_verified   INTEGER DEFAULT 1,
            correct_plate    TEXT,
            state            TEXT,
            validation_error TEXT    DEFAULT '',
            rto_code         TEXT    DEFAULT '',
            office_location  TEXT    DEFAULT 'NIL',
            jurisdiction_area TEXT   DEFAULT 'NIL',
            annotation       TEXT    DEFAULT 'NIL',
            old_code         TEXT    DEFAULT 'NIL',
            rto_district     TEXT    DEFAULT 'NIL',
            zone             TEXT    DEFAULT 'NIL'
        )
    """)
    for col, defn in [("plate_verified",    "INTEGER DEFAULT 1"),
                      ("correct_plate",     "TEXT"),
                      ("state",             "TEXT"),
                      ("validation_error",  "TEXT DEFAULT ''"),
                      ("rto_code",          "TEXT DEFAULT ''"),
                      ("office_location",   "TEXT DEFAULT 'NIL'"),
                      ("jurisdiction_area", "TEXT DEFAULT 'NIL'"),
                      ("annotation",        "TEXT DEFAULT 'NIL'"),
                      ("old_code",          "TEXT DEFAULT 'NIL'"),
                      ("rto_district",      "TEXT DEFAULT 'NIL'"),
                      ("zone",              "TEXT DEFAULT 'NIL'")]:
        try:
            con.execute(f"ALTER TABLE detections ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.close()


def db_insert(det: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("""
        INSERT INTO detections
          (plate, confidence, gross_weight, net_weight,
           modbus_sent, webhook_sent, detected_at,
           plate_verified, correct_plate, state, validation_error,
           rto_code, office_location, jurisdiction_area,
           annotation, old_code, rto_district, zone)
        VALUES
          (:plate, :confidence, :gross_weight, :net_weight,
           :modbus_sent, :webhook_sent, :detected_at,
           :plate_verified, :correct_plate, :state, :validation_error,
           :rto_code, :office_location, :jurisdiction_area,
           :annotation, :old_code, :rto_district, :zone)
    """, det)
    det["id"] = cur.lastrowid
    con.commit()
    con.close()


def db_stats() -> dict:
    con   = sqlite3.connect(DB_PATH)
    today = date.today().isoformat()
    today_c  = con.execute(
        "SELECT COUNT(*) FROM detections WHERE detected_at LIKE ?",
        (today + '%',)).fetchone()[0]
    total_c  = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    unique_t = con.execute(
        "SELECT COUNT(DISTINCT plate) FROM detections WHERE detected_at LIKE ?",
        (today + '%',)).fetchone()[0]
    row = con.execute(
        "SELECT plate, detected_at FROM detections ORDER BY id DESC LIMIT 1"
    ).fetchone()
    con.close()
    return {
        "today_detections":    today_c,
        "total_detections":    total_c,
        "unique_plates_today": unique_t,
        "open_weigh_events":   0,
        "last_plate":          row[0] if row else None,
        "last_detected_at":    row[1] if row else None,
    }


def db_history(limit: int = 100) -> list:
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT id, plate, confidence, gross_weight, net_weight,
               modbus_sent, webhook_sent, detected_at,
               plate_verified, correct_plate, state, validation_error,
               rto_code, office_location, jurisdiction_area,
               annotation, old_code, rto_district, zone
        FROM detections ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [
        {
            "id":                r[0],
            "plate":             r[1],
            "confidence":        r[2],
            "gross_weight":      r[3],
            "net_weight":        r[4],
            "modbus_sent":       bool(r[5]),
            "webhook_sent":      bool(r[6]),
            "detected_at":       r[7],
            "plate_verified":    bool(r[8]),
            "correct_plate":     r[9]  or r[1],
            "state":             r[10] or "",
            "validation_error":  r[11] or "",
            "rto_code":          r[12] or "",
            "office_location":   r[13] or "NIL",
            "jurisdiction_area": r[14] or "NIL",
            "annotation":        r[15] or "NIL",
            "old_code":          r[16] or "NIL",
            "rto_district":      r[17] or "NIL",
            "zone":              r[18] or "NIL",
        }
        for r in rows
    ]


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    ly1 = max(y1 - th - 12, 0)
    ly2 = max(y1, th + 12)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 8, ly2), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 4, ly2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TXT_COLOR, 2)


# ── Async helpers ──────────────────────────────────────────────────────────────
async def _broadcast(msg: dict):
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


async def _push_frame(jpg: bytes):
    for q in list(_mjpeg_clients):
        try:
            q.put_nowait(jpg)
        except asyncio.QueueFull:
            pass


def _schedule(coro):
    if _event_loop:
        asyncio.run_coroutine_threadsafe(coro, _event_loop)


# ── Thread 1: RTSP capture ─────────────────────────────────────────────────────
def capture_thread():
    global _raw_frame

    def open_cap():
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    print("[CAP] Connecting to RTSP…")
    cap = open_cap()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CAP] RTSP lost — reconnecting in 2 s…")
            cap.release()
            time.sleep(2)
            cap = open_cap()
            continue

        with _raw_lock:
            _raw_frame = frame


# ── Plate crop post-processor ──────────────────────────────────────────────────
def _find_ind_trim(gray: np.ndarray) -> int:
    """
    Column projection — finds the right edge of the IND emblem.
    Scans the left 35% of the crop; returns the column index where
    the emblem ends (last near-zero-count column in that zone).
    Returns 0 if no emblem is detected (safe fallback = no trim).
    """
    h, w    = gray.shape
    limit   = int(w * 0.22)          # search only first 22% — emblem is ≤15% wide
    region  = gray[:, :limit]

    _, binary = cv2.threshold(region, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_counts = np.sum(binary == 255, axis=0).astype(float)
    col_smooth = np.convolve(col_counts, np.ones(5) / 5, mode='same')

    threshold = np.max(col_smooth) * 0.15
    gap_cols  = np.where(col_smooth < threshold)[0]

    if len(gap_cols) == 0:
        return 0

    last_gap = int(gap_cols[-1]) + 1
    if last_gap > int(w * 0.18):     # never trim more than 18%
        return 0
    return last_gap


def _find_split_row(gray: np.ndarray) -> int:
    """
    Horizontal projection — finds the gap row between two text lines.
    Returns -1 if no clear inter-line gap exists (not a double-line plate).

    Guard: both halves above and below the split must contain significant
    text density. This prevents false splits caused by background padding
    (PAD pixels at top/bottom have near-zero density and look like a gap).
    """
    h = gray.shape[0]
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_counts = np.sum(binary == 255, axis=1)

    margin    = int(h * 0.20)
    search    = row_counts[margin: h - margin]
    min_val   = float(np.min(search))
    max_val   = float(np.max(search))

    # Require a true gap: min row density < 15% of peak
    if max_val == 0 or (min_val / max_val) > 0.15:
        return -1

    split_row = int(np.argmin(search)) + margin
    if split_row < int(h * 0.25) or split_row > int(h * 0.75):
        return -1

    # Both halves must have real text (≥ 25% of global peak)
    # Single-line plates: one half will be near-zero (pure background)
    top_max    = float(np.max(row_counts[margin:split_row]))    if split_row > margin else 0
    bottom_max = float(np.max(row_counts[split_row:h - margin])) if h - margin > split_row else 0
    if top_max < max_val * 0.25 or bottom_max < max_val * 0.25:
        return -1

    return split_row


def preprocess_plate_crop(frame, px1, py1, px2, py2, yolo_conf=1.0):
    """
    Post-process YOLO box before OCR.

    1. Content-aware IND-emblem trim  — skipped for low-confidence detections
       (conf < 0.30) to avoid cutting into the state code on marginal crops.
    2. Double-line split  — only when aspect < 2.0 AND a real gap exists.

    Returns a single BGR crop ready for OCR.
    """
    crop = frame[py1:py2, px1:px2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. Content-aware IND emblem trim — skip for weak YOLO detections
    trim_x = _find_ind_trim(gray) if yolo_conf >= 0.30 else 0
    if trim_x > 0:
        crop = crop[:, trim_x:]
        gray = gray[:, trim_x:]
        w    = crop.shape[1]

    # 2. Double-line split — only if aspect < 2.0 AND a real gap exists
    aspect = w / h if h > 0 else 999
    if aspect < 2.0:
        split = _find_split_row(gray)
        if split > 0:
            line1 = crop[:split, :]
            line2 = crop[split:, :]
        else:
            split = -1
        if split > 0 and line1.shape[0] > 0 and line2.shape[0] > 0:
            th    = max(line1.shape[0], line2.shape[0])
            line1 = cv2.resize(line1, (line1.shape[1], th))
            line2 = cv2.resize(line2, (line2.shape[1], th))
            crop  = np.hstack([line1, line2])

    return crop


# ── Thread 2: YOLO + EasyOCR inference ────────────────────────────────────────
def load_finetuned_reader(model_path: str):
    reader = easyocr.Reader(['en'], gpu=False)
    try:
        state = torch.load(model_path, map_location='cpu')
        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        reader.recognizer.load_state_dict(state)
        reader.recognizer.eval()
        print(f"[OCR] Fine-tuned model loaded: {model_path}")
    except FileNotFoundError:
        print(f"[OCR] Fine-tuned model not found at {model_path} — using default EasyOCR")
    except Exception as e:
        print(f"[OCR] Failed to load fine-tuned model ({e}) — using default EasyOCR")
    return reader

reader = load_finetuned_reader(FINETUNED_MODEL_PATH)

def inference_thread():
    global reader
    print("[INF] Loading YOLO model…")
    model  = YOLO(MODEL_PATH)
    print("[INF] Ready.")

    # ✅ FIX 3: clahe একবার তৈরি করো — loop-এর বাইরে
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ✅ FIX 4: _read একবার define করো — loop-এর বাইরে
    def _read(img, min_conf=0.10):
        return [(t, c) for (_, t, c) in reader.readtext(img) if c > min_conf]

    while True:
        if not _inference_active:
            with _boxes_lock:
                _last_boxes.clear()
            time.sleep(0.1)
            continue

        with _raw_lock:
            frame = _raw_frame

        if frame is None:
            time.sleep(0.05)
            continue

        frame = frame.copy()
        h, w  = frame.shape[:2]
        results = model(frame, conf=YOLO_CONF, verbose=False)[0]
        new_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            px1 = max(int(x1) - PAD, 0)
            py1 = max(int(y1) - PAD, 0)
            px2 = min(int(x2) + PAD, w)
            py2 = min(int(y2) + PAD, h)

            crop = preprocess_plate_crop(frame, px1, py1, px2, py2, yolo_conf=conf)
            if crop.size == 0:
                continue

            # Resize 2× for better OCR resolution
            crop_up = cv2.resize(crop, None, fx=2, fy=2,
                                 interpolation=cv2.INTER_CUBIC)

            # Preprocessing variants
            gray       = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
            gray       = cv2.bilateralFilter(gray, 9, 15, 15)
            gray_clahe = clahe.apply(gray)

            _, th_otsu = cv2.threshold(gray_clahe, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th_inv  = cv2.threshold(gray_clahe, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            green_clahe = clahe.apply(crop_up[:, :, 1])
            _, th_green = cv2.threshold(green_clahe, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ✅ FIX 5: processed images আগে, raw color image সবার শেষে
            candidates = (th_otsu, th_inv, gray_clahe, th_green, crop_up)

            best_txt   = ""
            best_score = -1

            for candidate in candidates:
                res = _read(candidate)
                txt = " ".join(t for t, _ in res).strip().upper()
                s   = _score(txt)

                if s > best_score:
                    best_score = s
                    best_txt   = txt

                if best_score == 3:   # valid format পেয়েছি → থামো
                    break

            raw_text                                  = best_txt
            plate_text, valid, reason, state, was_corrected = validate_plate(raw_text)

            print(f"[DBG] raw='{raw_text}' → plate='{plate_text}' valid={valid} corrected={was_corrected} conf={round(conf*100,1)}%")

            low_conf   = (conf * 100) < 50.0
            needs_edit = was_corrected or low_conf

            box_label = plate_text if valid else "???"
            new_boxes.append((px1, py1, px2, py2, box_label))

            now      = time.time()
            cool_key = plate_text or raw_text

            if now - _cooldown.get(cool_key, 0) < COOLDOWN_S:
                continue

            fuzzy_match = any(
                Levenshtein.distance(cool_key, k) <= 2
                and now - t < COOLDOWN_S
                for k, t in list(_cooldown.items())
            )
            if fuzzy_match:
                continue

            _cooldown[cool_key] = now

            if not valid:
                # Invalid plates: log only — not stored in DB or dashboard
                print(f"[REJ] raw='{raw_text}'  reason='{reason}'  conf={round(conf*100,1)}%")
                continue

            rto = get_rto_info(plate_text)
            det = {
                "plate":             plate_text,
                "confidence":        round(conf * 100, 1),
                "gross_weight":      None,
                "net_weight":        None,
                "modbus_sent":       False,
                "webhook_sent":      False,
                "detected_at":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "plate_verified":    False if needs_edit else True,
                "correct_plate":     plate_text,
                "state":             state,
                "validation_error":  "",
                "rto_code":          rto["rto_code"],
                "office_location":   rto["office_location"],
                "jurisdiction_area": rto["jurisdiction_area"],
                "annotation":        rto["annotation"],
                "old_code":          rto["old_code"],
                "rto_district":      rto["district"],
                "zone":              rto["zone"],
            }
            db_insert(det)
            log_detection(det)
            print(f"[INF] {plate_text}  verified={not needs_edit}  rto={rto['rto_code']}  conf={det['confidence']}%")
            _schedule(_broadcast({"type": "detection", "data": det}))

        with _boxes_lock:
            _last_boxes[:] = new_boxes


# ── Thread 3: MJPEG pump ───────────────────────────────────────────────────────
def mjpeg_pump_thread():
    while True:
        t0 = time.monotonic()

        with _raw_lock:
            frame = _raw_frame

        if frame is not None:
            display = frame.copy()
            with _boxes_lock:
                boxes = list(_last_boxes)
            for (px1, py1, px2, py2, label) in boxes:
                draw_box(display, px1, py1, px2, py2, label)

            ok, jpg = cv2.imencode(".jpg", display,
                                   [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                _schedule(_push_frame(jpg.tobytes()))

        elapsed = time.monotonic() - t0
        gap = MJPEG_INTERVAL - elapsed
        if gap > 0:
            time.sleep(gap)


# ── FastAPI ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    init_db()
    _event_loop = asyncio.get_running_loop()
    suggestion_engine.bootstrap_from_detections()
    threading.Thread(target=capture_thread,    daemon=True).start()
    threading.Thread(target=inference_thread,  daemon=True).start()
    threading.Thread(target=mjpeg_pump_thread, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return Path("web/static/index.html").read_text()


@app.get("/api/stream")
async def mjpeg_stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    _mjpeg_clients.append(q)

    async def generate():
        try:
            while True:
                try:
                    jpg = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        finally:
            try:
                _mjpeg_clients.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)


@app.get("/api/live")
async def get_live():
    return {"active": _inference_active}


@app.post("/api/live")
async def set_live(payload: dict):
    global _inference_active
    _inference_active = bool(payload.get("active", False))
    print(f"[SRV] Inference {'STARTED' if _inference_active else 'PAUSED'}")
    return {"active": _inference_active}


@app.get("/api/stats")
async def api_stats():
    return db_stats()


@app.get("/api/detections")
async def api_detections(limit: int = 100):
    return db_history(limit)


@app.patch("/api/detections/{det_id}")
async def update_detection(det_id: int, payload: dict):
    plate_verified = payload.get("plate_verified")
    correct_plate  = payload.get("correct_plate", "").strip().upper() or None

    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT plate, gross_weight, net_weight FROM detections WHERE id = ?",
        (det_id,)
    ).fetchone()
    if not row:
        con.close()
        return {"error": "not found"}, 404

    ocr_plate, gross, net = row

    if plate_verified and not correct_plate:
        correct_plate = ocr_plate

    # Validate the corrected plate and update validation_error + RTO info
    new_validation_error = None
    rto = None
    if correct_plate:
        cp_text, cp_valid, cp_reason, cp_state, _ = validate_plate(correct_plate)
        rto = get_rto_info(cp_text)
        if cp_valid:
            new_validation_error = ""
            plate_verified       = True
        else:
            new_validation_error = cp_reason

    if rto and new_validation_error is not None:
        con.execute("""
            UPDATE detections
            SET plate_verified = ?, correct_plate = ?, validation_error = ?,
                state = ?, rto_code = ?, office_location = ?,
                jurisdiction_area = ?, annotation = ?, old_code = ?,
                rto_district = ?, zone = ?
            WHERE id = ?
        """, (bool(plate_verified), correct_plate, new_validation_error,
              cp_state, rto["rto_code"], rto["office_location"],
              rto["jurisdiction_area"], rto["annotation"], rto["old_code"],
              rto["district"], rto["zone"], det_id))
    elif rto:
        con.execute("""
            UPDATE detections
            SET plate_verified = ?, correct_plate = ?,
                state = ?, rto_code = ?, office_location = ?,
                jurisdiction_area = ?, annotation = ?, old_code = ?,
                rto_district = ?, zone = ?
            WHERE id = ?
        """, (bool(plate_verified), correct_plate,
              cp_state, rto["rto_code"], rto["office_location"],
              rto["jurisdiction_area"], rto["annotation"], rto["old_code"],
              rto["district"], rto["zone"], det_id))
    elif new_validation_error is not None:
        con.execute("""
            UPDATE detections
            SET plate_verified = ?, correct_plate = ?, validation_error = ?
            WHERE id = ?
        """, (bool(plate_verified), correct_plate, new_validation_error, det_id))
    else:
        con.execute("""
            UPDATE detections
            SET plate_verified = ?, correct_plate = ?
            WHERE id = ?
        """, (bool(plate_verified), correct_plate, det_id))

    con.commit()
    con.close()

    if correct_plate:
        suggestion_engine.learn(ocr_plate, correct_plate, gross, net)

    return {
        "id":               det_id,
        "plate_verified":   bool(plate_verified),
        "correct_plate":    correct_plate,
        "validation_error": new_validation_error if new_validation_error is not None else "",
        "state":            cp_state            if rto else None,
        "rto_code":         rto["rto_code"]         if rto else None,
        "office_location":  rto["office_location"]  if rto else None,
        "jurisdiction_area":rto["jurisdiction_area"] if rto else None,
        "annotation":       rto["annotation"]       if rto else None,
        "old_code":         rto["old_code"]         if rto else None,
        "rto_district":     rto["district"]         if rto else None,
        "zone":             rto["zone"]             if rto else None,
    }


@app.get("/api/suggest")
async def api_suggest(plate: str):
    result = suggestion_engine.suggest(plate)
    if not result:
        return {"suggestion": None}
    return {
        "suggestion": result.correct_plate,
        "score":      round(result.score, 1),
        "source":     result.source,
        "history":    result.history,
    }


@app.get("/api/export/csv")
async def export_csv():
    from fastapi.responses import FileResponse
    import os
    path = get_csv_path()
    if not os.path.exists(path):
        return Response(content="No data yet", status_code=404)
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"anpr_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
 
    