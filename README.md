<img width="1600" height="806" alt="image" src="https://github.com/user-attachments/assets/becb71fd-507f-4454-8b51-e82235ed5cd1" />

<img width="1600" height="794" alt="WhatsApp Image 2026-06-09 at 2 53 07 PM" src="https://github.com/user-attachments/assets/f303d42e-2ee3-4fcd-9e5e-82de8f101e3c" />

<img width="1600" height="823" alt="WhatsApp Image 2026-06-09 at 3 38 12 PM" src="https://github.com/user-attachments/assets/a183d86e-b44d-4919-afa6-5ab25d3e5700" />

<img width="1600" height="819" alt="WhatsApp Image 2026-06-09 at 3 43 02 PM" src="https://github.com/user-attachments/assets/45f559f5-72b8-4b90-8b1f-db2158d4e370" />


#ICVLPR — Indian Commercial Vehicle Licence Plate Recognition

An Automatic Number Plate Recognition (ANPR) system built for Indian vehicles. It reads an RTSP camera stream, detects licence plates with YOLOv8, reads them with EasyOCR, validates them against the full Indian RTO database, and serves a live web dashboard.


## Features

- **Live RTSP stream** — connects to an IP camera over H.265/RTSP
- **YOLOv8 plate detection** — custom-trained model (`exp.pt`)
- **EasyOCR + fine-tuned recogniser** — improved accuracy on Indian plates
- **5-gate plate validation** — length → state code → RTO district → format → digit check
- **OCR auto-correction** — fixes common misreads (I↔1, O↔0, B↔8, missing leading zeros, IND-emblem bleed)
- **RTO lookup** — maps every detected plate to its RTO office, district, and zone
- **Suggestion engine** — learns from user corrections and fuzzy-matches future reads
- **FastAPI web dashboard** — live MJPEG stream + WebSocket detection feed
- **SQLite storage** — all detections persisted with full RTO metadata
- **CSV export** — one-click download of all detections

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Fuzzy matching | [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) |
| Web server | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Video capture | OpenCV (`cv2.VideoCapture` over RTSP) |
| Database | SQLite |
| Hardware | Raspberry Pi (Linux `aarch64`) |

---

## Project Structure

```
ICLPR_ultralytics/
├── anpr_server.py      # Main server — RTSP capture, YOLO inference, FastAPI
├── train.py            # Suggestion engine — learns from verified corrections
├── plate_validator.py  # Full validation pipeline + RTO lookup
├── csv_logger.py       # CSV export helper
├── models/
│   ├── exp.pt          # YOLOv8 plate-detection model
│   └── finetuned_model.pth  # Fine-tuned EasyOCR recogniser
├── validation/
│   └── RTO_CODE.xlsx   # Full Indian Commercial Vehicle RTO database
├── web/static/         # Dashboard HTML/CSS/JS
└── anpr.db             # SQLite database (detections + verified plates)
```



## How It Works

Three threads run in parallel:

1. **Capture thread** — reads the latest frame from the RTSP stream continuously
2. **Inference thread** — runs YOLOv8 on each frame, crops detected plates, pre-processes them (IND-emblem trim, double-line split, CLAHE), and passes them through EasyOCR
3. **MJPEG pump** — overlays bounding boxes onto frames and streams them to the dashboard at a fixed FPS

Every detection goes through the validation pipeline:
```
raw OCR text
  → clean (strip non-alphanumeric)
  → strip IND prefix / logo bleed
  → auto-correct (missing zeros, letter swaps, position-aware fixes)
  → 5-gate validation
  → RTO lookup
  → store in SQLite + broadcast via WebSocket
```

---

## Setup

### Requirements

- Python 3.10+
- Raspberry Pi (or any Linux machine with a camera/RTSP source)

### Install dependencies

```bash
cd /home/pi/Desktop/ICVLPR_ultralytics
python3 -m venv yoloenv
source yoloenv/bin/activate
pip install -r requirements.txt
```

### Configure

Edit `anpr_server.py` and set your camera URL:

```python
RTSP_URL = "rtsp://<your-camera-ip>:554/..."
```

### Run

```bash
cd /home/pi/Desktop/ICVLPR_ultralytics
source yoloenv/bin/activate
python3 anpr_server.py
```

Open the dashboard at `http://<your-pi-ip>:9000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live web dashboard |
| `GET` | `/api/stream` | MJPEG video stream |
| `WS` | `/ws/live` | WebSocket — real-time detection events |
| `GET` | `/api/stats` | Today's counts and last detected plate |
| `GET` | `/api/detections` | Detection history (latest 100) |
| `PATCH` | `/api/detections/{id}` | Verify / correct a plate |
| `GET` | `/api/suggest?plate=XX00XX0000` | Fuzzy-match suggestion for a plate |
| `GET` | `/api/export/csv` | Download all detections as CSV |
| `POST` | `/api/live` | Start / pause inference |

---

## Plate Validation Pipeline

Indian licence plates follow the format: **`SS DD LL NNNN`**

- `SS` — 2-letter state code (e.g. `KL`, `WB`, `MH`)
- `DD` — 2-digit RTO district code (e.g. `07`, `04`)
- `LL` — 1–3 series letters (e.g. `F`, `AB`, `BPR`)
- `NNNN` — 4-digit plate number

The validator runs 5 strict gates and auto-corrects common OCR mistakes before each gate:

1. Length must be 8–10 characters
2. State code must be a valid Indian state/UT
3. District code must exist in the RTO Excel database
4. Full format must match `SS + DD + LL + NNNN`
5. Last 4 characters must be digits

Bharat Series (BH) plates (`22BH1234AB`) are handled separately.

---

## License

MIT
