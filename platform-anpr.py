# =======================================================
# Ultralytics Platform Automatic Number Plate Recognition
# =======================================================

# Reads images from a local directory, sends each to the Ultralytics Platform inference API for license plate
# detection, crops and preprocesses the detected plate region, runs EasyOCR to extract the plate text,  displays
# the annotated image in an OpenCV window and write the processed image in output directory.

# Controls:
#   n  →  next image
#   q  →  quit and close window
# =======================================================

import requests
import cv2
import os
import easyocr


# -------------
# Configuration
# -------------
# Replace this from Ultralytics Platform Deploy -> Deployments -> code -> Python
url = "https://predict-69c7592f0bf0ca035a80-dproatj77a-no.a.run.app/predict"

# Replace this from Ultralytics Platform Deploy -> Deployments -> code -> Python
api_key = "ul_516d37fd6653ca4d49eef088918a7ec258c20b10"

# Path to testing images directory
images_dir = "images"


output_dir = "runs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Padding in cropped area for better OCR results
pad = 10


# -------------------------------
# Ultralytics Platform Python API
# -------------------------------
args = {"conf": 0.25, "iou": 0.7, "imgsz": 640}

def get_results_from_platform(img_bytes):
    """Method to get prediction results from Platform. Only CPU inference supported right now."""
    response = requests.post(url,
                             headers={"Authorization": f"Bearer {api_key}"},
                             data=args,
                             files={"file": ("frame.jpg", img_bytes, "image/jpeg")})
    return response


# -------------------
# Visualization utils
# -------------------
BOX_COLOR = (108, 27, 255)  # pink box
TEXT_COLOR = (255, 255, 255)  # white text
FONT_SCALE = 1.2
THICKNESS = 4
def draw_box(frame, x1 , y1 , x2 , y2, label):
    """Draw bounding box on an image"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
    (tw, th), baseline = cv2.getTextSize(label, 0, FONT_SCALE, THICKNESS)
    label_y1, label_y2 = max(y1 - th - 15, 0), max(y1, th + 15)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, label_y2), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 4, label_y2 - 5), 0, FONT_SCALE, TEXT_COLOR, THICKNESS)


# ------------------
# Initialize EasyOCR
# ------------------
reader = easyocr.Reader(['en'])


# -----------
# Image utils
# -----------
images_list = os.listdir(images_dir)

for idx, image in enumerate(images_list):
    frame = cv2.imread(os.path.join(images_dir, image))
    h, w = frame.shape[:2]

    # Send frame to model
    buffer = cv2.imencode(".jpg", frame)[1]
    response = get_results_from_platform(buffer.tobytes())

    if response.ok:
        results = response.json()["images"][0]

        # Extract results
        for pred in results["results"]:
            x1, y1, x2, y2 = pred["box"]["x1"], pred["box"]["y1"], pred["box"]["x2"], pred["box"]["y2"]

            confidence = pred["confidence"]
            class_name = pred["class"]

            # Calculate padded coordinates
            x1, y1, x2, y2 = max(int(x1) - pad, 0), max(int(y1) - pad, 0), min(int(x2) + pad, w), min(int(y2) + pad, h)

            # Preprocess for OCR
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR plate text
            ocr_results = reader.readtext(thresh)
            plate_text = " ".join([t for (_, t, c) in ocr_results if c > 0.3]).strip().upper() or "???"

            draw_box(frame, x1, y1, x2, y2, plate_text)
            print(f"Frame {idx+1:04d} | {plate_text}")

    cv2.imwrite(os.path.join(output_dir, image), frame)
    cv2.imshow("ANPR", frame)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
