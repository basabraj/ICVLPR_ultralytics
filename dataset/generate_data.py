"""
Synthetic License Plate Dataset Generator  (v2.0)
==================================================
তিন ধরনের Indian license plate image তৈরি করে:

  Type 1 → Single-line standard plates  (34,000 images)
            e.g. "WB04F7728" — একটা strip-এ সব
  Type 2 → Double-line standard plates  (30,000 images)
            e.g. Line1="WB04" + Line2="F7728" — দুটো আলাদা strip
  Type 3 → BH-series plates             ( 6,000 images)
            e.g. "22BH2345AA" — একটা strip-এ সব

Background:
  Yellow (হলুদ)  → 70%  commercial / HSRP
  White  (সাদা)  → 30%  private / BH-series

Output structure:
  dataset/
    train/
      images/   → সব training images
      labels.txt → fname\tplate_text
    val/
      images/
      labels.txt

Author  : ANPR Project
Version : 2.0
"""

from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────
FONT_PATH   = "/home/pi/Desktop/ultralytics/font/CharlesWright-Bold.otf"
OUTPUT_DIR  = "dataset"
IMG_HEIGHT  = 64
FONT_SIZE   = 52

# Background colors + probability weights
# Yellow 70%, White 30%  (Black সরানো হয়েছে)
BG_COLORS   = [(255, 240, 0), (255, 255, 255)]
BG_WEIGHTS  = [0.70,          0.30         ]

TEXT_COLOR_MAP = {
    (255, 240, 0  ): (0,   0,   0  ),   # yellow → black text
    (255, 255, 255): (0,   0,   0  ),   # white  → black text
}

# BH-series: O and I বাদ (0 এবং 1 এর সাথে confusion এড়াতে)
BH_LETTERS = [c for c in string.ascii_uppercase if c not in ('O', 'I')]

STATE_CODES = [
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP",
    "JH","KA","KL","MP","MH","MN","ML","MZ","NL",
    "OD","PB","RJ","SK","TN","TG","TR","UP","UK","WB",
    "DL","CH","PY","LA",
]

# Generation counts
SINGLE_LINE_PLATES   = 1700    # × 20 variants = 34,000 images
DOUBLE_LINE_PLATES   = 750     # × 2 strips × 20 variants ≈ 30,000 images
BH_SERIES_PLATES     = 300     # × 20 variants = 6,000 images
IMAGES_PER_PLATE     = 20
VAL_RATIO            = 0.15


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Dataset generation configuration।

    Attributes:
        font_path          : Charles Wright .otf font path
        output_dir         : Output directory
        img_height         : Output image height (pixels)
        images_per_plate   : প্রতিটা plate-এর augmented variant সংখ্যা
        single_line_plates : Single-line plate string সংখ্যা
        double_line_plates : Double-line plate string সংখ্যা
        bh_series_plates   : BH-series plate string সংখ্যা
        val_ratio          : Validation split ratio
    """
    font_path          : str   = FONT_PATH
    output_dir         : str   = OUTPUT_DIR
    img_height         : int   = IMG_HEIGHT
    images_per_plate   : int   = IMAGES_PER_PLATE
    single_line_plates : int   = SINGLE_LINE_PLATES
    double_line_plates : int   = DOUBLE_LINE_PLATES
    bh_series_plates   : int   = BH_SERIES_PLATES
    val_ratio          : float = VAL_RATIO


# ── Plate String Generators ────────────────────────────────────────────────────

class StandardPlateGenerator:
    """Standard Indian plate string generate করে।

    Format: SS DD LL NNNN
      SS   = 2 state letters   (e.g. WB)
      DD   = 2 district digits (e.g. 04)
      LL   = 1-3 series letters (e.g. F, FG, FGH)
      NNNN = 4 number digits   (e.g. 7728)

    Usage:
        gen = StandardPlateGenerator()
        plates = gen.generate(1700)
    """

    def _one(self) -> str:
        state   = random.choice(STATE_CODES)
        dist    = f"{random.randint(1, 99):02d}"
        series  = ''.join(random.choices(
            string.ascii_uppercase,
            k=random.choice([1, 1, 1, 2, 2, 3])
        ))
        number  = f"{random.randint(1, 9999):04d}"
        return state + dist + series + number

    def generate(self, count: int) -> List[str]:
        """count টা unique plate string তৈরি করে।"""
        plates: set[str] = set()
        while len(plates) < count:
            plates.add(self._one())
        return list(plates)


class BHPlateGenerator:
    """BH-series plate string generate করে।

    Format: YY BH NNNN XX
      YY   = year (21-25)
      BH   = fixed
      NNNN = 4 digits (1-9999)
      XX   = 1-2 letters (O এবং I বাদ)

    Ref: Ministry of Road Transport, 26 August 2021

    Usage:
        gen = BHPlateGenerator()
        plates = gen.generate(300)
    """

    def _one(self) -> str:
        year    = str(random.randint(21, 25))
        number  = f"{random.randint(1, 9999):04d}"
        # 1 বা 2 letters (O, I বাদ)
        n_chars = random.choice([1, 1, 2])
        suffix  = ''.join(random.choices(BH_LETTERS, k=n_chars))
        return year + "BH" + number + suffix

    def generate(self, count: int) -> List[str]:
        """count টা unique BH plate string তৈরি করে।"""
        plates: set[str] = set()
        while len(plates) < count:
            plates.add(self._one())
        return list(plates)


# ── Augmentation Pipeline ──────────────────────────────────────────────────────

class AugmentationPipeline:
    """Real-world plate condition simulate করার augmentations।

    প্রতিটা augmentation real scenario represent করে:
        blur          → camera focus issue
        motion_blur   → moving vehicle
        noise         → night / sensor noise
        perspective   → camera angle
        brightness    → different lighting conditions
        rotation      → slight plate tilt
        jpeg          → video compression artifact
    """

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)

    def _motion_blur(self, img: np.ndarray) -> np.ndarray:
        size   = random.randint(3, 7)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = np.ones(size) / size
        return cv2.filter2D(img, -1, kernel)

    def _salt_pepper(self, img: np.ndarray) -> np.ndarray:
        out    = img.copy()
        amount = random.uniform(0.01, 0.04)
        total  = int(amount * img.size)
        for val in [255, 0]:
            coords = [np.random.randint(0, d, total) for d in img.shape[:2]]
            out[coords[0], coords[1]] = val
        return out

    def _perspective(self, img: np.ndarray) -> np.ndarray:
        h, w   = img.shape[:2]
        m      = int(min(h, w) * 0.08)
        src    = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst    = np.float32([
            [random.randint(0,m),     random.randint(0,m)],
            [w-random.randint(0,m),   random.randint(0,m)],
            [w-random.randint(0,m),   h-random.randint(0,m)],
            [random.randint(0,m),     h-random.randint(0,m)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h),
                                   borderValue=(200, 200, 200))

    def _brightness(self, img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.6, 1.4)
        beta  = random.randint(-40, 40)
        return np.clip(alpha * img.astype(np.float32) + beta,
                       0, 255).astype(np.uint8)

    def _rotation(self, img: np.ndarray) -> np.ndarray:
        angle = random.uniform(-5.0, 5.0)
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              borderValue=(200, 200, 200))

    def _jpeg(self, img: np.ndarray) -> np.ndarray:
        q       = random.randint(40, 75)
        _, enc  = cv2.imencode('.jpg', img,
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """সব augmentation randomly apply করে।

        Args:
            img: Input BGR image

        Returns:
            Augmented BGR image
        """
        if random.random() < 0.40: img = self._gaussian_blur(img)
        if random.random() < 0.30: img = self._motion_blur(img)
        if random.random() < 0.40: img = self._salt_pepper(img)
        if random.random() < 0.50: img = self._perspective(img)
        if random.random() < 0.60: img = self._brightness(img)
        if random.random() < 0.30: img = self._rotation(img)
        if random.random() < 0.30: img = self._jpeg(img)
        return img


# ── Image Renderers ────────────────────────────────────────────────────────────

class SingleLineRenderer:
    """Charles Wright font দিয়ে single-line plate text strip render করে।

    Output: fixed-height strip image (BGR, numpy array)

    Usage:
        r = SingleLineRenderer(font_path, img_height=64)
        img = r.render("WB04F7728")
    """

    def __init__(self, font_path: str, img_height: int = IMG_HEIGHT) -> None:
        self.h     = img_height
        self.fs    = int(img_height * 0.78)
        self.font  = ImageFont.truetype(font_path, size=self.fs)
        self.pad_x = 10
        self.pad_y = 4

    def _text_size(self, text: str) -> Tuple[int, int]:
        dummy = Image.new("RGB", (1, 1))
        bbox  = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=self.font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def render(self, text: str,
               bg_color: Optional[Tuple] = None) -> np.ndarray:
        """Text string → BGR image strip।

        Args:
            text     : Plate text (e.g. "WB04F7728")
            bg_color : Background color tuple — None হলে weighted random

        Returns:
            np.ndarray: BGR image (height = self.h)
        """
        if bg_color is None:
            bg_color = random.choices(BG_COLORS, weights=BG_WEIGHTS, k=1)[0]

        txt_color = TEXT_COLOR_MAP.get(bg_color, (0, 0, 0))
        tw, th    = self._text_size(text)
        img_w     = tw + 2 * self.pad_x
        img_h     = self.h

        img  = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)
        y    = (img_h - th) // 2 - self.pad_y
        draw.text((self.pad_x, y), text, font=self.font, fill=txt_color)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class DoubleLineRenderer:
    """Double-line plate render করে এবং দুটো আলাদা strip return করে।

    Real plate layout:
        ┌──────────────────┐
        │  IND   WB  04    │  ← line 1 strip: "WB04"
        │        F  7728   │  ← line 2 strip: "F7728"
        └──────────────────┘

    দুটো strip আলাদা training sample হিসেবে save হয়।
    Inference-এ: line1 + line2 join করে full plate পাওয়া যায়।

    Usage:
        r = DoubleLineRenderer(font_path)
        l1_img, l1_label, l2_img, l2_label = r.render("WB04F7728")
    """

    # Standard split: first 4 chars = state+district (line 1)
    LINE1_LEN = 4

    def __init__(self, font_path: str, img_height: int = IMG_HEIGHT) -> None:
        self.renderer = SingleLineRenderer(font_path, img_height)

    def _split(self, plate: str) -> Tuple[str, str]:
        """Plate string কে দুটো line-এ ভাগ করে।

        Args:
            plate: Full plate string (e.g. "WB04F7728")

        Returns:
            (line1, line2): ("WB04", "F7728")
        """
        return plate[:self.LINE1_LEN], plate[self.LINE1_LEN:]

    def render(self, plate: str,
               bg_color: Optional[Tuple] = None
               ) -> Tuple[np.ndarray, str, np.ndarray, str]:
        """Double-line plate কে দুটো strip-এ render করে।

        Args:
            plate    : Full plate string
            bg_color : Background color — None হলে weighted random

        Returns:
            (line1_img, line1_text, line2_img, line2_text)
        """
        if bg_color is None:
            bg_color = random.choices(BG_COLORS, weights=BG_WEIGHTS, k=1)[0]

        line1, line2 = self._split(plate)
        img1 = self.renderer.render(line1, bg_color=bg_color)
        img2 = self.renderer.render(line2, bg_color=bg_color)
        return img1, line1, img2, line2


# ── Dataset Builder ────────────────────────────────────────────────────────────

class SyntheticDatasetBuilder:
    """তিন ধরনের plate image দিয়ে complete EasyOCR training dataset তৈরি করে।

    Types:
        1. Single-line standard  → "WB04F7728" (one strip)
        2. Double-line standard  → "WB04" + "F7728" (two strips)
        3. BH-series             → "22BH2345AA" (one strip)

    Usage:
        builder = SyntheticDatasetBuilder(config)
        builder.build()
    """

    def __init__(self, config: DatasetConfig = DatasetConfig()) -> None:
        self.cfg         = config
        self.single_rnd  = SingleLineRenderer(config.font_path, config.img_height)
        self.double_rnd  = DoubleLineRenderer(config.font_path, config.img_height)
        self.augmenter   = AugmentationPipeline()
        self.std_gen     = StandardPlateGenerator()
        self.bh_gen      = BHPlateGenerator()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray, variant: int) -> np.ndarray:
        """variant 0 → clean; অন্য সব → augmented।"""
        return img if variant == 0 else self.augmenter.apply(img)

    def _build_split(self,
                     single_plates : List[str],
                     double_plates : List[str],
                     bh_plates     : List[str],
                     split_dir     : Path) -> int:
        """একটা split-এর জন্য সব images এবং labels তৈরি করে।

        Returns:
            int: মোট saved image সংখ্যা
        """
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        labels : List[str] = []
        count  = 0

        # ── Type 1: Single-line standard ─────────────────────────────────
        for plate in tqdm(single_plates,
                          desc=f"  [{split_dir.name}] single-line"):
            bg = random.choices(BG_COLORS, weights=BG_WEIGHTS, k=1)[0]
            for v in range(self.cfg.images_per_plate):
                img  = self.single_rnd.render(plate, bg_color=bg)
                img  = self._augment(img, v)
                fname = f"SL_{plate}_{v:03d}.png"
                cv2.imwrite(str(img_dir / fname), img)
                labels.append(f"{fname}\t{plate}")
                count += 1

        # ── Type 2: Double-line → full plate single-line strip ───────────
        # Double-line plate-এর text একটা strip-এ render করো
        # label = full plate number (partial label নয়)
        for plate in tqdm(double_plates,
                          desc=f"  [{split_dir.name}] double-line"):
            bg = random.choices(BG_COLORS, weights=BG_WEIGHTS, k=1)[0]
            for v in range(self.cfg.images_per_plate):
                img   = self.single_rnd.render(plate, bg_color=bg)
                img   = self._augment(img, v)
                fname = f"DL_{plate}_{v:03d}.png"
                cv2.imwrite(str(img_dir / fname), img)
                labels.append(f"{fname}\t{plate}")
                count += 1

        # ── Type 3: BH-series ────────────────────────────────────────────
        for plate in tqdm(bh_plates,
                          desc=f"  [{split_dir.name}] BH-series  "):
            # BH-series → mostly white background
            bg = random.choices(BG_COLORS, weights=[0.30, 0.70], k=1)[0]
            for v in range(self.cfg.images_per_plate):
                img  = self.single_rnd.render(plate, bg_color=bg)
                img  = self._augment(img, v)
                fname = f"BH_{plate}_{v:03d}.png"
                cv2.imwrite(str(img_dir / fname), img)
                labels.append(f"{fname}\t{plate}")
                count += 1

        (split_dir / "labels.txt").write_text(
            "\n".join(labels), encoding="utf-8"
        )
        return count

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Complete dataset build করে এবং disk-এ save করে।"""
        cfg = self.cfg

        # Estimate total images
        est_single = cfg.single_line_plates * cfg.images_per_plate
        est_double = cfg.double_line_plates * cfg.images_per_plate * 2
        est_bh     = cfg.bh_series_plates   * cfg.images_per_plate
        est_total  = est_single + est_double + est_bh

        print("=" * 60)
        print("  Synthetic Plate Dataset Generator  v2.0")
        print("=" * 60)
        print(f"  Font          : {Path(cfg.font_path).name}")
        print(f"  Single-line   : {cfg.single_line_plates} plates → ~{est_single:,} images")
        print(f"  Double-line   : {cfg.double_line_plates} plates → ~{est_double:,} images")
        print(f"  BH-series     : {cfg.bh_series_plates} plates  → ~{est_bh:,} images")
        print(f"  Total         : ~{est_total:,} images")
        print(f"  Background    : Yellow 70% / White 30%")
        print(f"  Output        : {cfg.output_dir}/")
        print("=" * 60)

        # Generate all plate strings
        print("\n[1] Generating plate strings…")
        single_plates = self.std_gen.generate(cfg.single_line_plates)
        double_plates = self.std_gen.generate(cfg.double_line_plates)
        bh_plates     = self.bh_gen.generate(cfg.bh_series_plates)

        # Train / val split
        def split(lst):
            n = int(len(lst) * (1 - cfg.val_ratio))
            return lst[:n], lst[n:]

        s_train, s_val = split(single_plates)
        d_train, d_val = split(double_plates)
        b_train, b_val = split(bh_plates)

        print(f"  Single  → train={len(s_train)}, val={len(s_val)}")
        print(f"  Double  → train={len(d_train)}, val={len(d_val)}")
        print(f"  BH      → train={len(b_train)}, val={len(b_val)}")

        out = Path(cfg.output_dir)

        # Build train split
        print("\n[2] Building training set…")
        train_count = self._build_split(
            s_train, d_train, b_train, out / "train_1"
        )
        print(f"  Train total: {train_count:,} images")

        # Build val split
        print("\n[3] Building validation set…")
        val_count = self._build_split(
            s_val, d_val, b_val, out / "val_1"
        )
        print(f"  Val total:   {val_count:,} images")

        print("\n" + "=" * 60)
        print(f"  Done!  Total images: {train_count + val_count:,}")
        print(f"  Dataset → {cfg.output_dir}/")
        print("=" * 60)

        # ── Create ZIP archive of the generated dataset ───────────────────────
        import zipfile
        zip_path = out / "plate_dataset.zip"
        print(f"\n[4] Creating ZIP archive → {zip_path} …")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in ["train_1", "val_1"]:
                folder_path = out / folder
                if folder_path.exists():
                    files = list(folder_path.rglob("*"))
                    for i, f in enumerate(files, 1):
                        if f.is_file():
                            zf.write(f, f.relative_to(out))
                        if i % 5000 == 0:
                            print(f"    {folder}: {i}/{len(files)} files zipped…")
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  ZIP done!  Size: {zip_size_mb:.1f} MB → {zip_path}")
        print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = DatasetConfig(
        font_path          = FONT_PATH,
        output_dir         = OUTPUT_DIR,
        img_height         = IMG_HEIGHT,
        images_per_plate   = IMAGES_PER_PLATE,
        single_line_plates = SINGLE_LINE_PLATES,   # 1700 → 34k images
        double_line_plates = DOUBLE_LINE_PLATES,   #  750 → 30k images
        bh_series_plates   = BH_SERIES_PLATES,     #  300 →  6k images
        val_ratio          = VAL_RATIO,
    )

    builder = SyntheticDatasetBuilder(cfg)
    builder.build()
