"""
EasyOCR Fine-Tuning — Indian License Plates
============================================
Google Colab (T4 GPU) — Training only script
Dataset already generated করা আছে।

Usage (Colab-এ cell-by-cell run করো):
    Cell 1 → Install
    Cell 2 → এই file import করো
    Cell 3 → Fine-tune চালাও
    Cell 4 → Model download করো

Author  : ANPR Project
Version : 2.0 (corrected)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import easyocr
from torch.utils.data import Dataset, DataLoader


# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"        # dataset/train/ এবং dataset/val/ থাকতে হবে
SAVE_DIR      = "finetuned"
IMG_H         = 64               # EasyOCR recognition model input height
IMG_W         = 200              # fixed width (pad/crop হবে)
BATCH_SIZE    = 64               # T4 GPU-তে ভালো
EPOCHS        = 20
LR            = 1e-4             # pre-trained model → ছোট LR দরকার
PATIENCE      = 5
NUM_WORKERS   = 2
MAX_LABEL_LEN = 25               # EasyOCR default batch_max_length


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """License plate image dataset for EasyOCR fine-tuning।

    Image preprocessing:
        Grayscale → resize H=64 → pad W=200 → normalize [-1, 1]

    Attributes:
        samples: list of (image_filename, plate_text) tuples
    """

    def __init__(self, split_dir: str) -> None:
        """
        Args:
            split_dir: "dataset/train" বা "dataset/val"
        """
        self.img_dir = Path(split_dir) / "images"
        self.samples: List[Tuple[str, str]] = []

        labels_file = Path(split_dir) / "labels.txt"
        for line in labels_file.read_text(encoding="utf-8").strip().split("\n"):
            if "\t" in line:
                fname, text = line.split("\t", 1)
                text = text.strip()
                # MAX_LABEL_LEN-এর বেশি লম্বা plate বাদ দাও
                if 4 <= len(text) <= MAX_LABEL_LEN:
                    self.samples.append((fname.strip(), text))

        print(f"  Loaded {len(self.samples):,} samples from {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname, text = self.samples[idx]
        img = cv2.imread(str(self.img_dir / fname), cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.full((IMG_H, IMG_W), 200, dtype=np.uint8)

        # Height → IMG_H, width proportional, max IMG_W
        h, w  = img.shape
        new_w = min(int(w * IMG_H / h), IMG_W)
        img   = cv2.resize(img, (new_w, IMG_H))

        # Right-pad to IMG_W
        if new_w < IMG_W:
            pad = np.full((IMG_H, IMG_W - new_w), 200, dtype=np.uint8)
            img = np.hstack([img, pad])

        # EasyOCR normalization: [0,255] → [-1, 1]
        img    = (img.astype(np.float32) - 127.5) / 127.5
        tensor = torch.tensor(img).unsqueeze(0)   # (1, H, W)
        return tensor, text


def collate_fn(batch: List) -> Tuple[torch.Tensor, List[str]]:
    """Variable-length label-এর জন্য custom collate।"""
    imgs, texts = zip(*batch)
    return torch.stack(imgs, 0), list(texts)


# ── Fine-tuner ─────────────────────────────────────────────────────────────────

class EasyOCRFinetuner:
    """EasyOCR recognition model fine-tune করে।

    EasyOCR দুই ধরনের model ব্যবহার করে:
        CTC   → CTCLoss + greedy decode
        Attn  → CrossEntropyLoss + attention decode

    এই class automatically detect করে কোন type ব্যবহার হচ্ছে।

    Usage:
        tuner = EasyOCRFinetuner(gpu=True)
        tuner.train()
    """

    def __init__(self, gpu: bool = True) -> None:
        # Device setup
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"\nDevice  : {self.device}")

        # EasyOCR reader load করো (এটা base model download করবে)
        print("EasyOCR model loading…")
        self.reader    = easyocr.Reader(['en'], gpu=gpu,
                                         verbose=False)
        self.model     = self.reader.recognizer.to(self.device)
        self.converter = self.reader.converter

        # Model type detect করো
        conv_name       = type(self.converter).__name__
        self.model_type = "CTC" if "CTC" in conv_name else "Attn"
        print(f"Converter: {conv_name}")
        print(f"Type    : {self.model_type}")

        # Save directory
        Path(SAVE_DIR).mkdir(exist_ok=True)
        self.best_acc = 0.0

        # Optimizer — pre-trained model থেকে শুরু → ছোট LR
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LR
        )
        # val_acc না বাড়লে LR কমাবে
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max",
            patience=2, factor=0.5
        )

    # ── Loss functions ─────────────────────────────────────────────────────────

    def _loss_attn(self, imgs: torch.Tensor,
                   texts: List[str]) -> torch.Tensor:
        """Attention model-এর loss। Teacher forcing ব্যবহার করে।"""
        text_enc, _ = self.converter.encode(
            texts, batch_max_length=MAX_LABEL_LEN
        )
        text_enc = text_enc.to(self.device)

        # Forward: teacher forcing (text[:-1] দিয়ে predict করে text[1:])
        preds  = self.model(imgs, text_enc[:, :-1], is_train=True)
        target = text_enc[:, 1:].contiguous()

        B, T, C = preds.size()
        loss = nn.CrossEntropyLoss(ignore_index=0)(
            preds.view(-1, C),
            target.view(-1)
        )
        return loss

    def _loss_ctc(self, imgs: torch.Tensor,
                  texts: List[str]) -> torch.Tensor:
        """CTC model-এর loss।"""
        text_enc, length = self.converter.encode(texts)
        text_enc = text_enc.to(self.device)

        preds      = self.model(imgs).log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * imgs.size(0))

        loss = nn.CTCLoss(blank=0, zero_infinity=True)(
            preds.permute(1, 0, 2),
            text_enc, preds_size, length
        )
        return loss

    # ── Decode functions ───────────────────────────────────────────────────────

    def _decode_attn(self, imgs: torch.Tensor) -> List[str]:
        """Attention model inference (greedy decode)।"""
        B          = imgs.size(0)
        start_tok  = torch.zeros(B, MAX_LABEL_LEN + 1,
                                 dtype=torch.long).to(self.device)

        preds            = self.model(imgs, start_tok, is_train=False)
        _, preds_idx     = preds.max(2)
        length_for_pred  = torch.IntTensor([MAX_LABEL_LEN] * B)

        preds_str = self.converter.decode(preds_idx, length_for_pred)

        # [s] end token সরিয়ে দাও
        return [p.split("[s]")[0].upper().strip() for p in preds_str]

    def _decode_ctc(self, imgs: torch.Tensor) -> List[str]:
        """CTC model inference (greedy decode)।"""
        preds      = self.model(imgs)
        preds_size = torch.IntTensor([preds.size(1)] * imgs.size(0))
        _, idx     = preds.max(2)
        preds_str  = self.converter.decode(idx.data, preds_size.data)
        return [p.upper().strip() for p in preds_str]

    # ── Training / validation ──────────────────────────────────────────────────

    def _train_epoch(self, dl: DataLoader, epoch: int) -> float:
        """একটা training epoch।"""
        self.model.train()
        total_loss = 0.0
        steps      = len(dl)
        t0         = time.time()

        for i, (imgs, texts) in enumerate(dl, 1):
            imgs = imgs.to(self.device)

            if self.model_type == "Attn":
                loss = self._loss_attn(imgs, texts)
            else:
                loss = self._loss_ctc(imgs, texts)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Progress দেখাও
            if i % 100 == 0 or i == steps:
                elapsed = time.time() - t0
                eta     = elapsed / i * (steps - i)
                print(
                    f"\r  Epoch {epoch:02d} [{i:4d}/{steps}] "
                    f"loss={total_loss/i:.4f}  "
                    f"ETA {int(eta//60)}m{int(eta%60):02d}s",
                    end=""
                )
        print()
        return total_loss / steps

    def _val_epoch(self, dl: DataLoader) -> float:
        """Validation accuracy (exact plate match)।"""
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for imgs, texts in dl:
                imgs = imgs.to(self.device)

                if self.model_type == "Attn":
                    preds = self._decode_attn(imgs)
                else:
                    preds = self._decode_ctc(imgs)

                for pred, gt in zip(preds, texts):
                    if pred == gt.upper():
                        correct += 1
                    total += 1

        return correct / total if total else 0.0

    def _save(self, name: str) -> None:
        path = Path(SAVE_DIR) / name
        torch.save(self.model.state_dict(), str(path))
        print(f"  Saved → {path}")

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self) -> None:
        """Complete fine-tuning loop।"""
        # DataLoaders তৈরি করো
        train_ds = PlateDataset(f"{DATASET_DIR}/train")
        val_ds   = PlateDataset(f"{DATASET_DIR}/val")

        train_dl = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True,  num_workers=NUM_WORKERS,
            collate_fn=collate_fn, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=BATCH_SIZE * 2,
            shuffle=False, num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )

        print("\n" + "=" * 55)
        print("  EasyOCR Fine-tuning")
        print("=" * 55)
        print(f"  Train    : {len(train_ds):,} samples")
        print(f"  Val      : {len(val_ds):,} samples")
        print(f"  Epochs   : {EPOCHS}")
        print(f"  LR       : {LR}")
        print(f"  Patience : {PATIENCE}")
        print(f"  Device   : {self.device}")
        print("=" * 55 + "\n")

        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            t0         = time.time()
            train_loss = self._train_epoch(train_dl, epoch)
            val_acc    = self._val_epoch(val_dl)
            elapsed    = time.time() - t0

            self.scheduler.step(val_acc)

            print(
                f"  Epoch {epoch:02d}/{EPOCHS} | "
                f"loss={train_loss:.4f} | "
                f"val_acc={val_acc*100:.1f}% | "
                f"time={elapsed/60:.1f}m"
            )

            if val_acc > self.best_acc:
                self.best_acc    = val_acc
                patience_counter = 0
                self._save("best_model.pth")
                print(f"  ★ New best: {val_acc*100:.1f}%")
            else:
                patience_counter += 1
                print(f"  No improve: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("  Early stopping!")
                    break

            self._save("last_model.pth")

        print("\n" + "=" * 55)
        print(f"  Done!  Best accuracy: {self.best_acc*100:.1f}%")
        print(f"  Model  → {SAVE_DIR}/best_model.pth")
        print("=" * 55)
