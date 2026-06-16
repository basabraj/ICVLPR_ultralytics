"""
Lightweight CRNN Trainer for Indian License Plate OCR
======================================================
Raspberry Pi-এর জন্য optimized lightweight CRNN model।
Charles Wright font-এ generate করা synthetic data দিয়ে train করে।

Architecture:
    CNN (feature extraction) → BiLSTM (sequence) → CTC (decode)

Usage:
    python3 train_crnn.py

Output:
    best_model.pth  ← সবচেয়ে ভালো checkpoint
    last_model.pth  ← শেষ epoch-এর checkpoint

Author  : ANPR Project
Version : 1.0
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_DIR = "."
SAVE_DIR    = "models"
IMG_H       = 32        # CRNN standard input height
IMG_W       = 128       # fixed width (padding/crop করা হবে)
BATCH_SIZE  = 32        # Pi-র জন্য ছোট batch
EPOCHS      = 20
PATIENCE    = 5         # early stopping patience
LR          = 1e-3
NUM_WORKERS = 2         # Pi 4-এ 2 workers ভালো

# Character set — Indian plate-এ শুধু এগুলোই আসে
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK   = len(CHARSET)  # CTC blank token index
NUM_CLS = len(CHARSET) + 1   # 36 chars + 1 blank = 37


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training configuration।

    Attributes:
        dataset_dir  : Dataset folder-এর path
        save_dir     : Model save করার folder
        img_h        : Input image height
        img_w        : Input image width (fixed)
        batch_size   : Training batch size
        epochs       : Total training epochs
        lr           : Learning rate
        num_workers  : DataLoader workers
        device       : CPU বা CUDA
    """
    dataset_dir : str   = DATASET_DIR
    save_dir    : str   = SAVE_DIR
    img_h       : int   = IMG_H
    img_w       : int   = IMG_W
    batch_size  : int   = BATCH_SIZE
    epochs      : int   = EPOCHS
    lr          : float = LR
    num_workers : int   = NUM_WORKERS
    device      : str   = "cpu"   # Pi-তে সবসময় CPU


# ── Character Codec ────────────────────────────────────────────────────────────

class PlateCodec:
    """Plate text এবং integer index-এর মধ্যে convert করে।

    Usage:
        codec = PlateCodec()
        encoded = codec.encode("WB04F7728")  # → tensor
        text    = codec.decode(output)       # → "WB04F7728"
    """

    def __init__(self) -> None:
        self.charset  = CHARSET
        self.char2idx = {c: i for i, c in enumerate(self.charset)}
        self.idx2char = {i: c for i, c in enumerate(self.charset)}
        self.blank    = BLANK

    def encode(self, text: str) -> torch.Tensor:
        """Text string → integer tensor।"""
        return torch.tensor(
            [self.char2idx[c] for c in text.upper() if c in self.char2idx],
            dtype=torch.long
        )

    def decode(self, indices: List[int]) -> str:
        """CTC output → text string (blank এবং duplicate সরায়)।"""
        result = []
        prev   = self.blank
        for idx in indices:
            if idx != self.blank and idx != prev:
                result.append(self.idx2char.get(idx, ""))
            prev = idx
        return "".join(result)


# ── Dataset ────────────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """Synthetic plate image dataset।

    labels.txt format:
        image_name.png\tWB04F7728

    Usage:
        ds = PlateDataset("dataset/train_1", codec, img_h=32, img_w=128)
    """

    def __init__(self, split_dir: str, codec: PlateCodec,
                 img_h: int = IMG_H, img_w: int = IMG_W) -> None:
        """
        Args:
            split_dir : "dataset/train_1" বা "dataset/val_1"
            codec     : PlateCodec instance
            img_h     : Target image height
            img_w     : Target image width
        """
        self.img_dir = Path(split_dir) / "images"
        self.codec   = codec
        self.img_h   = img_h
        self.img_w   = img_w

        labels_path  = Path(split_dir) / "labels.txt"
        self.samples : List[Tuple[str, str]] = []

        for line in labels_path.read_text(encoding="utf-8").strip().split("\n"):
            if "\t" in line:
                fname, text = line.split("\t", 1)
                self.samples.append((fname.strip(), text.strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess(self, img_path: Path) -> torch.Tensor:
        """Image load → grayscale → resize → normalize → tensor।"""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        # Fixed size-এ resize (aspect ratio বজায় রেখে, padding সহ)
        h, w = img.shape
        scale = self.img_h / h
        new_w = min(int(w * scale), self.img_w)
        img   = cv2.resize(img, (new_w, self.img_h))

        # Right-pad করো width পূরণ করতে
        if new_w < self.img_w:
            pad = np.full((self.img_h, self.img_w - new_w),
                          200, dtype=np.uint8)
            img = np.hstack([img, pad])

        # Normalize [0, 1] এবং tensor বানাও
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        fname, text      = self.samples[idx]
        img              = self._preprocess(self.img_dir / fname)
        label            = self.codec.encode(text)
        return img, label, len(label)


def collate_fn(batch):
    """Variable-length label গুলো pad করে batch বানায়।"""
    imgs, labels, lengths = zip(*batch)
    imgs    = torch.stack(imgs, 0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    # Labels গুলো concatenate করো (CTC-র জন্য)
    labels_cat = torch.cat(labels, 0)
    return imgs, labels_cat, lengths


# ── Model ──────────────────────────────────────────────────────────────────────

class LightCRNN(nn.Module):
    """Raspberry Pi-এর জন্য lightweight CRNN model।

    Architecture:
        CNN  : 4টা Conv block (feature extraction)
        RNN  : 2-layer Bidirectional LSTM (sequence modeling)
        FC   : 37-class output (36 chars + CTC blank)

    Model size: ~1.5 MB
    Pi inference: ~50ms per plate image
    """

    def __init__(self, num_classes: int = NUM_CLS) -> None:
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1: 1→32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # H/2, W/2

            # Block 2: 32→64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # H/4, W/4

            # Block 3: 64→128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),        # H/8, W/4

            # Block 4: 128→128
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # After CNN: (B, 128, H/8, W/4) = (B, 128, 4, 32) for H=32,W=128

        # ── RNN ───────────────────────────────────────────────────────────
        # CNN feature-কে sequence-এ reshape করার জন্য
        self.rnn = nn.LSTM(
            input_size  = 128 * 4,   # 128 channels × 4 height
            hidden_size = 128,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout = 0.2,
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.fc = nn.Linear(256, num_classes)  # 128*2 (bidirectional) → 37

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale plate image

        Returns:
            (T, B, num_classes) — CTC-র জন্য time-first format
        """
        # CNN features
        feat = self.cnn(x)                    # (B, 128, 4, W')

        B, C, H, W = feat.shape
        # Reshape: width = time sequence
        feat = feat.permute(0, 3, 1, 2)       # (B, W', 128, 4)
        feat = feat.reshape(B, W, C * H)      # (B, W', 512)

        # RNN
        rnn_out, _ = self.rnn(feat)           # (B, W', 256)

        # Classifier
        out = self.fc(rnn_out)                # (B, W', 37)
        return out.permute(1, 0, 2)           # (T=W', B, 37)  ← CTC format


# ── Trainer ────────────────────────────────────────────────────────────────────

class CRNNTrainer:
    """CRNN model training পরিচালনা করে।

    Features:
        - CTC loss দিয়ে train করে
        - Best model checkpoint save করে
        - Pi-এর জন্য optimized (CPU, small batch)
        - প্রতি epoch-এ ETA দেখায়

    Usage:
        trainer = CRNNTrainer(config)
        trainer.train()
    """

    def __init__(self, config: TrainConfig = TrainConfig()) -> None:
        self.cfg    = config
        self.codec  = PlateCodec()
        self.device = torch.device(config.device)

        # Pi-এর সব CPU core ব্যবহার করো
        torch.set_num_threads(os.cpu_count() or 4)

        Path(config.save_dir).mkdir(exist_ok=True)
        self._build()

    def _build(self) -> None:
        """Dataset, model, optimizer, loss তৈরি করে।"""
        cfg = self.cfg

        # Dataset
        self.train_ds = PlateDataset(
            f"{cfg.dataset_dir}/train_1", self.codec,
            cfg.img_h, cfg.img_w
        )
        self.val_ds = PlateDataset(
            f"{cfg.dataset_dir}/val_1", self.codec,
            cfg.img_h, cfg.img_w
        )

        # DataLoader
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            num_workers = cfg.num_workers,
            collate_fn  = collate_fn,
            pin_memory  = False,   # CPU-তে pin_memory=False
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size  = cfg.batch_size * 2,
            shuffle     = False,
            num_workers = cfg.num_workers,
            collate_fn  = collate_fn,
        )

        # Model
        self.model = LightCRNN(NUM_CLS).to(self.device)

        # Optimizer + Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        # CTC Loss
        self.criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

        self.best_acc = 0.0

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model params : {total_params:,}  (~{total_params*4//1024} KB)")

    def _train_epoch(self, epoch: int) -> float:
        """একটা epoch train করে।"""
        self.model.train()
        total_loss = 0.0
        steps      = len(self.train_dl)
        t0         = time.time()

        for i, (imgs, labels, label_lens) in enumerate(self.train_dl, 1):
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            logits      = self.model(imgs)          # (T, B, C)
            T, B, _     = logits.shape
            input_lens  = torch.full((B,), T, dtype=torch.long)
            log_probs   = logits.log_softmax(2)

            loss = self.criterion(log_probs, labels, input_lens, label_lens)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()

            # ETA দেখাও
            if i % 50 == 0 or i == steps:
                elapsed = time.time() - t0
                eta_s   = elapsed / i * (steps - i)
                eta_min = int(eta_s // 60)
                eta_sec = int(eta_s % 60)
                print(f"  Epoch {epoch:02d} [{i:4d}/{steps}] "
                      f"loss={total_loss/i:.4f}  "
                      f"ETA {eta_min}m{eta_sec:02d}s", end="\r")

        print()
        return total_loss / steps

    def _val_epoch(self) -> float:
        """Validation accuracy calculate করে।"""
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for imgs, labels, label_lens in self.val_dl:
                imgs      = imgs.to(self.device)
                logits    = self.model(imgs)      # (T, B, C)
                preds     = logits.argmax(2)      # (T, B)
                preds_t   = preds.permute(1, 0)  # (B, T)

                # labels unpack করো
                offset = 0
                for b, ll in enumerate(label_lens.tolist()):
                    gt_text  = self.codec.decode(
                        labels[offset:offset + ll].tolist()
                    )
                    pred_text = self.codec.decode(preds_t[b].tolist())
                    if pred_text == gt_text:
                        correct += 1
                    total   += 1
                    offset  += ll

        return correct / total if total else 0.0

    def _save(self, name: str) -> None:
        path = Path(self.cfg.save_dir) / name
        torch.save({
            "model":   self.model.state_dict(),
            "charset": CHARSET,
            "img_h":   self.cfg.img_h,
            "img_w":   self.cfg.img_w,
        }, path)
        print(f"  Saved → {path}")

    def train(self) -> None:
        """Full training loop চালায়।"""
        cfg = self.cfg
        print("=" * 55)
        print("  LightCRNN Training — Raspberry Pi")
        print("=" * 55)
        print(f"  Train samples : {len(self.train_ds):,}")
        print(f"  Val samples   : {len(self.val_ds):,}")
        print(f"  Epochs        : {cfg.epochs}")
        print(f"  Batch size    : {cfg.batch_size}")
        print(f"  Device        : {cfg.device} "
              f"({os.cpu_count()} cores)")
        print("=" * 55)

        patience_counter = 0
        for epoch in range(1, cfg.epochs + 1):
            t_start  = time.time()
            train_loss = self._train_epoch(epoch)
            val_acc    = self._val_epoch()
            elapsed    = time.time() - t_start

            self.scheduler.step()

            print(f"  Epoch {epoch:02d}/{cfg.epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"val_acc={val_acc*100:.1f}% | "
                  f"time={elapsed/60:.1f}m")

            # Best model save
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save("best_model.pth")
                print(f"  ★ New best: {val_acc*100:.1f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improve: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("  Early stopping!")
                    break

            # Last model always save
            self._save("last_model.pth")

        print("=" * 55)
        print(f"  Training done!  Best accuracy: {self.best_acc*100:.1f}%")
        print(f"  Model saved → {cfg.save_dir}/best_model.pth")
        print("=" * 55)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = TrainConfig(
        dataset_dir = DATASET_DIR,
        save_dir    = SAVE_DIR,
        img_h       = IMG_H,
        img_w       = IMG_W,
        batch_size  = BATCH_SIZE,
        epochs      = EPOCHS,
        lr          = LR,
        num_workers = NUM_WORKERS,
        device      = "cpu",        # Pi-তে সবসময় CPU
    )

    trainer = CRNNTrainer(config)
    trainer.train()
