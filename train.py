"""
ANPR Suggestion Engine
Learns from user-verified plate corrections and suggests corrections for new OCR reads.

Pipeline for each new OCR result:
  1. Apply positional rules  (0↔O, 1↔I, 8↔B by plate position)
  2. Check exact match in known OCR variants  → instant suggestion
  3. Fuzzy match against all verified correct plates  → suggestion if score >= threshold
  4. None matched → no suggestion, user types manually
  5. User confirmation → saved as new variant, model updated instantly
"""

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

# ── Constants ──────────────────────────────────────────────────────────────────
SUGGEST_THRESHOLD = 78          # minimum fuzzy score (0-100) to show suggestion
DB_PATH           = "anpr.db"

# Common OCR character swaps: letter ↔ digit
_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2'}
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z'}


# ── Data class for a suggestion result ────────────────────────────────────────
@dataclass
class Suggestion:
    correct_plate: str
    score: float                        # 0-100 confidence
    source: str                         # 'exact_variant' | 'fuzzy' | 'positional'
    history: dict = field(default_factory=dict)   # avg_gross, avg_net, hit_count


# ── Positional rules for Indian licence plates ─────────────────────────────────
def apply_positional_rules(text: str) -> str:
    """
    Indian plate format:  SS DD LLL NNNN
      SS   = 2 state letters      (must be letters)
      DD   = 2 district digits    (must be digits)
      LLL  = 1-3 series letters   (must be letters)
      NNNN = 4 number digits      (must be digits)

    Fixes obvious OCR swaps based on expected character class at each position.
    """
    clean = text.replace(' ', '').upper()
    if len(clean) < 6:
        return text

    chars = list(clean)
    n = len(chars)

    def force_letter(c):
        return _DIGIT_TO_LETTER.get(c, c)

    def force_digit(c):
        return _LETTER_TO_DIGIT.get(c, c)

    # Positions 0,1 → must be letters (state code)
    for i in (0, 1):
        chars[i] = force_letter(chars[i])

    # Positions 2,3 → must be digits (district code)
    for i in (2, 3):
        chars[i] = force_digit(chars[i])

    # Last 4 positions → must be digits (plate number)
    for i in range(max(4, n - 4), n):
        chars[i] = force_digit(chars[i])

    # Middle section (4 to n-4) → must be letters (series)
    for i in range(4, max(4, n - 4)):
        chars[i] = force_letter(chars[i])

    return ''.join(chars)


# ── Normalise plate for comparison ────────────────────────────────────────────
def _norm(plate: str) -> str:
    return plate.replace(' ', '').upper()


# ── SuggestionEngine ──────────────────────────────────────────────────────────
class SuggestionEngine:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path   = db_path
        self._lock     = threading.Lock()
        self._plates: list[str] = []          # all correct_plate values (normalised)
        self._plate_raw: list[str] = []       # original spacing versions
        self._variants: dict[str, str] = {}   # normalised_variant → correct_plate
        self._init_db()
        self.rebuild()

    # ── DB setup ──────────────────────────────────────────────────────────────
    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS verified_plates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                correct_plate TEXT    NOT NULL UNIQUE,
                ocr_variants  TEXT    DEFAULT '[]',
                hit_count     INTEGER DEFAULT 1,
                last_seen     TEXT,
                avg_gross     REAL,
                avg_net       REAL
            )
        """)
        con.commit()
        con.close()

    # ── Rebuild in-memory index from DB ───────────────────────────────────────
    def rebuild(self):
        con  = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT correct_plate, ocr_variants FROM verified_plates"
        ).fetchall()
        con.close()

        plates     = []
        plates_raw = []
        variants   = {}

        for correct, var_json in rows:
            norm = _norm(correct)
            plates.append(norm)
            plates_raw.append(correct)
            variants[norm] = correct
            for v in json.loads(var_json or '[]'):
                variants[_norm(v)] = correct

        with self._lock:
            self._plates     = plates
            self._plate_raw  = plates_raw
            self._variants   = variants

    # ── Core: suggest correct plate for a raw OCR string ──────────────────────
    def suggest(self, ocr_text: str) -> Optional[Suggestion]:
        if not ocr_text:
            return None

        norm_ocr = _norm(ocr_text)

        with self._lock:
            plates   = list(self._plates)
            variants = dict(self._variants)

        # Step 1 — exact variant match (instant, no fuzzy needed)
        if norm_ocr in variants:
            correct = variants[norm_ocr]
            hist    = self._get_history(correct)
            return Suggestion(correct, 100.0, 'exact_variant', hist)

        # Step 2 — apply positional rules then check again
        fixed     = apply_positional_rules(ocr_text)
        norm_fixed = _norm(fixed)
        if norm_fixed in variants:
            correct = variants[norm_fixed]
            hist    = self._get_history(correct)
            return Suggestion(correct, 97.0, 'positional', hist)

        if not plates:
            return None

        # Step 3 — fuzzy match against all known correct plates
        # Use token_set_ratio to handle spacing differences
        best = process.extractOne(
            norm_ocr,
            plates,
            scorer=fuzz.token_set_ratio,
        )
        if best and best[1] >= SUGGEST_THRESHOLD:
            idx     = plates.index(best[0])
            correct = self._plate_raw[idx]
            hist    = self._get_history(correct)
            return Suggestion(correct, float(best[1]), 'fuzzy', hist)

        # Step 4 — try positional-fixed version in fuzzy
        if norm_fixed != norm_ocr:
            best2 = process.extractOne(
                norm_fixed,
                plates,
                scorer=fuzz.token_set_ratio,
            )
            if best2 and best2[1] >= SUGGEST_THRESHOLD:
                idx     = plates.index(best2[0])
                correct = self._plate_raw[idx]
                hist    = self._get_history(correct)
                return Suggestion(correct, float(best2[1]), 'fuzzy+positional', hist)

        return None

    # ── Learn: user confirmed correct_plate for this OCR read ─────────────────
    def learn(self, ocr_text: str, correct_plate: str,
              gross: float = None, net: float = None):
        if not correct_plate:
            return

        norm_ocr     = _norm(ocr_text)
        norm_correct = _norm(correct_plate)

        con = sqlite3.connect(self.db_path)

        row = con.execute(
            "SELECT id, ocr_variants, hit_count, avg_gross, avg_net "
            "FROM verified_plates WHERE correct_plate = ?",
            (correct_plate,)
        ).fetchone()

        if row:
            vid, var_json, hits, ag, an = row
            variants = json.loads(var_json or '[]')
            norm_vars = [_norm(v) for v in variants]

            if norm_ocr not in norm_vars and norm_ocr != norm_correct:
                variants.append(ocr_text.upper())

            new_hits  = hits + 1
            new_gross = ((ag or 0) * hits + (gross or 0)) / new_hits if gross else ag
            new_net   = ((an or 0) * hits + (net   or 0)) / new_hits if net   else an

            con.execute("""
                UPDATE verified_plates
                SET ocr_variants = ?, hit_count = ?, last_seen = datetime('now'),
                    avg_gross = ?, avg_net = ?
                WHERE id = ?
            """, (json.dumps(variants), new_hits, new_gross, new_net, vid))
        else:
            variants = [] if norm_ocr == norm_correct else [ocr_text.upper()]
            con.execute("""
                INSERT OR IGNORE INTO verified_plates
                  (correct_plate, ocr_variants, hit_count, last_seen, avg_gross, avg_net)
                VALUES (?, ?, 1, datetime('now'), ?, ?)
            """, (correct_plate, json.dumps(variants), gross, net))

        con.commit()
        con.close()
        self.rebuild()

    # ── History for a verified plate ──────────────────────────────────────────
    def _get_history(self, correct_plate: str) -> dict:
        con = sqlite3.connect(self.db_path)
        row = con.execute(
            "SELECT hit_count, avg_gross, avg_net, last_seen "
            "FROM verified_plates WHERE correct_plate = ?",
            (correct_plate,)
        ).fetchone()
        con.close()
        if not row:
            return {}
        return {
            "hit_count": row[0],
            "avg_gross": row[1],
            "avg_net":   row[2],
            "last_seen": row[3],
        }

    # ── Bulk rebuild from detections (called on server start) ─────────────────
    def bootstrap_from_detections(self):
        """Seed verified_plates from existing confirmed detections in the DB."""
        con  = sqlite3.connect(self.db_path)
        rows = con.execute("""
            SELECT plate, correct_plate, gross_weight, net_weight
            FROM detections
            WHERE plate_verified = 1 AND correct_plate IS NOT NULL
        """).fetchall()
        con.close()
        for ocr, correct, gross, net in rows:
            if correct:
                self.learn(ocr, correct, gross, net)


# ── Module-level singleton (imported by anpr_server) ─────────────────────────
engine = SuggestionEngine(DB_PATH)
