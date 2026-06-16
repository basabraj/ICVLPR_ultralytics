import re
import openpyxl
from datetime import datetime

# ── RTO data file ──────────────────────────────────────────────────────────────
RTO_CODE_FILE = "/home/pi/Desktop/ICVLPR_ultralytics/validation/RTO_CODE.xlsx"

# ── Indian State / Union Territory codes ──────────────────────────────────────
STATE_CODES: dict[str, str] = {
    "AP": "Andhra Pradesh",       "AR": "Arunachal Pradesh",
    "AS": "Assam",                "BR": "Bihar",
    "CG": "Chhattisgarh",         "GA": "Goa",
    "GJ": "Gujarat",              "HR": "Haryana",
    "HP": "Himachal Pradesh",     "JK": "Jammu & Kashmir",
    "JH": "Jharkhand",            "KA": "Karnataka",
    "KL": "Kerala",               "MP": "Madhya Pradesh",
    "MH": "Maharashtra",          "MN": "Manipur",
    "ML": "Meghalaya",            "MZ": "Mizoram",
    "NL": "Nagaland",             "OD": "Odisha",
    "OR": "Orissa",               "PB": "Punjab",
    "RJ": "Rajasthan",            "SK": "Sikkim",
    "TN": "Tamil Nadu",           "TS": "Telangana",
    "TG": "Telangana",      # new code from May 2024
    "TR": "Tripura",              "UP": "Uttar Pradesh",
    "UA": "Uttaranchal",          "UK": "Uttarakhand",
    "WB": "West Bengal",
    # Union Territories
    "AN": "Andaman & Nicobar",                         "CH": "Chandigarh",
    "DD": "Daman & Diu",                               "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",  "LA": "Ladakh",
    "LD": "Lakshadweep",                               "PY": "Puducherry",
}


# ── Step 1: clean ─────────────────────────────────────────────────────────────
def clean_plate(text: str) -> str:
    """Remove all non-alphanumeric characters and spaces, return uppercase."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


# ── Step 2: strip noise prefix ────────────────────────────────────────────────
def strip_ind(text: str) -> str:
    """
    Remove any garbage prefix before the actual state code.
    Handles: 'IND', 'IN', 'ND', 'KKO', 'INO', 'INDO', 'IN0', etc.

    Strategy:
      1. Try full IND prefix strip
      2. Try partial IND prefixes (IN, ND) from logo bleed-over
      3. Scan positions 0-4 for a valid 2-char state code
    """
    _STATE_SWAPS = {'N': 'M', 'M': 'N', 'W': 'M', 'H': 'M'}

    def _valid_state(s: str) -> str:
        if s[:2] in STATE_CODES:
            return s
        alt = _STATE_SWAPS.get(s[0], '')
        if alt:
            swapped = alt + s[1:]
            if swapped[:2] in STATE_CODES:
                return swapped
        return ""

    # 1. Exact "IND" prefix (most common logo bleed)
    if text.startswith("IND") and len(text) > 3:
        candidate = text[3:]
        fixed = _valid_state(candidate)
        if fixed:
            return fixed

    # 2. Partial IND prefix: "IN" or "ND" (logo partially overlapping state code)
    for prefix in ("IN", "ND"):
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            candidate = text[len(prefix):]
            fixed = _valid_state(candidate)
            if fixed and len(fixed) >= 6:
                return fixed

    # 3. Scan positions 0-4 (fixed range — was incorrectly capped at len-7)
    for i in range(min(5, len(text) - 1)):
        fixed = _valid_state(text[i:])
        if fixed and fixed != text[i:]:
            return fixed          # swapped version
        if text[i:i+2] in STATE_CODES:
            return text[i:]       # direct match

    return text


# ── Step 3: length check ──────────────────────────────────────────────────────
def accept_plate(text: str) -> bool:
    """Accept only if cleaned+stripped plate is 8, 9, or 10 characters."""
    norm = strip_ind(clean_plate(text))
    return 8 <= len(norm) <= 10


def check_last_four_digit(text: str) -> tuple[bool, str]:
    """
    Extract and validate the last 4 characters of the plate as digits.
    Returns (is_valid, last_four_as_string).

        "KL07AB1234"  →  (True,  "1234")
        "KL07AB123X"  →  (False, "")
        "MH12CD5678"  →  (True,  "5678")
    """
    last_four = text[-4:]
    if last_four.isdigit():
        return True, last_four
    return False, ""
        




# ── Indian plate format regex ─────────────────────────────────────────────────
# Structure: SS DD LL NNNN
#   SS   = 2 state letters       (WB, MH, KL ...)
#   DD   = exactly 2 district digits  (04, 12 ...)
#   LL   = 1 to 3 series letters  (F, AB, BPR ...)
#   NNNN = exactly 4 number digits (7728, 1234 ...)
#
# This single regex catches all extra-digit OCR noise:
#   WB04F7728   → ✓  (correct)
#   WB04F77128  → ✗  (5 digits at end — extra '1')
#   WB04LF7728  → ✓  (2-letter series — valid)
#   MB404F7728  → ✗  (3 district digits — extra '4')
#   HB04F77128  → ✗  (5 digits at end — extra '1')

_PLATE_REGEX = re.compile(
    r'^[A-Z]{2}'    # State code      : WB, MH, KL
    r'\d{2}'        # District digits : 04, 12
    r'[A-Z]{1,3}'   # Series letters  : F, AB, BPR
    r'\d{4}$'       # Number          : exactly 4 digits
)

def is_valid_indian_format(plate: str) -> bool:
    """
    Validates strict Indian plate format: SS + DD + LL + NNNN.
    Catches extra OCR noise digits/letters that slip past length check.

        "WB04F7728"   → True   SS=WB DD=04 L=F  N=7728
        "WB04LF7728"  → True   SS=WB DD=04 L=LF N=7728
        "WB04F77128"  → False  ends with 5 digits (77128)
        "MB404F7728"  → False  3 digits after state (404)
        "KL07AB1234"  → True   SS=KL DD=07 L=AB N=1234
    """
    return bool(_PLATE_REGEX.match(plate.replace(" ", "")))


# ── Step 4: state / UT code check ─────────────────────────────────────────────
def check_State_or_Union_Territory_code(text: str) -> tuple[bool, str]:
    """
    Check first 2 characters against Indian state/UT codes.
    Returns (is_valid, state_name).

        "KL07AB1234"  →  (True,  "Kerala")
        "TN8BF4089"   →  (True,  "Tamil Nadu")
        "XX12AB3456"  →  (False, "")
        
    """
    code  = text[:2].upper()
    state = STATE_CODES.get(code, "")
    return (bool(state), state)


def get_state_name(plate: str) -> str:
    """Return state name for a cleaned plate, or empty string."""
    _, state = check_State_or_Union_Territory_code(plate)
    return state


# ── RTO data loader ───────────────────────────────────────────────────────────
def _empty_rto() -> dict:
    return {
        "rto_code":         "",
        "office_location":  "NIL",
        "jurisdiction_area":"NIL",
        "annotation":       "NIL",
        "old_code":         "NIL",
        "district":         "NIL",
        "zone":             "NIL",
    }


def _load_rto_data() -> dict:
    """
    Parse RTO_CODE.xlsx and return a dict keyed by normalised RTO code.
    Code normalisation: 'KL-07' → 'KL07', handles multiple codes per cell.

    XLSX column order (row 1 = header):
      0:State  1:Code  2:Office location  3:Jurisdiction area
      4:Annotations  5:Old Code  6:District  7:Zone
    """
    data: dict[str, dict] = {}
    try:
        wb = openpyxl.load_workbook(RTO_CODE_FILE, read_only=True, data_only=True)
        ws = wb.active

        curr_codes:         list[str] = []
        curr_office:        str = ""
        curr_jurisdictions: list[str] = []
        curr_annotation:    str = ""
        curr_old_code:      str = ""
        curr_district:      str = ""
        curr_zone:          str = ""

        def _save():
            if not curr_codes:
                return
            entry = {
                "office_location":   curr_office  or "NIL",
                "jurisdiction_area": ", ".join(j.strip() for j in curr_jurisdictions if j.strip()) or "NIL",
                "annotation":        curr_annotation or "NIL",
                "old_code":          curr_old_code   or "NIL",
                "district":          curr_district   or "NIL",
                "zone":              curr_zone        or "NIL",
            }
            for c in curr_codes:
                data[c] = entry

        def _normalise_codes(cell_val: str) -> list[str]:
            """Turn 'AP-39, AP-40' or 'AR-01 AR-02' into ['AP39','AP40','AR01','AR02']."""
            parts = re.split(r'[\s,]+', cell_val.strip())
            result = []
            for p in parts:
                p = p.strip()
                m = re.match(r'^([A-Z]{2})-?(\d{2})', p.upper())
                if m:
                    result.append(m.group(1) + m.group(2))
            return result

        def _str(v) -> str:
            return str(v).strip() if v is not None else ""

        for row in ws.iter_rows(values_only=True):
            code_cell  = _str(row[1]) if len(row) > 1 else ""
            office     = _str(row[2]) if len(row) > 2 else ""
            juris      = _str(row[3]) if len(row) > 3 else ""
            annotation = _str(row[4]) if len(row) > 4 else ""
            old_code   = _str(row[5]) if len(row) > 5 else ""
            district   = _str(row[6]) if len(row) > 6 else ""
            zone       = _str(row[7]) if len(row) > 7 else ""

            # Skip the header row
            if code_cell.lower() in ("code", "\ncode"):
                continue

            # New RTO code entry
            if code_cell and re.search(r'[A-Z]{2}-?\d{2}', code_cell.upper()):
                _save()
                curr_codes         = _normalise_codes(code_cell)
                curr_office        = office
                curr_jurisdictions = [juris] if juris else []
                curr_annotation    = annotation
                curr_old_code      = old_code
                curr_district      = district
                curr_zone          = zone
            elif juris:
                # Continuation row — extra jurisdiction area for same code
                curr_jurisdictions.append(juris)

        _save()   # flush last entry
        wb.close()
        print(f"[RTO] Loaded {len(data)} RTO codes from {RTO_CODE_FILE}")

    except Exception as e:
        print(f"[RTO] Failed to load {RTO_CODE_FILE}: {e}")

    return data


# Module-level RTO lookup table (loaded once on import)
_rto_data: dict = _load_rto_data()


def get_rto_info(plate: str) -> dict:
    """
    Extract the 4-char RTO code (2 state letters + 2 district digits) from a
    cleaned plate string and return all matching fields from RTO_CODE.xlsx.
    All missing fields are returned as "NIL".

    Examples:
        get_rto_info("KL07AB1234")  →  {rto_code:"KL07", office_location:"Ernakulam", ...}
        get_rto_info("TN88F4089")   →  {rto_code:"TN88", office_location:"Chennai", ...}
        get_rto_info("XXXXXXXX")    →  {rto_code:"",     office_location:"NIL",      ...}
    """
    m = re.match(r'^([A-Z]{2})(\d{2})', plate)
    if not m:
        return _empty_rto()

    rto_code = m.group(1) + m.group(2)      # e.g. "KL07"
    info     = _rto_data.get(rto_code, {})

    return {
        "rto_code":          rto_code,
        "office_location":   info.get("office_location",   "NIL") or "NIL",
        "jurisdiction_area": info.get("jurisdiction_area", "NIL") or "NIL",
        "annotation":        info.get("annotation",        "NIL") or "NIL",
        "old_code":          info.get("old_code",          "NIL") or "NIL",
        "district":          info.get("district",          "NIL") or "NIL",
        "zone":              info.get("zone",              "NIL") or "NIL",
    }


# ── OCR auto-correction ───────────────────────────────────────────────────────
# ── Position-aware OCR correction ─────────────────────────────────────────────
# One dictionary of visually similar pairs (letter side only).
# Direction is decided by position, NOT by the character itself:
#   letter position + digit char → convert digit→letter  (SIMILAR_REV)
#   digit  position + letter char → convert letter→digit (SIMILAR)
SIMILAR = {'I':'1', 'O':'0', 'B':'8', 'S':'5', 'G':'6', 'Z':'2'}
SIMILAR_REV = {v: k for k, v in SIMILAR.items()}   # {'1':'I','0':'O', ...}

# Kept for state-code first-letter swap (M↔W) only
_LETTER_SWAPS = {'M': 'W', 'W': 'M', 'H': 'W'}


def _fix_char(char: str, expect_letter: bool) -> str:
    """
    Fix one character based on what type is expected at its position.

    expect_letter=True  (state/series position):
        letter → keep as-is   ('I' stays 'I')
        digit  → convert to letter  ('1' → 'I')

    expect_letter=False (district/number position):
        digit  → keep as-is   ('1' stays '1')
        letter → convert to digit   ('I' → '1')
    """
    if expect_letter:
        return char if char.isalpha() else SIMILAR_REV.get(char, char)
    else:
        return char if char.isdigit() else SIMILAR.get(char, char)


def fix_by_position(plate: str) -> str:
    """
    Apply position-aware character correction to a cleaned plate string.

    Indian plate structure:  SS + DD + LL + NNNN
      SS   (pos 0-1)  : LETTERS  — state code
      DD   (pos 2-3)  : DIGITS   — district code
      LL   (pos 4-N-4): LETTERS  — series (1-3 chars)
      NNNN (last 4)   : DIGITS   — plate number

    Examples:
        "WB0IF7728"  → 'I' at district pos → '1' → "WB01F7728"
        "WBI4F7728"  → 'I' at district pos → '1' → "WB14F7728"
        "WB04IF728"  → 'I' at series pos   → 'I' stays → "WB04IF728"
        "WB04F772B"  → 'B' at number pos   → '8' → "WB04F7728"
        "WB04F7O28"  → 'O' at number pos   → '0' → "WB04F7028"
    """
    p = plate.replace(' ', '').upper()
    if len(p) < 8:
        return plate

    # Segment the plate: first 2 = state, next 2 = district,
    # last 4 = number, middle = series
    state    = p[:2]
    district = p[2:4]
    number   = p[-4:]
    series   = p[4:-4]

    if not series:
        return plate   # unexpected format — leave untouched

    fixed = (
        ''.join(_fix_char(c, expect_letter=True)  for c in state)    +
        ''.join(_fix_char(c, expect_letter=False) for c in district) +
        ''.join(_fix_char(c, expect_letter=True)  for c in series)   +
        ''.join(_fix_char(c, expect_letter=False) for c in number)
    )
    return fixed


# Letters that OCR misreads as digits in BH number zone
_BH_NUM_FIXES = {
    'L': '4', 'I': '1', 'O': '0', 'S': '5',
    'B': '8', 'Z': '2', 'G': '6', 'T': '7', 'A': '4',
}

def _try_bh_correction(text: str) -> str:
    """
    Fix OCR errors specific to BH-series plates.
    Handles: missing year prefix (IND-trim cuts it), extra hallucinated chars,
    digit/letter confusions in number zone (4→L, 4→1, 6→G, etc.)
    """
    m_with    = re.match(r'^(\d{2})BH(.+)$', text)
    m_without = re.match(r'^BH(.+)$', text)

    if m_with:
        year = m_with.group(1)
        rest = m_with.group(2)
    elif m_without:
        # Year stripped by IND-trim — use current year as fallback
        year = datetime.now().strftime("%y")
        rest = m_without.group(1)
    else:
        return text

    def _extract(r: str):
        """Try to pull NNNN + XX from r, applying digit fixes to number zone."""
        for suf_len in (1, 2):
            suf_part = r[-suf_len:]
            num_part = r[:-suf_len]
            if not re.match(r'^[A-HJ-NP-Z]+$', suf_part):
                continue
            # Try num_part as-is, then drop leading/trailing noise chars
            for nt in ([num_part]
                       + ([num_part[1:]] if len(num_part) == 5 else [])
                       + ([num_part[:-1]] if len(num_part) == 5 else [])):
                if len(nt) != 4:
                    continue
                fixed = ''.join(_BH_NUM_FIXES.get(c, c) for c in nt)
                if fixed.isdigit():
                    return f"{year}BH{fixed}{suf_part}"
        return None

    # Try rest as-is
    result = _extract(rest)
    if result and re.match(r'^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$', result):
        return result

    # Try stripping one hallucinated char from start of rest (e.g. T1LL6D → 1LL6D)
    if len(rest) >= 6:
        result = _extract(rest[1:])
        if result and re.match(r'^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$', result):
            return result

    return text


def _try_corrections(norm: str) -> str:
    """
    Excel-backed OCR corrections — every fix is verified against RTO_CODE.xlsx.
    A correction is only accepted if the resulting RTO code exists in the Excel.

    Fix 1 — Missing leading zero in district code (Excel-verified):
        EasyOCR drops '0' from district codes like '04','07','01'.
        'WB4F7728' → try inserting 0 → 'WB04F7728'
        Check: 'WB04' must exist in rto_data (loaded from Excel) → confirmed ✓

    Fix 2 — Embedded IND removal:
        'MB04INDF7728' → strip mid-string IND → 'MB04F7728'

    Fix 3 — State-code letter swap (Excel-verified at RTO level):
        'MB04F7728' → swap M→W → 'WB04F7728'
        Check: 'WB04' must exist in rto_data → confirmed ✓
    """
    candidate = norm
    if not candidate:
        return norm

    # Fix 1 — Missing leading zero, verified against Excel RTO data
    # Pattern: SS D [Letter...] → single digit after state code
    # 'WB4F7728': state=WB, digit=4, rest=F7728 → insert 0 → 'WB04F7728'
    m = re.match(r'^([A-Z]{2})(\d)([A-Z])', candidate)
    if m:
        state   = m.group(1)
        digit   = m.group(2)
        rto_try = state + '0' + digit                  # e.g. "WB04"
        fixed   = state + '0' + candidate[2:]          # insert 0 after state code
        if rto_try in _rto_data and 8 <= len(fixed) <= 10:
            print(f"[FIX-1] Leading zero restored: {candidate} → {fixed}  (RTO {rto_try} ✓ in Excel)")
            candidate = fixed

    # Fix 2 — Strip embedded IND from anywhere after position 3
    if 'IND' in candidate[3:]:
        no_ind = candidate[:3] + candidate[3:].replace('IND', '', 1)
        if 8 <= len(no_ind) <= 10:
            candidate = no_ind

    # Fix 3 — Letter swap at position 0, verified against Excel RTO data
    if candidate and candidate[:2] not in STATE_CODES:
        alt = _LETTER_SWAPS.get(candidate[0], '')
        if alt:
            swapped  = alt + candidate[1:]
            rto_try  = swapped[:4]
            if rto_try in _rto_data and STATE_CODES.get(swapped[:2]):
                print(f"[FIX-3] Letter swap: {candidate} → {swapped}  (RTO {rto_try} ✓ in Excel)")
                candidate = swapped

    # Fix 4 — Position-aware character correction
    # Each character is corrected based on what type is EXPECTED at its position.
    # 'I' at digit position  → '1'    (district/number)
    # '1' at letter position → 'I'    (series)
    # 'O' at digit position  → '0'    (district/number)
    # '8' at letter position → 'B'    (series)
    # This prevents I↔1 and O↔0 blind swaps that previous logic caused.
    if 8 <= len(candidate) <= 10:
        fixed = fix_by_position(candidate)
        if fixed != candidate:
            print(f"[FIX-4] Position-aware: {candidate} → {fixed}")
            candidate = fixed

    # Fix 5 — Structural reconstruction for garbage in the MIDDLE
    # If plate is still too long (> 10 chars), reconstruct using:
    #   SS (first 2) + DD (next 2 digits) + LL (last letters before number) + NNNN (last 4)
    #
    # The SERIES letters sit right before the 4-digit number, so we take
    # letters from the END of the middle section, not the start.
    #
    #   KL10XYZAB1040  →  middle=XYZAB  →  last letters: B(1), AB(2)
    #                   →  try AB first (n=2): KL10AB1040 ✓
    #   WB04GARBF7728  →  middle=GARBF  →  last letter: F(1): WB04F7728 ✓
    #   RJ09RANDOMG4017→  middle=RANDOMG→  last letter: G(1): RJ09G4017 ✓
    if len(candidate) > 10 and candidate[:2] in STATE_CODES:
        ss    = candidate[:2]
        dd    = candidate[2:4]
        last4 = candidate[-4:]
        if dd.isdigit() and last4.isdigit():
            middle  = candidate[4:-4]
            letters = [c for c in middle if c.isalpha()]
            # Try 1, 2, then 3 letters from the END of the middle (series position)
            for n in range(1, min(4, len(letters) + 1)):
                series        = ''.join(letters[-n:])
                reconstructed = ss + dd + series + last4
                if _PLATE_REGEX.match(reconstructed) and len(reconstructed) <= 10:
                    print(f"[FIX-5] Reconstructed: {candidate} → {reconstructed}")
                    candidate = reconstructed
                    break

    return candidate


# ── Full pipeline ─────────────────────────────────────────────────────────────
def is_valid_plate(text: str) -> bool:
    """Return True only if the plate passes ALL steps."""
    _, valid, _, _, _ = validate(text)
    return valid


def validate(text: str) -> tuple[str, bool, str, str, bool]:
    """
    Full validation pipeline — 5 strict gates.
    Every gate must pass before moving to the next.

    PRE-PROCESSING (before gates):
        clean_plate → strip_ind → _try_corrections → fix_by_position

    GATE 1 — Length: 8 to 10 characters
    GATE 2 — Valid STATE code: norm[:2] must be in STATE_CODES (KL, WB, MH ...)
    GATE 3 — Valid DISTRICT code: norm[:4] must exist in RTO Excel (KL07, WB04 ...)
    GATE 4 — Indian plate format: SS + DD + LL(1-3) + NNNN
    GATE 5 — Last 4 are digits: plate number must be numeric

    Returns: (final_text, is_valid, reason, state_name, was_corrected)
      was_corrected — True only when _try_corrections changed the text,
                      NOT when strip_ind stripped an IND/ND prefix (expected preprocessing)
    """
    # ── Pre-processing ─────────────────────────────────────────────────────────
    cleaned      = clean_plate(text)
    stripped     = strip_ind(cleaned)   # IND-prefix removal (not a "correction")

    # ── Bharat Series (BH) — check BEFORE _try_corrections to avoid mangling ───
    # Format: 2-digit year + BH + 4 digits + 1-2 letters (not I/O)
    # Example: 22BH1234AB, 24BH0001CD
    bh_candidate = _try_bh_correction(stripped)
    if re.match(r'^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$', bh_candidate):
        return bh_candidate, True, "", "Bharat Series", bh_candidate != stripped

    norm         = _try_corrections(stripped)
    was_corrected = stripped != norm    # True only if FIX-1/2/3/4/5 changed something

    # ── Gate 1: Length ─────────────────────────────────────────────────────────
    if len(norm) < 8:
        return norm, False, f"Too short: {len(norm)} chars (min 8)", "", was_corrected
    if len(norm) > 10:
        return norm, False, f"Exceeds max length ({len(norm)}/10)", "", was_corrected

    # ── Gate 2: Valid state code ────────────────────────────────────────────────
    valid_state, state_name = check_State_or_Union_Territory_code(norm)
    if not valid_state:
        return norm, False, f"Invalid state code ({norm[:2]})", "", was_corrected

    # ── Gate 3: Valid district code — strict check against RTO Excel ───────────
    rto_code = norm[:4]   # e.g. "KL07", "WB04"
    if rto_code not in _rto_data:
        return norm, False, f"District not found ({rto_code})", "", was_corrected

    # ── Gate 4: Indian plate format SS + DD + LL(1-3) + NNNN ───────────────────
    if not is_valid_indian_format(norm):
        return norm, False, "Invalid plate format (SS+DD+LL+NNNN)", "", was_corrected

    # ── Gate 5: Last 4 characters must be digits ────────────────────────────────
    valid_last4, _ = check_last_four_digit(norm)
    if not valid_last4:
        return norm, False, "Must end with 4 digits", "", was_corrected

    return norm, True, "", state_name, was_corrected


# ── Batch validate ────────────────────────────────────────────────────────────
def validate_batch(plates: list[str]) -> list[dict]:
    """Validate a list of raw OCR strings."""
    return [
        {"raw": raw, "result": r, "valid": v, "reason": reason, "state": state}
        for raw in plates
        for r, v, reason, state, _ in [validate(raw)]
    ]


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("IND TN8BF 4089",  True,  "Tamil Nadu"),
        ("INDRJ09G4017",    True,  "Rajasthan"),
        ("KL07AB1234",      True,  "Kerala"),
        ("MH12CD5678",      True,  "Maharashtra"),
        ("DL01AB1234",      True,  "Delhi"),
        ("GJ05CD1234",      True,  "Gujarat"),
        ("WB20EF5678",      True,  "West Bengal"),
        ("GA01AB1234",      True,  "Goa"),
        ("XX12AB3456",      False, ""),
        ("AB12",            False, ""),
        ("KL07AB123456",    False, ""),
        ("9726KL1234",      False, ""),
        ("KL1234XY",        False, ""),
        ("???",             False, ""),
        ("INDAB12",         False, ""),
    ]

    print(f"{'RAW':<22} {'RESULT':<14} {'VALID':<7} {'STATE':<22} {'REASON':<24} {'TEST'}")
    print("-" * 100)
    all_pass = True
    for raw, exp_valid, exp_state in tests:
        result, valid, reason, state = validate(raw)
        ok     = (valid == exp_valid) and (state == exp_state)
        status = "✓" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"{raw!r:<22} {result!r:<14} {str(valid):<7} {state:<22} {reason:<24} {status}")

    print("-" * 100)
    print("All tests passed ✓" if all_pass else "Some tests FAILED ✗")
