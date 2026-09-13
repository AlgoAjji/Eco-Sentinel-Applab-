"""
Real-Time Avian Bio-Acoustic Edge Monitoring Station
BirdNET v2.4 ONNX inference for Arduino UNO Q

Version: 4.0

Main fixes:
1. Automatically detects and quarantines the supplied HTML/corrupt ONNX file.
2. Downloads the ARM-friendly BirdNET v2.4 INT8 ONNX model when needed.
3. Downloads the exact matching 6,522-label file.
4. Validates ONNX input/output contract before starting audio.
5. Validates that labels are in BirdNET "scientific_Common name" format.
6. Waits for a complete 3-second audio window before the first inference.
7. Uses 50% overlapping 3-second windows.
8. Keeps inference on the audio loop and heavy feature extraction on a worker.
9. Sends species, confidence, battery and timestamp to NodeMCU.
10. Logs detections to SQLite and automatically exports detections.csv.
11. Saves a mel spectrogram and acoustic features for every accepted detection.
12. Includes startup diagnostics and clearer failure messages.

BirdNET v2.4 contract:
- 48,000 Hz
- 3 seconds
- 144,000 float32 samples
- 6,522 output classes
- raw logits -> sigmoid confidence

Keep birdnet_model.onnx and birdnet_labels.txt in the same directory as this file.
"""

import csv
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import requests
import sqlite3
from static_ffmpeg import run as ffmpeg_run

# UNO Q Linux MPU -> STM32 MCU Bridge for onboard LED matrix.
try:
    from arduino.app_utils import Bridge
    MATRIX_BRIDGE_AVAILABLE = True
except Exception as exc:
    Bridge = None
    MATRIX_BRIDGE_AVAILABLE = False
    print(f"[MATRIX] Bridge unavailable; monitoring continues: {exc}")


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "birdnet_model.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "birdnet_labels.txt")
DB_PATH = os.path.join(BASE_DIR, "birds.db")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "spectrograms")
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# Officially documented BirdNET v2.4 ONNX source used for this project.
# INT8 ARM is the preferred variant for a low-RAM ARM edge computer.
MODEL_URL = (
    "https://huggingface.co/tphakala/BirdNET-v2.4/"
    "resolve/main/BirdNET_v2.4_int8_arm.onnx"
)
LABELS_URL = (
    "https://huggingface.co/tphakala/BirdNET-v2.4/"
    "resolve/main/labels.txt"
)

# Field network from the supplied project configuration.
COOLPAD_IP = "My_Phone_IP" #change here
NODEMCU_IP = "MY_NodeMCU_IP" #change here

# Indian Standard Time (IST = UTC+05:30).
IST_OFFSET_SECONDS = 19800
INDIA_NTP_SERVERS = (
    "time.google.com",
    "pool.ntp.org",
    "time.cloudflare.com",
)
TIME_SYNC_INTERVAL_SECONDS = 6 * 60 * 60
AUDIO_STREAM = f"http://{COOLPAD_IP}/audio.wav"
TELEMETRY_URL = f"{NODEMCU_IP}/telemetry"

# BirdNET v2.4 audio contract.
SAMPLE_RATE = 48000
CHUNK_LEN = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_LEN)
HOP_SAMPLES = CHUNK_SAMPLES // 2  # 50% overlap

# Detection settings.
# Low-SNR mode: allow weak candidates, then require repeated evidence.
CANDIDATE_CONFIDENCE_THRESHOLD = 0.05
CONFIRM_CONFIDENCE_THRESHOLD = 0.05
RMS_SILENCE_THRESHOLD = 0.01
BATTERY_POLL_SECONDS = 30

EVIDENCE_WINDOW_SECONDS = 15.0
EVIDENCE_MIN_HITS = 2
EVIDENCE_MIN_TOTAL_SCORE = 0.18
CANDIDATE_DECAY_SECONDS = 4.5
TOP_K_CANDIDATES = 5

CONFIDENCE_THRESHOLD = CANDIDATE_CONFIDENCE_THRESHOLD

# Prevent the same continuous call from filling the database every 1.5 s.
# Set to 0 to log every accepted inference.
DUPLICATE_SUPPRESSION_SECONDS = 10

# How often a "Scanning..." heartbeat is sent to the NodeMCU.
HEARTBEAT_SECONDS = 12

# Matrix control is asynchronous so Bridge/RPC cannot block BirdNET inference.
_matrix_q = queue.Queue(maxsize=4)
_matrix_last_state = None


def request_matrix_state(state):
    global _matrix_last_state
    state = str(state)

    if state == _matrix_last_state:
        return

    _matrix_last_state = state
    try:
        _matrix_q.put_nowait(state)
    except queue.Full:
        try:
            _matrix_q.get_nowait()
        except queue.Empty:
            pass
        try:
            _matrix_q.put_nowait(state)
        except queue.Full:
            pass


def _matrix_worker():
    while True:
        state = _matrix_q.get()
        try:
            if MATRIX_BRIDGE_AVAILABLE and Bridge is not None:
                Bridge.call("set_bird_matrix_state", state)
                print(f"[MATRIX] State -> {state}")
        except Exception as exc:
            print(f"[MATRIX] Bridge call failed ({state}): {exc}")
        finally:
            _matrix_q.task_done()


threading.Thread(
    target=_matrix_worker,
    name="matrix-worker",
    daemon=True,
).start()

# ============================================================================
# Bengaluru project whitelist
# ============================================================================
# These are the common-name strings used after parsing birdnet_labels.txt.
# The whitelist is intentionally kept from the supplied project rather than
# silently replacing it with an external species list.
NATIVE_WHITELIST = {
    "House Sparrow",
    "Common Myna",
    "Rose-ringed Parakeet",
    "Black Kite",
    "Asian Koel",
    "Jungle Myna",
    "Brahminy Kite",
    "Rock Pigeon",
    "Greater Coucal",
    "White-throated Kingfisher",
    "Spotted Dove",
    "Indian Pond Heron",
    "Red-whiskered Bulbul",
    "Red-vented Bulbul",
    "Purple Sunbird",
    "Coppersmith Barbet",
    "Indian Robin",
    "House Crow",
}


# ============================================================================
# 1. UTILITY FUNCTIONS
# ============================================================================

_time_lock = threading.Lock()
_time_offset = None
_last_time_sync = 0.0


def sync_india_time(force=False):
    """Get network time using NTP and maintain the offset to the UNO Q clock."""
    global _time_offset, _last_time_sync

    now = time.time()
    with _time_lock:
        if (
            not force
            and _time_offset is not None
            and now - _last_time_sync < TIME_SYNC_INTERVAL_SECONDS
        ):
            return True

    import socket
    import struct

    packet = b"\x1b" + 47 * b"\0"

    for host in INDIA_NTP_SERVERS:
        try:
            t0 = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(2.0)
                sock.sendto(packet, (host, 123))
                data, _ = sock.recvfrom(512)
            t1 = time.time()

            if len(data) < 48:
                continue

            words = struct.unpack("!12I", data[:48])
            ntp_seconds = words[10]
            ntp_fraction = words[11]
            ntp_time = (
                ntp_seconds - 2208988800
                + ntp_fraction / 4294967296.0
            )

            offset = ntp_time - ((t0 + t1) / 2.0)

            with _time_lock:
                _time_offset = offset
                _last_time_sync = t1

            print(
                f"[TIME] Network time synchronized via {host}; "
                f"offset={offset:+.3f}s; timezone=IST (UTC+05:30)"
            )
            return True

        except Exception as exc:
            print(f"[TIME] NTP server {host} unavailable: {exc}")

    print("[TIME] Network time sync failed; using existing UNO Q clock.")
    return False


def now_local_iso():
    with _time_lock:
        offset = _time_offset

    current = time.time()
    if offset is not None:
        current += offset

    # UTC -> IST.
    current += IST_OFFSET_SECONDS

    current_dt = dt.datetime.fromtimestamp(
        current,
        dt.timezone.utc,
    )
    return current_dt.isoformat(timespec="seconds")


def _time_sync_worker():
    while True:
        time.sleep(TIME_SYNC_INTERVAL_SECONDS)
        sync_india_time(force=True)



def safe_species_filename(species):
    """Make a filesystem-safe species name."""
    allowed = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789-_")
    return "".join(c if c in allowed else "_" for c in species)


def is_probably_html(path):
    """Detect the exact failure seen in the supplied ONNX file."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).lstrip().lower()
        return (
            head.startswith(b"<!doctype html")
            or head.startswith(b"<html")
            or b"<html" in head[:200]
            or b"<head" in head[:200]
        )
    except OSError:
        return False


def quarantine_bad_file(path):
    """Rename a corrupt download instead of silently deleting it."""
    if not os.path.exists(path):
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{path}.invalid_{stamp}"

    try:
        os.replace(path, backup)
        print(f"[SETUP] Invalid file moved to: {os.path.basename(backup)}")
    except OSError as exc:
        raise RuntimeError(
            f"Could not quarantine invalid file {path}: {exc}"
        ) from exc


def download_atomic(url, path, what, min_bytes):
    """
    Download to a temporary file and replace the target only after the
    download passes basic sanity checks.
    """
    tmp_path = path + ".download"

    print(f"[SETUP] Downloading {what}...")
    print(f"[SETUP] URL: {url}")

    try:
        with requests.get(
            url,
            allow_redirects=True,
            timeout=(10, 180),
            stream=True,
            headers={"User-Agent": "Avian-BioAcoustic-Edge/3.0"},
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            total = 0

            with open(tmp_path, "wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    out.write(chunk)
                    total += len(chunk)

        if total < min_bytes:
            raise RuntimeError(
                f"{what} download is too small: {total} bytes."
            )

        if is_probably_html(tmp_path):
            raise RuntimeError(
                f"{what} download is HTML instead of the real file. "
                f"Content-Type was {content_type!r}."
            )

        os.replace(tmp_path, path)
        print(f"[SETUP] {what} downloaded: {total / 1e6:.2f} MB")

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def ensure_labels():
    """Download labels only when absent or obviously invalid."""
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) >= 6000:
                print(f"[SETUP] Existing labels found: {len(lines)} lines")
                return

            print("[SETUP] Existing labels file is unexpectedly small.")
        except (OSError, UnicodeError) as exc:
            print(f"[SETUP] Existing labels file cannot be read: {exc}")

        quarantine_bad_file(LABELS_PATH)

    download_atomic(
        LABELS_URL,
        LABELS_PATH,
        "BirdNET labels",
        min_bytes=100_000,
    )


def load_labels():
    """
    Parse the exact BirdNET labels format:
        Scientific name_Common name

    Return both full labels and common names.
    """
    with open(LABELS_PATH, encoding="utf-8") as f:
        raw_labels = [line.strip() for line in f if line.strip()]

    common_names = []
    for line in raw_labels:
        if "_" not in line:
            raise RuntimeError(
                f"Invalid label format: {line!r}. "
                "Expected 'Scientific name_Common name'."
            )
        common_names.append(line.rsplit("_", 1)[1].strip())

    if len(raw_labels) != 6522:
        raise RuntimeError(
            f"Expected 6,522 BirdNET v2.4 labels, found {len(raw_labels)}. "
            "Do not mix labels from another BirdNET model."
        )

    return raw_labels, common_names


# ============================================================================
# 2. MODEL VALIDATION AND LOADING
# ============================================================================

EXPECTED_INPUT_SAMPLES = 144000
EXPECTED_CLASSES = 6522


def shape_is_compatible(shape, expected_last_dim):
    """
    Accept [144000], [None,144000], [1,144000], etc., but require the
    final fixed dimension to match BirdNET v2.4.
    """
    if not shape:
        return False
    last = shape[-1]
    return isinstance(last, int) and last == expected_last_dim


def create_ort_session(path):
    """Create an optimized CPU ONNX Runtime session."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3

    return ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def validate_model_contract(session):
    """Validate BirdNET v2.4 input/output dimensions."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if not inputs:
        raise RuntimeError("ONNX model has no input tensors.")
    if not outputs:
        raise RuntimeError("ONNX model has no output tensors.")

    model_input = inputs[0]
    model_output = outputs[0]

    print(
        f"[SETUP] ONNX input:  name={model_input.name!r}, "
        f"type={model_input.type}, shape={model_input.shape}"
    )
    print(
        f"[SETUP] ONNX output: name={model_output.name!r}, "
        f"type={model_output.type}, shape={model_output.shape}"
    )

    if model_input.type != "tensor(float)":
        raise RuntimeError(
            f"Unexpected model input type: {model_input.type}. "
            "BirdNET v2.4 should use float32."
        )

    if not shape_is_compatible(model_input.shape, EXPECTED_INPUT_SAMPLES):
        raise RuntimeError(
            f"Unexpected BirdNET input shape: {model_input.shape}. "
            "Expected a final dimension of 144000 samples."
        )

    if not shape_is_compatible(model_output.shape, EXPECTED_CLASSES):
        raise RuntimeError(
            f"Unexpected BirdNET output shape: {model_output.shape}. "
            "Expected a final dimension of 6522 classes."
        )

    return model_input.name, model_output.name


def load_validated_model():
    """
    Load the local model.

    If the supplied model is HTML/corrupt or cannot be parsed, quarantine it,
    download the known BirdNET v2.4 ARM model, then validate again.
    """
    ensure_labels()
    raw_labels, labels = load_labels()

    if not os.path.exists(MODEL_PATH):
        download_atomic(
            MODEL_URL,
            MODEL_PATH,
            "BirdNET v2.4 INT8 ARM ONNX model",
            min_bytes=20_000_000,
        )

    if is_probably_html(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 20_000_000:
        print("[SETUP] Local birdnet_model.onnx is not a real BirdNET model.")
        quarantine_bad_file(MODEL_PATH)
        download_atomic(
            MODEL_URL,
            MODEL_PATH,
            "BirdNET v2.4 INT8 ARM ONNX model",
            min_bytes=20_000_000,
        )

    try:
        print(
            f"[SETUP] Loading ONNX model "
            f"({os.path.getsize(MODEL_PATH) / 1e6:.2f} MB)..."
        )
        session = create_ort_session(MODEL_PATH)
        input_name, output_name = validate_model_contract(session)

    except Exception as first_error:
        print(f"[SETUP] Local ONNX load failed: {first_error}")
        print("[SETUP] The local model will be quarantined and re-downloaded.")

        quarantine_bad_file(MODEL_PATH)

        download_atomic(
            MODEL_URL,
            MODEL_PATH,
            "BirdNET v2.4 INT8 ARM ONNX model",
            min_bytes=20_000_000,
        )

        try:
            session = create_ort_session(MODEL_PATH)
            input_name, output_name = validate_model_contract(session)
        except Exception as second_error:
            raise RuntimeError(
                "BirdNET ONNX model could not be loaded even after a fresh "
                f"download.\nOriginal error: {first_error}\n"
                f"Fresh-download error: {second_error}"
            ) from second_error

    if len(raw_labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Labels contain {len(raw_labels)} entries, but BirdNET v2.4 "
            f"requires {EXPECTED_CLASSES}."
        )

    print(
        f"[SETUP] BirdNET model validated successfully: "
        f"{EXPECTED_CLASSES} classes, 48 kHz, 3 seconds."
    )
    print(
        f"[SETUP] Label examples: {raw_labels[:3]}"
    )

    return session, input_name, output_name, raw_labels, labels


session, input_name, output_name, raw_labels, labels = load_validated_model()

request_matrix_state("listening")


# ============================================================================
# 3. BACKGROUND TELEMETRY
# ============================================================================

_telemetry_q = queue.Queue(maxsize=20)


def _telemetry_worker():
    while True:
        payload = _telemetry_q.get()
        try:
            response = requests.post(
                TELEMETRY_URL,
                json=payload,
                timeout=2,
            )
            if not response.ok:
                print(
                    f"[TELEMETRY] NodeMCU HTTP {response.status_code}"
                )
        except requests.RequestException as exc:
            # Do not stop bird detection because the display is offline.
            print(f"[TELEMETRY] NodeMCU unavailable: {exc}")
        finally:
            _telemetry_q.task_done()


threading.Thread(
    target=_telemetry_worker,
    name="telemetry-worker",
    daemon=True,
).start()


def send_telemetry(species, confidence, battery):
    payload = {
        "species": species,
        "confidence": round(float(confidence) * 100.0, 1),
        "battery": battery,
        "timestamp": now_local_iso(),
        "timezone": "Asia/Kolkata",
    }

    try:
        _telemetry_q.put_nowait(payload)
    except queue.Full:
        pass


def _telemetry_heartbeat_worker():
    """
    Keep the NodeMCU informed that the UNO Q Python process is alive.

    This is deliberately independent of FFmpeg/BirdNET. Otherwise, if the
    audio stream stalls, the NodeMCU can incorrectly conclude that the UNO Q
    itself is offline even though Python is still running normally.
    """
    # Send one immediately after the telemetry worker has started.
    send_telemetry("UNO Q Online", 0.0, _battery_level)

    while True:
        time.sleep(10)
        send_telemetry("UNO Q Online", 0.0, _battery_level)


# ============================================================================
# 4. COOLPAD BATTERY POLLING
# ============================================================================

_battery_level = "N/A"

threading.Thread(
    target=_telemetry_heartbeat_worker,
    name="telemetry-heartbeat",
    daemon=True,
).start()


def deep_scan_json(data, target_keys):
    """Find battery-like fields anywhere in a nested JSON response."""
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = str(key).lower()

            if any(target in key_lower for target in target_keys):
                if isinstance(value, (int, float, str)):
                    return value

                if isinstance(value, dict) and "data" in value:
                    return value["data"]

            found = deep_scan_json(value, target_keys)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = deep_scan_json(item, target_keys)
            if found is not None:
                return found

    return None


def _battery_worker():
    global _battery_level

    keys = ["battery", "percent", "capacity", "charge"]

    while True:
        value = None

        for endpoint in ("status.json", "sensors.json"):
            try:
                response = requests.get(
                    f"http://{COOLPAD_IP}/{endpoint}",
                    timeout=2,
                )

                if response.ok:
                    value = deep_scan_json(response.json(), keys)
                    if value is not None:
                        break

            except (
                requests.RequestException,
                ValueError,
            ):
                continue

        if value is not None:
            try:
                battery = int(round(float(value)))
                _battery_level = max(0, min(100, battery))
            except (TypeError, ValueError):
                pass

        time.sleep(BATTERY_POLL_SECONDS)


threading.Thread(
    target=_battery_worker,
    name="battery-worker",
    daemon=True,
).start()


# ============================================================================
# 5. SQLITE + CSV DATA STORAGE
# ============================================================================

DB_LOCK = threading.Lock()


def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY,
                ts TEXT,
                species TEXT,
                confidence REAL,
                battery INTEGER,
                rms REAL,
                detection_status TEXT,
                whitelisted INTEGER,
                spectrogram_path TEXT,
                features_json TEXT
            )
        """)

        # Migration for older project databases.  The original database may
        # already contain an inventory table but be missing one or more of
        # the columns used by the current application.
        migrations = [
            ("ts", "TEXT"),
            ("species", "TEXT"),
            ("confidence", "REAL"),
            ("battery", "INTEGER"),
            ("rms", "REAL"),
            ("detection_status", "TEXT"),
            ("whitelisted", "INTEGER"),
            ("spectrogram_path", "TEXT"),
            ("features_json", "TEXT"),
        ]

        for column, column_type in migrations:
            try:
                conn.execute(
                    f"ALTER TABLE inventory ADD COLUMN {column} {column_type}"
                )
                print(f"[DB] Added missing column: {column}")
            except sqlite3.OperationalError:
                # Column already exists; this is expected on normal starts.
                pass

        conn.commit()
        conn.close()


def log_detection(
    species,
    confidence,
    battery,
    rms,
    detection_status="CONFIRMED",
    whitelisted=False,
):
    """
    Store EVERY meaningful BirdNET event, including rejected/false-positive
    candidates.

    detection_status examples:
      CONFIRMED               -> temporal evidence accepted it
      LOW_CONFIDENCE          -> below the 5% candidate threshold
      NON_WHITELISTED         -> BirdNET predicted a species outside our list
      CANDIDATE               -> weak whitelisted candidate still accumulating
                                  evidence
    """
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)

        cursor = conn.execute(
            """
            INSERT INTO inventory
                (
                    ts,
                    species,
                    confidence,
                    battery,
                    rms,
                    detection_status,
                    whitelisted
                )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_local_iso(),
                species,
                float(confidence),
                battery if isinstance(battery, int) else None,
                float(rms),
                detection_status,
                1 if whitelisted else 0,
            ),
        )

        conn.commit()
        row_id = cursor.lastrowid
        conn.close()

    return row_id

def log_features(row_id, spectrogram_path, features_json):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)

        conn.execute(
            """
            UPDATE inventory
            SET spectrogram_path = ?, features_json = ?
            WHERE id = ?
            """,
            (
                spectrogram_path,
                features_json,
                row_id,
            ),
        )

        conn.commit()
        conn.close()


def export_csv():
    """
    Export the complete SQLite inventory into a normal CSV file so the data
    can be opened in Excel, R Studio, Minitab, etc.
    """
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)

        cursor = conn.execute(
            """
            SELECT
                id,
                ts,
                species,
                confidence,
                battery,
                rms,
                detection_status,
                whitelisted,
                spectrogram_path,
                features_json
            FROM inventory
            ORDER BY id
            """
        )

        rows = cursor.fetchall()
        headers = [
            "id",
            "timestamp",
            "species",
            "confidence",
            "battery",
            "rms",
            "detection_status",
            "whitelisted",
            "spectrogram_path",
            "features_json",
        ]

        tmp_csv = CSV_PATH + ".tmp"

        with open(
            tmp_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        os.replace(tmp_csv, CSV_PATH)
        conn.close()


init_db()
export_csv()


# ============================================================================
# 6. ACOUSTIC FEATURES + SPECTROGRAMS
# ============================================================================

def extract_features(audio, sr):
    """
    Scalar acoustic features for the future R Studio / Minitab phase.

    The complete feature vector is stored as JSON so additional fields can be
    added later without changing the database schema.
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    rms = librosa.feature.rms(
        y=audio,
    )

    flatness = librosa.feature.spectral_flatness(
        y=audio,
    )

    return {
        "mfcc_mean": [float(x) for x in mfccs.mean(axis=1)],
        "mfcc_std": [float(x) for x in mfccs.std(axis=1)],
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_bandwidth_mean": float(bandwidth.mean()),
        "spectral_rolloff_mean": float(rolloff.mean()),
        "zero_crossing_rate_mean": float(zcr.mean()),
        "rms_mean": float(rms.mean()),
        "spectral_flatness_mean": float(flatness.mean()),
    }


def save_spectrogram(audio, sr, out_path):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        fmax=sr // 2,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
    )

    fig.tight_layout(pad=0)

    fig.savefig(
        out_path,
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)


_feature_q = queue.Queue(maxsize=20)


def _feature_worker():
    while True:
        audio_copy, species, row_id = _feature_q.get()

        try:
            features = extract_features(
                audio_copy,
                SAMPLE_RATE,
            )

            filename = (
                f"{row_id}_"
                f"{safe_species_filename(species)}.png"
            )

            spectrogram_path = os.path.join(
                SPECTROGRAM_DIR,
                filename,
            )

            save_spectrogram(
                audio_copy,
                SAMPLE_RATE,
                spectrogram_path,
            )

            log_features(
                row_id,
                spectrogram_path,
                json.dumps(features),
            )

            # Keep the CSV current after feature extraction completes.
            export_csv()

            print(
                f"[FEATURES] Saved spectrogram/features for "
                f"row {row_id}"
            )

        except Exception as exc:
            print(
                f"[FEATURES] Extraction failed for row "
                f"{row_id}: {exc}"
            )

        finally:
            _feature_q.task_done()


threading.Thread(
    target=_feature_worker,
    name="feature-worker",
    daemon=True,
).start()


# ============================================================================
# 7. BIRDNET INFERENCE
# ============================================================================

def sigmoid(values):
    """
    Numerically stable sigmoid for model logits.
    """
    values = np.asarray(values, dtype=np.float32)

    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)

    result[positive] = 1.0 / (
        1.0 + np.exp(-values[positive])
    )

    exp_x = np.exp(values[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)

    return result


def run_birdnet(audio):
    """
    Run one complete 3-second BirdNET inference.

    Returns:
        species_name, confidence, top_index
    """
    if audio.shape != (CHUNK_SAMPLES,):
        raise ValueError(
            f"Audio buffer has shape {audio.shape}; "
            f"expected ({CHUNK_SAMPLES},)."
        )

    input_tensor = np.asarray(
        audio,
        dtype=np.float32,
    )[np.newaxis, :]

    logits = session.run(
        [output_name],
        {input_name: input_tensor},
    )[0]

    logits = np.asarray(
        logits,
        dtype=np.float32,
    ).reshape(-1)

    if logits.size != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Model returned {logits.size} outputs; "
            f"expected {EXPECTED_CLASSES}."
        )

    scores = sigmoid(logits)

    top_index = int(np.argmax(scores))
    confidence = float(scores[top_index])

    species_name = labels[top_index]

    k = min(TOP_K_CANDIDATES, scores.size)
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    top_candidates = [
        (labels[int(idx)], float(scores[int(idx)]), int(idx))
        for idx in top_indices
    ]

    return species_name, confidence, top_index, top_candidates


# ============================================================================
# 8. FFmpeg AUDIO STREAM
# ============================================================================

ffmpeg_path, _ = (
    ffmpeg_run.get_or_fetch_platform_executables_else_raise()
)

ffmpeg_cmd = [
    ffmpeg_path,
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    AUDIO_STREAM,
    "-f",
    "f32le",
    "-ac",
    "1",
    "-ar",
    str(SAMPLE_RATE),
    "-",
]


def start_ffmpeg():
    return subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def read_exact_stdout(process, byte_count):
    """
    Read exactly byte_count bytes from FFmpeg stdout unless FFmpeg exits.

    A live pipe can legally return fewer bytes than requested even though the
    producer is still running. The previous code treated that normal partial
    read as a stream failure, causing the repeated:
        "Stream returned 17764 bytes; reconnecting..."
    loop.

    This function accumulates partial reads until the requested amount is
    available or the FFmpeg process reaches EOF.
    """
    if process is None or process.stdout is None:
        return b""

    chunks = []
    total = 0

    while total < byte_count:
        if process.poll() is not None:
            try:
                tail = process.stdout.read(byte_count - total)
            except Exception:
                tail = b""

            if tail:
                chunks.append(tail)
                total += len(tail)
            break

        try:
            piece = process.stdout.read(byte_count - total)
        except Exception:
            break

        if not piece:
            # No bytes were returned. Give the producer a tiny amount of time
            # before checking again, while still allowing EOF/process exit.
            time.sleep(0.01)
            continue

        chunks.append(piece)
        total += len(piece)

    return b"".join(chunks)


def stop_ffmpeg(process):
    if process is None:
        return

    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# ============================================================================
# LOW-SNR TEMPORAL EVIDENCE
# ============================================================================

candidate_history = {}


def update_candidate_evidence(species, confidence, timestamp):
    state = candidate_history.setdefault(
        species,
        {"events": [], "last_seen": timestamp},
    )

    state["events"].append((timestamp, float(confidence)))
    state["last_seen"] = timestamp

    cutoff = timestamp - EVIDENCE_WINDOW_SECONDS
    state["events"] = [
        (ts, score)
        for ts, score in state["events"]
        if ts >= cutoff
    ]

    hits = len(state["events"])
    total_score = sum(score for _, score in state["events"])
    peak_score = max(score for _, score in state["events"])

    return hits, total_score, peak_score


def clear_candidate_evidence(species):
    candidate_history.pop(species, None)


# ============================================================================
# 9. MAIN REAL-TIME LOOP
# ============================================================================

print("")
print("=" * 72)
print("REAL-TIME AVIAN BIO-ACOUSTIC EDGE MONITOR")
print("=" * 72)
print(f"[SYSTEM] Coolpad audio : {AUDIO_STREAM}")
print(f"[SYSTEM] NodeMCU       : {TELEMETRY_URL}")
print(f"[SYSTEM] Model         : {MODEL_PATH}")
print(f"[SYSTEM] Database      : {DB_PATH}")
print(f"[SYSTEM] CSV           : {CSV_PATH} (includes raw/false-positive events)")
print(
    f"[SYSTEM] Candidate threshold: "
    f"{CANDIDATE_CONFIDENCE_THRESHOLD:.2f} (5%)"
)
print(
    f"[SYSTEM] Evidence rule    : "
    f"{EVIDENCE_MIN_HITS} hits / {EVIDENCE_WINDOW_SECONDS:.0f}s, "
    f"sum >= {EVIDENCE_MIN_TOTAL_SCORE:.2f}"
)
print(f"[SYSTEM] RMS threshold: {RMS_SILENCE_THRESHOLD:.4f}")
print("=" * 72)

process = None
last_heartbeat = 0.0
last_detection_by_species = {}

# Start with an empty buffer and explicitly count samples.
# This fixes the old behavior where inference could run before a complete
# 3-second window had been received.
audio_buffer = np.zeros(
    CHUNK_SAMPLES,
    dtype=np.float32,
)

samples_filled = 0


try:
    while True:

        if process is None or process.poll() is not None:
            print("[AUDIO] Starting/restarting FFmpeg audio stream...")
            stop_ffmpeg(process)
            process = start_ffmpeg()
            samples_filled = 0
            audio_buffer.fill(0)

        bytes_needed = HOP_SAMPLES * 4

        # Accumulate partial pipe reads instead of treating them as an FFmpeg
        # failure. This is the critical fix for the repeated 17,764-byte loop.
        raw = read_exact_stdout(process, bytes_needed)

        if len(raw) != bytes_needed:
            print(
                f"[AUDIO] FFmpeg ended/returned an incomplete block: "
                f"{len(raw)}/{bytes_needed} bytes; reconnecting..."
            )
            stop_ffmpeg(process)
            process = None
            time.sleep(1)
            continue

        new_samples = np.frombuffer(
            raw,
            dtype=np.float32,
        )

        # Shift old samples left and append new samples.
        audio_buffer[:-HOP_SAMPLES] = audio_buffer[HOP_SAMPLES:]
        audio_buffer[-HOP_SAMPLES:] = new_samples

        samples_filled = min(
            CHUNK_SAMPLES,
            samples_filled + HOP_SAMPLES,
        )

        # Do not run BirdNET until a full 3-second window exists.
        if samples_filled < CHUNK_SAMPLES:
            if time.time() - last_heartbeat > HEARTBEAT_SECONDS:
                print(
                    f"[AUDIO] Buffering: "
                    f"{samples_filled}/{CHUNK_SAMPLES} samples"
                )
                last_heartbeat = time.time()
            continue

        audio = audio_buffer.copy()

        # Remove NaN/Inf values if the audio source ever emits them.
        if not np.isfinite(audio).all():
            print("[AUDIO] Invalid numeric samples detected; skipping.")
            continue

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio.astype(np.float64)
                    )
                )
            )
        )

        if rms < RMS_SILENCE_THRESHOLD:
            if time.time() - last_heartbeat > HEARTBEAT_SECONDS:
                print(
                    f"[SCANNING] Near silence | "
                    f"RMS={rms:.6f} | battery={_battery_level}%"
                )

                send_telemetry(
                    "Scanning...",
                    0.0,
                    _battery_level,
                )

                last_heartbeat = time.time()

            continue

        try:
            (
                species_name,
                confidence,
                top_index,
                top_candidates,
            ) = run_birdnet(audio)

        except Exception as exc:
            print(f"[INFERENCE] BirdNET inference failed: {exc}")
            time.sleep(0.5)
            continue

        # LOW-SNR MODE
        # Every non-silent BirdNET window is now written to SQLite/CSV.
        # This intentionally includes false positives and rejected classes,
        # making detections.csv useful for ROC/threshold analysis in R/Minitab.

        top1_is_whitelisted = species_name in NATIVE_WHITELIST

        if confidence < CANDIDATE_CONFIDENCE_THRESHOLD:
            top1_status = "LOW_CONFIDENCE"
        elif not top1_is_whitelisted:
            top1_status = "NON_WHITELISTED"
        else:
            top1_status = "CANDIDATE"

        # Log the top-1 BirdNET result immediately, regardless of acceptance.
        event_row_id = log_detection(
            species_name,
            confidence,
            _battery_level,
            rms,
            detection_status=top1_status,
            whitelisted=top1_is_whitelisted,
        )

        # Give whitelisted candidates temporal evidence.
        confirmed_candidate = None
        now_ts = time.time()

        for candidate_species, candidate_confidence, _ in top_candidates:
            if (
                candidate_confidence >= CANDIDATE_CONFIDENCE_THRESHOLD
                and candidate_species in NATIVE_WHITELIST
            ):
                hits, total_score, peak_score = update_candidate_evidence(
                    candidate_species,
                    candidate_confidence,
                    now_ts,
                )

                if (
                    hits >= EVIDENCE_MIN_HITS
                    and (
                        peak_score >= 0.12
                        or total_score >= EVIDENCE_MIN_TOTAL_SCORE
                    )
                ):
                    confirmed_candidate = (
                        candidate_species,
                        max(peak_score, candidate_confidence),
                        hits,
                        total_score,
                    )
                    break

        if confirmed_candidate is None:
            has_whitelisted_candidate = any(
                score >= CANDIDATE_CONFIDENCE_THRESHOLD
                and species in NATIVE_WHITELIST
                for species, score, _ in top_candidates
            )
            request_matrix_state(
                "candidate" if has_whitelisted_candidate else "listening"
            )

        if confirmed_candidate is not None:
            (
                confirmed_species,
                confirmed_confidence,
                evidence_hits,
                evidence_total,
            ) = confirmed_candidate

            previous_time = last_detection_by_species.get(
                confirmed_species,
                0.0,
            )

            if (
                DUPLICATE_SUPPRESSION_SECONDS > 0
                and now_ts - previous_time
                < DUPLICATE_SUPPRESSION_SECONDS
            ):
                print(
                    f"[MATCH-SUPPRESSED] {confirmed_species} "
                    f"(peak={confirmed_confidence * 100:.1f}%, "
                    f"hits={evidence_hits})"
                )
                clear_candidate_evidence(confirmed_species)
            else:
                last_detection_by_species[confirmed_species] = now_ts
                clear_candidate_evidence(confirmed_species)

                print(
                    f"[MATCH] {confirmed_species} "
                    f"(peak={confirmed_confidence * 100:.1f}%) | "
                    f"evidence_hits={evidence_hits} | "
                    f"evidence_sum={evidence_total:.3f} | "
                    f"RMS={rms:.5f} | "
                    f"battery={_battery_level}% | "
                    f"time={now_local_iso()}"
                )

                # Log a second, explicit CONFIRMED event. The preceding top-1
                # row remains in the dataset, so the raw model output is never
                # hidden.
                request_matrix_state(
                    "crow" if confirmed_species == "House Crow" else "bird"
                )

                confirmed_row_id = log_detection(
                    confirmed_species,
                    confirmed_confidence,
                    _battery_level,
                    rms,
                    detection_status="CONFIRMED",
                    whitelisted=True,
                )

                send_telemetry(
                    confirmed_species,
                    confirmed_confidence,
                    _battery_level,
                )

                try:
                    _feature_q.put_nowait(
                        (
                            audio.copy(),
                            confirmed_species,
                            confirmed_row_id,
                        )
                    )
                except queue.Full:
                    print(
                        "[FEATURES] Queue full; "
                        "core detection was still saved."
                    )

                export_csv()
                last_heartbeat = now_ts

        else:
            # Remove old weak candidates so stale evidence cannot accumulate.
            stale_species = [
                species
                for species, state in candidate_history.items()
                if now_ts - state.get("last_seen", 0.0)
                > CANDIDATE_DECAY_SECONDS
            ]

            for stale_species_name in stale_species:
                clear_candidate_evidence(stale_species_name)

            # Export at heartbeat intervals so false-positive rows are still
            # available on disk while the station is running.
            if now_ts - last_heartbeat > HEARTBEAT_SECONDS:
                candidate_summary = ", ".join(
                    f"{name} {score * 100:.1f}%"
                    for name, score, _ in top_candidates[:3]
                )

                print(
                    f"[SCANNING] Top={candidate_summary} | "
                    f"RMS={rms:.5f} | "
                    f"battery={_battery_level}%"
                )

                send_telemetry(
                    "Scanning...",
                    0.0,
                    _battery_level,
                )

                export_csv()
                last_heartbeat = now_ts


except KeyboardInterrupt:
    print("\n[SYSTEM] Stopping...")


finally:
    request_matrix_state("idle")
    stop_ffmpeg(process)

    try:
        export_csv()
    except Exception as exc:
        print(f"[CSV] Final export failed: {exc}")

    print("[SYSTEM] Shutdown complete.")
