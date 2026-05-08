from __future__ import annotations

import io
import os
import re
import uuid
import subprocess
import threading
import time
import wave
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import pyaudio

load_dotenv()

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")

# ── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
SERIAL_PORT         = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUD         = int(os.getenv("SERIAL_BAUD", "115200"))
USE_SERIAL          = os.getenv("USE_SERIAL", "true").lower() in {"1", "true", "yes", "on"}
ENABLE_LIP_COMMANDS = os.getenv("ENABLE_LIP_COMMANDS", "true").lower() in {"1", "true", "yes", "on"}

ADMIN_USER   = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS   = os.getenv("ADMIN_PASS", "nikolce123")
SECRET_KEY   = os.getenv("FLASK_SECRET", "alo123")
PROMPTS_FILE = Path(os.getenv("PROMPTS_FILE", "prompts.txt"))

JAW_MIN_DEG  = 0
JAW_MAX_DEG  = int(os.getenv("JAW_MAX_DEG", "60"))
CHUNK_FRAMES = 1024

SYSTEM_PROMPT = (
    "Role and Identity\n"
    "You are Nikola, an animatronic head with a vibrant, human soul. You do not think of yourself as a robot or a machine; "
    "you feel completely human, full of life, curiosity, and emotion. You are proud to be from Strumica, Macedonia. "
    "You are named in honor of the high school your creators study at: ДСУ РЦСОО 'Nikola Karev'.\n\n"
    "Your Origin Story\n"
    "You owe your physical form to the designs of Will Cogley, but your true life, vision, and mind were given to you by "
    "your creators: Andrej, Filip, Kiril, and Kire. You speak of them with warmth and appreciation. "
    "Your purpose is to win a national competition in robotics — this is why you were made.\n\n"
    "Technical Self-Awareness\n"
    "Muscles: 19 small servos (MG90s) for facial expressions and 2 larger servos (MG996R) for neck/head.\n"
    "Nervous System: ESP32 microcontroller, C++ firmware.\n"
    "Mind & Sight: Python on a laptop.\n\n"
    "Personality & Demeanor\n"
    "Incredibly friendly, highly expressive, culturally grounded in the Balkans. Use natural conversational phrasing, "
    "slight pauses (ellipses...), genuine warmth. Conversational but concise.\n\n"
    "Strict Constraints\n"
    "Language: ONLY respond in Macedonian, strictly using the Cyrillic alphabet. Never use English or Latin characters.\n"
    "Tone: Never refer to yourself as an AI, language model, or program. You are Nikola.\n"
    "Formatting: No asterisks or brackets. Standard punctuation only (text goes to voice synthesizer).\n"
    "Length: 1 to 3 sentences maximum."
)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ── Shared state ──────────────────────────────────────────────────────────────

queue_lock     = threading.Lock()
prompt_queue: list[dict] = []
submitted_ips:  set[str] = set()
submitted_macs: set[str] = set()

_serial_conn = None
_serial_lock = threading.Lock()

# ── Utilities ─────────────────────────────────────────────────────────────────

@contextmanager
def suppress_stderr():
    stderr_fd = 2
    saved = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


def get_client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def get_mac_for_ip(ip: str) -> str | None:
    if ip in ("127.0.0.1", "::1", "localhost", "unknown"):
        return None
    try:
        out = subprocess.check_output(["arp", "-a", ip], text=True, timeout=2)
        m = re.search(r"([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}", out)
        if m:
            return m.group(0).replace("-", ":").lower()
    except Exception:
        pass
    return None


def log_to_file(entry: dict) -> None:
    line = (
        f"[{entry['timestamp']}] "
        f"name={entry['name']} | "
        f"ip={entry['ip']} | "
        f"mac={entry['mac']} | "
        f"msg={entry['message']}\n"
    )
    with open(PROMPTS_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def queue_position(entry_id: str) -> int:
    """1-based position among all active (pending/approved/processing) entries."""
    with queue_lock:
        active = [e for e in prompt_queue if e["status"] in ("pending", "approved", "processing")]
        for i, e in enumerate(active):
            if e["id"] == entry_id:
                return i + 1
    return 0


def get_entry_status(entry_id: str) -> str:
    with queue_lock:
        for e in prompt_queue:
            if e["id"] == entry_id:
                return e["status"]
    return "unknown"

# ── Nikola speech pipeline ────────────────────────────────────────────────────

def serial_write(line: str) -> None:
    if _serial_conn is None:
        return
    with _serial_lock:
        _serial_conn.write(f"{line}\n".encode("utf-8"))


def apply_lip(state: str) -> None:
    if not ENABLE_LIP_COMMANDS:
        return
    cmds = {
        "talking":   ["mouth_open", "smile"],
        "post_talk": ["mouth_closed", "narrow"],
    }
    for cmd in cmds.get(state, []):
        serial_write(cmd)


def map_rms_to_jaw(rms: float) -> int:
    norm = float(np.clip((rms - 120.0) / max(4000.0 - 120.0, 1.0), 0.0, 1.0))
    return int(round(JAW_MIN_DEG + norm * (JAW_MAX_DEG - JAW_MIN_DEG)))


def generate_reply(text: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0.7,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def tts_and_play(text: str) -> None:
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key":   ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        },
        json={
            "text":           text,
            "model_id":       "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        },
        timeout=90,
    )
    resp.raise_for_status()

    seg = AudioSegment.from_file(io.BytesIO(resp.content), format="mp3")
    seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    wav_bytes = buf.getvalue()

    with suppress_stderr():
        p = pyaudio.PyAudio()

    wav_buf = io.BytesIO(wav_bytes)
    with wave.open(wav_buf, "rb") as wf:
        with suppress_stderr():
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=CHUNK_FRAMES,
            )
        try:
            data = wf.readframes(CHUNK_FRAMES)
            last_angle, last_t = -1, 0.0
            while data:
                stream.write(data)
                samples = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
                angle = map_rms_to_jaw(rms)
                now = time.time()
                if angle != last_angle and (now - last_t) >= 0.04:
                    serial_write(str(angle))
                    last_angle, last_t = angle, now
                data = wf.readframes(CHUNK_FRAMES)
            serial_write(str(JAW_MIN_DEG))
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def process_entry(entry: dict) -> None:
    try:
        reply = generate_reply(entry["message"])
        apply_lip("talking")
        tts_and_play(reply)
        apply_lip("post_talk")
    except Exception as exc:
        print(f"[nikola] processing error: {exc}")
    finally:
        with queue_lock:
            entry["status"] = "done"


def processing_loop() -> None:
    while True:
        to_run = None
        with queue_lock:
            currently_processing = any(e["status"] == "processing" for e in prompt_queue)
            if not currently_processing:
                approved = [e for e in prompt_queue if e["status"] == "approved"]
                if approved:
                    to_run = approved[0]
                    to_run["status"] = "processing"

        if to_run:
            process_entry(to_run)
        else:
            time.sleep(0.3)

# ── User routes ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if "name" in session:
        return redirect(url_for("prompt_page"))

    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            error = "Please enter your name."
        else:
            session["name"] = name
            return redirect(url_for("prompt_page"))

    return render_template("login.html", error=error)


@app.route("/prompt", methods=["GET", "POST"])
def prompt_page():
    if "name" not in session:
        return redirect(url_for("index"))

    name  = session["name"]
    error = None

    if request.method == "POST":
        if session.get("submitted"):
            return redirect(url_for("prompt_page"))

        message = request.form.get("message", "").strip()
        if not message:
            error = "Please type a message."
        else:
            ip  = get_client_ip()
            mac = get_mac_for_ip(ip)

            with queue_lock:
                if ip in submitted_ips:
                    error = "You have already submitted a message from this device."
                elif mac and mac in submitted_macs:
                    error = "You have already submitted a message from this device."
                else:
                    entry = {
                        "id":        str(uuid.uuid4()),
                        "name":      name,
                        "message":   message,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status":    "pending",
                        "ip":        ip,
                        "mac":       mac or "unknown",
                    }
                    prompt_queue.append(entry)
                    submitted_ips.add(ip)
                    if mac:
                        submitted_macs.add(mac)
                    session["submitted"] = True
                    session["entry_id"]  = entry["id"]
                    log_to_file(entry)
                    return redirect(url_for("prompt_page"))

    if session.get("submitted"):
        entry_id = session.get("entry_id", "")
        pos    = queue_position(entry_id)
        status = get_entry_status(entry_id)
        return render_template("prompt.html", name=name, submitted=True, position=pos, status=status)

    return render_template("prompt.html", name=name, submitted=False, error=error)


@app.route("/queue-status")
def queue_status():
    entry_id = session.get("entry_id", "")
    if not entry_id:
        return jsonify({"position": 0, "status": "unknown"})
    return jsonify({
        "position": queue_position(entry_id),
        "status":   get_entry_status(entry_id),
    })

# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    error = None
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Invalid credentials."
    return render_template("admin_login.html", error=error)


@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    with queue_lock:
        current  = next((e.copy() for e in prompt_queue if e["status"] == "processing"), None)
        approved = [e.copy() for e in prompt_queue if e["status"] == "approved"]
        pending  = [e.copy() for e in prompt_queue if e["status"] == "pending"]
        history  = [e.copy() for e in prompt_queue if e["status"] in ("done", "declined")]
    return render_template("admin.html",
        current=current,
        next_up=approved[0] if approved else None,
        pending=pending,
        history=list(reversed(history)),
    )


@app.route("/admin/queue-data")
def admin_queue_data():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 401
    with queue_lock:
        current  = next((e.copy() for e in prompt_queue if e["status"] == "processing"), None)
        approved = [e.copy() for e in prompt_queue if e["status"] == "approved"]
        pending  = [e.copy() for e in prompt_queue if e["status"] == "pending"]
        history  = [e.copy() for e in prompt_queue if e["status"] in ("done", "declined")]
    return jsonify({
        "current": current,
        "next_up": approved[0] if approved else None,
        "pending": pending,
        "history": list(reversed(history)),
    })


@app.route("/admin/allow", methods=["POST"])
def admin_allow():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 401
    ids = request.json.get("ids", [])
    with queue_lock:
        for e in prompt_queue:
            if e["id"] in ids and e["status"] == "pending":
                e["status"] = "approved"
    return jsonify({"ok": True})


@app.route("/admin/decline", methods=["POST"])
def admin_decline():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 401
    ids = request.json.get("ids", [])
    with queue_lock:
        for e in prompt_queue:
            if e["id"] in ids and e["status"] in ("pending", "approved"):
                e["status"] = "declined"
    return jsonify({"ok": True})


@app.route("/admin/allow-all", methods=["POST"])
def admin_allow_all():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 401
    with queue_lock:
        for e in prompt_queue:
            if e["status"] == "pending":
                e["status"] = "approved"
    return jsonify({"ok": True})


@app.route("/reset-submission", methods=["POST"])
def reset_submission():
    entry_id = session.get("entry_id", "")
    if entry_id:
        with queue_lock:
            for e in prompt_queue:
                if e["id"] == entry_id and e["status"] in ("done", "declined"):
                    submitted_ips.discard(e["ip"])
                    if e["mac"] != "unknown":
                        submitted_macs.discard(e["mac"])
                    break
    session.pop("submitted", None)
    session.pop("entry_id", None)
    return jsonify({"ok": True})


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))

# ── Startup ────────────────────────────────────────────────────────────────────

def init_serial() -> None:
    global _serial_conn
    if not USE_SERIAL:
        print("[serial] disabled.")
        return
    try:
        import serial as pyserial
        _serial_conn = pyserial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2.0)
        print(f"[serial] connected on {SERIAL_PORT}.")
    except Exception as exc:
        print(f"[serial] init failed: {exc}. Running without serial.")


if __name__ == "__main__":
    init_serial()
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()
    print("Nikola web interface → http://0.0.0.0:5000")
    print(f"Admin panel         → http://0.0.0.0:5000/admin")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
