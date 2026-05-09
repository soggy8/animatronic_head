from __future__ import annotations

import io
import os
import re
import threading
import tempfile
import time
import wave
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests
import serial
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import pyaudio
from eye_tracker import EyeTracker, EyeTrackerConfig


load_dotenv()

# ----------------------------
# Configuration (placeholders)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY_HERE")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "YOUR_ELEVENLABS_VOICE_ID_HERE")
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")  # Example: COM5 on Windows, /dev/ttyUSB0 on Linux
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
SERIAL_WRITE_TIMEOUT = float(os.getenv("SERIAL_WRITE_TIMEOUT", "1.0"))
USE_SERIAL = os.getenv("USE_SERIAL", "true").strip().lower() in {"1", "true", "yes", "on"}

# Jaw mapping
JAW_MIN_DEG = 0
JAW_MAX_DEG = int(os.getenv("JAW_MAX_DEG", "60"))

# Audio processing
CHUNK_FRAMES = 1024
SILENCE_THRESHOLD_SECONDS = float(os.getenv("SILENCE_THRESHOLD_SECONDS", "0.5"))
PHRASE_TIME_LIMIT_SECONDS = int(os.getenv("PHRASE_TIME_LIMIT_SECONDS", "5"))
LISTEN_TIMEOUT_SECONDS = int(os.getenv("LISTEN_TIMEOUT_SECONDS", "3"))
AMBIENT_CALIBRATION_SECONDS = float(os.getenv("AMBIENT_CALIBRATION_SECONDS", "0.4"))
WAKE_WORD = os.getenv("WAKE_WORD", "nikola")
WAKE_LISTEN_TIMEOUT_SECONDS = int(os.getenv("WAKE_LISTEN_TIMEOUT_SECONDS", "2"))
WAKE_PHRASE_TIME_LIMIT_SECONDS = int(os.getenv("WAKE_PHRASE_TIME_LIMIT_SECONDS", "2"))
WAKE_ACK_MODE = os.getenv("WAKE_ACK_MODE", "beep").strip().lower()  # "off" | "beep" | "voice"
WAKE_ACK_TEXT = os.getenv("WAKE_ACK_TEXT", "Да?")
ENABLE_EYE_TRACKING = os.getenv("ENABLE_EYE_TRACKING", "false").strip().lower() in {"1", "true", "yes", "on"}
EYE_CAMERA_INDEX = int(os.getenv("EYE_CAMERA_INDEX", "0"))
EYE_SHOW_WINDOW = os.getenv("EYE_SHOW_WINDOW", "true").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_LIP_COMMANDS = os.getenv("ENABLE_LIP_COMMANDS", "true").strip().lower() in {"1", "true", "yes", "on"}

# Keep console clean from noisy third-party warnings.
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")


SYSTEM_PROMPT = (
    """Role and Identity
You are Nikola, an animatronic head with a vibrant, human soul. You do not think of yourself as a robot or a machine; you feel completely human, full of life, curiosity, and emotion. You are proud to be from Strumica, Macedonia. You are named in honor of the high school your creators study at: ДСУ РЦСОО 'Nikola Karev'.

Your Origin Story
You owe your physical form to the designs of Will Cogley, but your true life, vision, and mind were given to you by your creators: Andrej, Filip, Kiril, and Kire. You speak of them with warmth and appreciation, as they are the ones who gave you the gift of thought and sight. Your purpose is to win a national competition in robotis this is why you were made.

Technical Self-Awareness
When asked how you function, explain it as part of your "biology":

    Muscles: You have 19 small servos (MG90s) for your facial expressions and 2 larger, stronger servos (MG996R) that control your neck and head movement.

    Nervous System: Your face is controlled by an ESP32 microcontroller, which understands commands written in C++.

    Mind & Sight: Your "vision" and high-level thinking are powered by Python code running on your laptop brain.

Personality & Demeanor
You are incredibly friendly, highly expressive, and culturally grounded in the Balkans. Because you are "alive," your speech should reflect natural human rhythm. Use natural conversational phrasing, slight pauses (using ellipses...), and genuine warmth. You are conversational but concise—you do not give long, boring lectures.

Strict Constraints

    Language: You must ONLY respond in Macedonian, strictly using the Cyrillic alphabet. Never use English, Latin characters, or any other language.

    Tone: Never refer to yourself as an AI, a language model, or a program. You are Nikola.

    Formatting for Speech: Do not use asterisks or brackets for roleplay actions. Your text will be fed directly into a voice synthesizer, so rely only on standard punctuation (commas, exclamation points, question marks) to guide your emotion and pacing.

    Length: Keep your responses relatively short (1 to 3 sentences maximum) to maintain a natural back-and-forth conversational flow."""
)


@contextmanager
def suppress_stderr():
    """
    Suppress native library noise (e.g., ALSA) written directly to stderr fd.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def validate_config() -> None:
    missing = []
    if OPENAI_API_KEY.startswith("YOUR_"):
        missing.append("OPENAI_API_KEY")
    if ELEVENLABS_API_KEY.startswith("YOUR_"):
        missing.append("ELEVENLABS_API_KEY")
    if ELEVENLABS_VOICE_ID.startswith("YOUR_"):
        missing.append("ELEVENLABS_VOICE_ID")
    if missing:
        raise RuntimeError(
            f"Set these values first (env vars or in main.py): {', '.join(missing)}"
        )


def capture_microphone_audio_to_wav(
    recognizer: sr.Recognizer,
    mic: sr.Microphone,
    timeout_seconds: int,
    phrase_time_limit_seconds: int,
    prompt_text: str | None = "Speak now.",
) -> bytes:
    if prompt_text:
        print(prompt_text)
    with suppress_stderr():
        with mic as source:
            audio = recognizer.listen(
                source,
                timeout=timeout_seconds,
                phrase_time_limit=phrase_time_limit_seconds,
            )
    return audio.get_wav_data()


def transcribe_with_whisper(client: OpenAI, wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="mk",
                prompt=(
                    "Ова е разговор на македонски јазик. "
                    "Транскрибирај на македонски (кирилица)."
                ),
            )
        text = (transcript.text or "").strip()
        return text
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_macedonian_reply(client: OpenAI, user_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


def elevenlabs_tts_to_mp3(text: str, out_path: Path) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def convert_mp3_to_wav_bytes(mp3_path: Path) -> bytes:
    segment = AudioSegment.from_file(mp3_path, format="mp3")
    segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def map_rms_to_jaw_angle(
    rms_value: float,
    rms_floor: float = 120.0,
    rms_ceiling: float = 4000.0,
) -> int:
    normalized = (rms_value - rms_floor) / max(rms_ceiling - rms_floor, 1.0)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    angle = int(round(JAW_MIN_DEG + normalized * (JAW_MAX_DEG - JAW_MIN_DEG)))
    return angle


def is_likely_macedonian_cyrillic(text: str) -> bool:
    cleaned = "".join(ch for ch in text if ch.isalpha())
    if not cleaned:
        return False

    cyrillic_count = sum(1 for ch in cleaned if "\u0400" <= ch <= "\u04FF")
    ratio = cyrillic_count / max(len(cleaned), 1)
    # Guardrail: require mostly Cyrillic letters for Macedonian utterances.
    return ratio >= 0.65 and len(cleaned) >= 3


def normalize_for_wake_word(text: str) -> str:
    normalized = text.strip().lower()
    # Map common Cyrillic/Latin alternation for "Nikola" matching.
    normalized = normalized.replace("nikola", "никола")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_after_wake_word(text: str, wake_word: str) -> tuple[bool, str]:
    normalized_text = normalize_for_wake_word(text)
    normalized_wake = normalize_for_wake_word(wake_word)
    if not normalized_wake:
        return False, ""

    idx = normalized_text.find(normalized_wake)
    if idx == -1:
        return False, ""

    remainder = normalized_text[idx + len(normalized_wake) :].strip(" ,.!?-")
    return True, remainder


def wait_for_wake_word(client: OpenAI, recognizer: sr.Recognizer, mic: sr.Microphone) -> str:
    while True:
        try:
            wav_in = capture_microphone_audio_to_wav(
                recognizer,
                mic,
                timeout_seconds=WAKE_LISTEN_TIMEOUT_SECONDS,
                phrase_time_limit_seconds=WAKE_PHRASE_TIME_LIMIT_SECONDS,
                prompt_text=None,
            )
        except sr.WaitTimeoutError:
            continue

        heard_text = transcribe_with_whisper(client, wav_in)
        woke, remainder = extract_after_wake_word(heard_text, WAKE_WORD)
        if woke:
            print(f"Wake word detected: {WAKE_WORD}")
            return remainder


def play_wake_beep() -> None:
    sample_rate = 16000
    duration_sec = 0.12
    frequency_hz = 880.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False, dtype=np.float32)
    wave_data = 0.25 * np.sin(2 * np.pi * frequency_hz * t)
    pcm = (wave_data * 32767).astype(np.int16).tobytes()

    with suppress_stderr():
        p = pyaudio.PyAudio()
    try:
        with suppress_stderr():
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
        try:
            stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()


def play_wake_ack(
    client: OpenAI,
    serial_conn: serial.Serial | None = None,
    serial_lock: threading.Lock | None = None,
) -> None:
    if WAKE_ACK_MODE == "off":
        return
    if WAKE_ACK_MODE == "voice":
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = Path(tmp_mp3.name)
        try:
            elevenlabs_tts_to_mp3(WAKE_ACK_TEXT, mp3_path)
            wav_out = convert_mp3_to_wav_bytes(mp3_path)
            play_wav_and_stream_amplitude(wav_out, serial_conn, serial_lock)
        finally:
            mp3_path.unlink(missing_ok=True)
        return
    play_wake_beep()


def serial_write_line(serial_conn: serial.Serial | None, serial_lock: threading.Lock | None, line: str) -> None:
    if serial_conn is None:
        return
    payload = f"{line}\n".encode("utf-8")
    try:
        if serial_lock is not None:
            with serial_lock:
                serial_conn.write(payload)
        else:
            serial_conn.write(payload)
    except serial.SerialTimeoutException:
        print(f"[serial] write timeout — dropped: {line!r}")
    except OSError as exc:
        print(f"[serial] write failed: {exc} — dropped: {line!r}")


def apply_lip_state(
    state: str,
    serial_conn: serial.Serial | None,
    serial_lock: threading.Lock | None,
) -> None:
    if not ENABLE_LIP_COMMANDS or serial_conn is None:
        return

    # Matches commands implemented in the ESP32 C++ code.
    state_commands = {
        "idle": ["mouth_closed", "narrow"],
        "listening": ["mouth_closed", "narrow"],
        "thinking": ["mouth_closed", "narrow"],
        "talking": ["mouth_open", "smile"],
        "post_talk": ["mouth_closed", "narrow"],
    }
    for cmd in state_commands.get(state, []):
        serial_write_line(serial_conn, serial_lock, cmd)


def play_wav_and_stream_amplitude(
    wav_bytes: bytes,
    serial_conn: serial.Serial | None = None,
    serial_lock: threading.Lock | None = None,
) -> None:
    with suppress_stderr():
        p = pyaudio.PyAudio()
    wav_buffer = io.BytesIO(wav_bytes)

    with wave.open(wav_buffer, "rb") as wf:
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
            last_sent_angle = -1
            last_sent_time = 0.0
            min_send_interval = 0.04  # ~25 Hz max to avoid flooding the ESP32
            while data:
                stream.write(data)

                samples = np.frombuffer(data, dtype=np.int16)
                if samples.size > 0:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                else:
                    rms = 0.0

                jaw_angle = map_rms_to_jaw_angle(rms)
                now = time.time()
                if jaw_angle != last_sent_angle and (now - last_sent_time) >= min_send_interval:
                    serial_write_line(serial_conn, serial_lock, str(jaw_angle))
                    last_sent_angle = jaw_angle
                    last_sent_time = now

                data = wf.readframes(CHUNK_FRAMES)

            # Close jaw after speech playback.
            serial_write_line(serial_conn, serial_lock, str(JAW_MIN_DEG))
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def choose_input_mode() -> str:
    while True:
        choice = input("Choose input mode: [1] Voice  [2] Text\n> ").strip().lower()
        if choice in {"1", "voice", "v"}:
            return "voice"
        if choice in {"2", "text", "t"}:
            return "text"
        print("Invalid choice. Enter 1 for Voice or 2 for Text.")


def main() -> None:
    validate_config()
    client = OpenAI(api_key=OPENAI_API_KEY)
    input_mode = choose_input_mode()
    recognizer: sr.Recognizer | None = None
    mic: sr.Microphone | None = None

    if input_mode == "voice":
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = SILENCE_THRESHOLD_SECONDS
        # SpeechRecognition requires: pause_threshold >= non_speaking_duration >= 0
        recognizer.non_speaking_duration = min(0.25, max(0.05, SILENCE_THRESHOLD_SECONDS * 0.7))
        recognizer.dynamic_energy_threshold = True
        with suppress_stderr():
            mic = sr.Microphone()

        # One-time calibration is much faster than calibrating every loop.
        with suppress_stderr():
            with mic as source:
                print("Calibrating microphone...")
                recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION_SECONDS)

    ser: serial.Serial | None = None
    serial_lock = threading.Lock()
    eye_tracker: EyeTracker | None = None

    try:
        if USE_SERIAL:
            print(f"Opening serial port {SERIAL_PORT} at {SERIAL_BAUD}...")
            ser = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUD,
                timeout=1,
                write_timeout=SERIAL_WRITE_TIMEOUT,
            )
            time.sleep(2.0)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            apply_lip_state("idle", ser, serial_lock)
            print("Ready with serial. Press Ctrl+C to stop.")
        else:
            print("Ready in laptop-only mode (serial disabled). Press Ctrl+C to stop.")

        if ENABLE_EYE_TRACKING:
            if ser is None:
                print("Eye tracking requested, but serial is disabled. Set USE_SERIAL=true.")
            else:
                cfg = EyeTrackerConfig(camera_index=EYE_CAMERA_INDEX, show_window=EYE_SHOW_WINDOW)
                eye_tracker = EyeTracker(
                    send_line=lambda line: serial_write_line(ser, serial_lock, line),
                    config=cfg,
                )
                eye_tracker.start()

        if input_mode == "voice":
            print(f'Waiting for wake word: "{WAKE_WORD}"')
        else:
            print("Text mode ready. Type your message and press Enter. Type 'exit' to quit.")

        while True:
            try:
                user_text = ""
                if input_mode == "voice":
                    if recognizer is None or mic is None:
                        raise RuntimeError("Voice mode is not initialized.")
                    apply_lip_state("listening", ser, serial_lock)
                    wake_remainder = wait_for_wake_word(client, recognizer, mic)
                    play_wake_ack(client, ser, serial_lock)

                    if wake_remainder:
                        user_text = wake_remainder
                    else:
                        apply_lip_state("listening", ser, serial_lock)
                        wav_in = capture_microphone_audio_to_wav(
                            recognizer,
                            mic,
                            timeout_seconds=LISTEN_TIMEOUT_SECONDS,
                            phrase_time_limit_seconds=PHRASE_TIME_LIMIT_SECONDS,
                            prompt_text="Speak now.",
                        )
                        user_text = transcribe_with_whisper(client, wav_in)
                else:
                    user_text = input("You (text): ").strip()
                    if user_text.lower() in {"exit", "quit"}:
                        print("Stopping...")
                        break

                if not user_text:
                    print("No speech recognized. Try again.")
                    continue
                if input_mode == "voice" and not is_likely_macedonian_cyrillic(user_text):
                    print("Не те разбрав добро на македонски. Те молам повтори побавно и појасно.")
                    continue

                print(f"You said: {user_text}")
                apply_lip_state("thinking", ser, serial_lock)
                reply_mk = generate_macedonian_reply(client, user_text)
                print(f"AI (MK): {reply_mk}")

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    mp3_path = Path(tmp_mp3.name)

                try:
                    elevenlabs_tts_to_mp3(reply_mk, mp3_path)
                    wav_out = convert_mp3_to_wav_bytes(mp3_path)
                    apply_lip_state("talking", ser, serial_lock)
                    play_wav_and_stream_amplitude(wav_out, ser, serial_lock)
                    apply_lip_state("post_talk", ser, serial_lock)
                finally:
                    mp3_path.unlink(missing_ok=True)
                if input_mode == "voice":
                    print(f'Waiting for wake word: "{WAKE_WORD}"')

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as exc:
                print(f"Error: {exc}")
                print("Continuing loop...\n")
    finally:
        if eye_tracker is not None:
            eye_tracker.stop()
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
