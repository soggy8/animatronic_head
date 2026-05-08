# Animatronic Head Talking

Voice-driven control system for an animatronic head with:
- Macedonian speech recognition (wake-word based)
- GPT-powered response generation
- ElevenLabs text-to-speech playback
- Real-time jaw/lip motion over serial
- Optional camera-based eye/neck/brow tracking (MediaPipe)

The project is designed for a laptop + ESP32 setup where Python handles audio/AI and the microcontroller handles servo actuation.

## Features

- Wake word flow (`nikola`) for hands-free interaction
- Macedonian transcription using Whisper (`whisper-1`)
- Short conversational replies using OpenAI chat completions
- Speech synthesis via ElevenLabs (`eleven_multilingual_v2`)
- Jaw movement mapped from playback RMS amplitude
- Optional high-level lip state commands (`mouth_open`, `smile`, etc.)
- Optional eye tracking pipeline using OpenCV + MediaPipe FaceMesh

## Project Structure

- `main.py` - Main conversation loop, wake-word handling, transcription, LLM response, TTS playback, and serial output.
- `eye_tracker.py` - Camera tracking thread that maps face position to eye/neck/brow servo targets.
- `requirements.txt` - Python dependencies.

## Requirements

- Python 3.10+ (3.12 recommended)
- A working microphone and speaker
- FFmpeg installed (required by `pydub` for audio conversion)
- Optional: ESP32 (or compatible) connected over serial
- Optional: USB camera for eye tracking

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id

SERIAL_PORT=/dev/ttyUSB0
SERIAL_BAUD=115200
USE_SERIAL=true

WAKE_WORD=nikola
ENABLE_EYE_TRACKING=false
EYE_CAMERA_INDEX=0
EYE_SHOW_WINDOW=true
ENABLE_LIP_COMMANDS=true
```

Important runtime settings supported by `main.py`:
- `JAW_MAX_DEG` - jaw travel ceiling (default `60`)
- `SILENCE_THRESHOLD_SECONDS` - phrase end detection
- `LISTEN_TIMEOUT_SECONDS` / `PHRASE_TIME_LIMIT_SECONDS` - input capture limits
- `WAKE_ACK_MODE` - `off`, `beep`, or `voice`
- `WAKE_ACK_TEXT` - spoken wake acknowledgement text when `WAKE_ACK_MODE=voice`

## Running

```bash
source venv/bin/activate
python main.py
```

On startup, choose input mode:
- `1` for voice mode (wake word + microphone)
- `2` for text mode (keyboard input)

In text mode, type `exit` or `quit` to stop.

## Serial Protocol Notes

The app writes newline-delimited commands to the serial device:
- Jaw angle values as integers (for example `0` to `60`)
- Lip-state command strings (for example `mouth_open`, `smile`)
- Eye tracker packets that start with `t` and contain comma-separated servo targets

Make sure your firmware parser matches these command formats.

## Eye Tracking

Enable with:

```env
ENABLE_EYE_TRACKING=true
```

When enabled, `EyeTracker` runs in a background thread and sends smoothed targets for:
- Eye left-right / up-down
- Eyelids
- Neck
- Right and left brows

If preview window is enabled, press `q` in the OpenCV window to close tracking.

## Troubleshooting

- No audio playback/input: verify OS audio devices and PyAudio installation.
- `pydub` conversion issues: ensure FFmpeg is installed and in PATH.
- Serial connection fails: verify `SERIAL_PORT`, baud rate, and cable permissions.
- Unstable wake-word behavior: adjust capture timeout and silence settings.
- Camera not found: check `EYE_CAMERA_INDEX` and webcam permissions.

## Safety Note

This project drives physical actuators. Start with conservative servo limits and test with power-limited conditions before full-speed operation.
