import os
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

# Reduce noisy backend logging in MediaPipe/TFLite/OpenCV.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")

import cv2
import mediapipe as mp

@dataclass
class EyeTrackerConfig:
    camera_index: int = 0
    send_interval: float = 0.03
    smoothing: float = 0.7
    startup_frames: int = 15
    show_window: bool = False
    window_name: str = "tracking"

    # Servo limits from your current tracker script
    lr_min: int = 30
    lr_max: int = 150
    ud_up: int = 145
    ud_down: int = 55
    lid_up: int = 30
    lid_down: int = 150

    # Face movement amplification
    face_gain: float = 2.4

    # Neck (channel 8) coordination
    neck_center: int = 90
    neck_min: int = 55
    neck_max: int = 125
    neck_deadzone: float = 0.12  # normalized around screen center (0.0-0.5)
    neck_hysteresis: float = 0.03  # prevents deadzone edge oscillation
    neck_error_smoothing: float = 0.2  # low-pass on horizontal error
    neck_smoothing: float = 0.16  # slower than eye smoothing
    neck_return_smoothing: float = 0.1  # gentle return to center
    neck_max_step_deg: float = 1.6  # max neck change per update
    neck_eye_compensation: float = 0.35  # how much neck motion recenters eyes

    # Right brow animation (channels 9/10 on ESP32 parser)
    rb_back_neutral: int = 100
    rb_front_neutral: int = 80
    rb_back_min: int = 60
    rb_back_max: int = 130
    rb_front_min: int = 60
    rb_front_max: int = 130
    rb_vertical_amp_back: float = 22.0
    rb_vertical_amp_front: float = 28.0
    rb_turn_bias: float = 8.0
    rb_smoothing: float = 0.22

    # Left brow animation (channels 12/11 on ESP32 parser: LB_BACK, LB_FRONT)
    lb_back_neutral: int = 95
    lb_front_neutral: int = 100
    lb_back_min: int = 65
    lb_back_max: int = 125
    lb_front_min: int = 70
    lb_front_max: int = 130
    lb_vertical_amp_back: float = 22.0
    lb_vertical_amp_front: float = 28.0
    lb_turn_bias: float = 8.0
    lb_smoothing: float = 0.22


def map_range(v: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@contextmanager
def suppress_native_stderr():
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


class EyeTracker:
    def __init__(self, send_line: Callable[[str], None], config: EyeTrackerConfig | None = None):
        self.send_line = send_line
        self.config = config or EyeTrackerConfig()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
            self.config.show_window = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="EyeTrackerThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cfg = self.config
        with suppress_native_stderr():
            face_mesh_module = None
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                face_mesh_module = mp.solutions.face_mesh
            else:
                try:
                    from mediapipe.python.solutions import face_mesh as face_mesh_fallback

                    face_mesh_module = face_mesh_fallback
                except Exception as exc:
                    print(f"Eye tracker: failed to load MediaPipe FaceMesh: {exc}")
                    return

            face_mesh = face_mesh_module.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            cap = cv2.VideoCapture(cfg.camera_index)

        if not cap.isOpened():
            print(f"Eye tracker: camera {cfg.camera_index} not available.")
            face_mesh.close()
            return

        current_x = 90.0
        current_y = 90.0
        current_lid_y = 90.0
        current_neck = float(cfg.neck_center)
        current_rb_back = float(cfg.rb_back_neutral)
        current_rb_front = float(cfg.rb_front_neutral)
        current_lb_back = float(cfg.lb_back_neutral)
        current_lb_front = float(cfg.lb_front_neutral)
        filtered_horizontal_error = 0.0
        neck_active = False
        last_send = 0.0
        last_sent_x = None
        last_sent_y = None
        last_sent_lid = None
        last_sent_neck = None
        last_sent_rb_back = None
        last_sent_rb_front = None
        last_sent_lb_back = None
        last_sent_lb_front = None
        startup_frames = 0

        print("Eye tracker started.")
        try:
            while cap.isOpened() and not self._stop_event.is_set():
                with suppress_native_stderr():
                    ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with suppress_native_stderr():
                    results = face_mesh.process(rgb)

                h, w, _ = frame.shape
                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    nose = face.landmark[1]

                    cx = int(nose.x * w)
                    cy = int(nose.y * h)
                    if cfg.show_window:
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    if startup_frames < cfg.startup_frames:
                        startup_frames += 1
                    else:
                        fx = max(0.0, min(1.0, (nose.x - 0.5) * cfg.face_gain + 0.5))
                        fy = max(0.0, min(1.0, (nose.y - 0.5) * cfg.face_gain + 0.5))

                        # Neck target is active only outside a center deadzone with hysteresis.
                        horizontal_error = fx - 0.5
                        filtered_horizontal_error += (
                            horizontal_error - filtered_horizontal_error
                        ) * cfg.neck_error_smoothing
                        deadzone = max(0.0, min(0.49, cfg.neck_deadzone))
                        hysteresis = max(0.0, min(0.2, cfg.neck_hysteresis))
                        activate_threshold = deadzone + hysteresis
                        release_threshold = max(0.0, deadzone - hysteresis)

                        if neck_active:
                            if abs(filtered_horizontal_error) < release_threshold:
                                neck_active = False
                        else:
                            if abs(filtered_horizontal_error) > activate_threshold:
                                neck_active = True

                        if not neck_active:
                            neck_target = float(cfg.neck_center)
                        else:
                            sign = 1.0 if filtered_horizontal_error >= 0 else -1.0
                            excess = abs(filtered_horizontal_error) - deadzone
                            norm = excess / max(0.5 - deadzone, 1e-6)
                            norm = max(0.0, min(1.0, norm))
                            if sign > 0:
                                neck_target = cfg.neck_center + norm * (cfg.neck_max - cfg.neck_center)
                            else:
                                neck_target = cfg.neck_center - norm * (cfg.neck_center - cfg.neck_min)

                        # Move eyes quickly, neck slower; as neck turns, eyes compensate opposite to recenter.
                        neck_smoothing = cfg.neck_smoothing if neck_active else cfg.neck_return_smoothing
                        neck_delta = (neck_target - current_neck) * neck_smoothing
                        neck_delta = max(-cfg.neck_max_step_deg, min(cfg.neck_max_step_deg, neck_delta))
                        current_neck += neck_delta
                        neck_norm = (current_neck - cfg.neck_center) / max((cfg.neck_max - cfg.neck_min) / 2.0, 1e-6)
                        compensated_fx = fx - neck_norm * cfg.neck_eye_compensation
                        compensated_fx = max(0.0, min(1.0, compensated_fx))

                        target_x = map_range(compensated_fx, 0.0, 1.0, cfg.lr_min, cfg.lr_max)
                        target_y = map_range(fy, 0.0, 1.0, cfg.ud_up, cfg.ud_down)
                        target_lid_y = map_range(fy, 0.0, 1.0, cfg.lid_up, cfg.lid_down)

                        current_x += (target_x - current_x) * cfg.smoothing
                        current_y += (target_y - current_y) * cfg.smoothing
                        current_lid_y += (target_lid_y - current_lid_y) * cfg.smoothing

                        # Right brow follows vertical attention + slight turn expression.
                        face_vertical = (fy - 0.5) * 2.0  # -1..1
                        turn_amount = max(0.0, min(1.0, abs(filtered_horizontal_error) / 0.5))
                        rb_back_target = cfg.rb_back_neutral - face_vertical * cfg.rb_vertical_amp_back - turn_amount * cfg.rb_turn_bias
                        rb_front_target = cfg.rb_front_neutral + face_vertical * cfg.rb_vertical_amp_front - turn_amount * cfg.rb_turn_bias
                        rb_back_target = max(cfg.rb_back_min, min(cfg.rb_back_max, rb_back_target))
                        rb_front_target = max(cfg.rb_front_min, min(cfg.rb_front_max, rb_front_target))
                        current_rb_back += (rb_back_target - current_rb_back) * cfg.rb_smoothing
                        current_rb_front += (rb_front_target - current_rb_front) * cfg.rb_smoothing

                        # Left brow mirrors right-brow behavior with opposite turn bias.
                        lb_back_target = cfg.lb_back_neutral - face_vertical * cfg.lb_vertical_amp_back + turn_amount * cfg.lb_turn_bias
                        lb_front_target = cfg.lb_front_neutral + face_vertical * cfg.lb_vertical_amp_front + turn_amount * cfg.lb_turn_bias
                        lb_back_target = max(cfg.lb_back_min, min(cfg.lb_back_max, lb_back_target))
                        lb_front_target = max(cfg.lb_front_min, min(cfg.lb_front_max, lb_front_target))
                        current_lb_back += (lb_back_target - current_lb_back) * cfg.lb_smoothing
                        current_lb_front += (lb_front_target - current_lb_front) * cfg.lb_smoothing

                        now = time.time()
                        if now - last_send > cfg.send_interval:
                            send_x = int(round(current_x))
                            send_y = int(round(current_y))
                            send_lid = int(round(current_lid_y))
                            send_neck = int(round(current_neck))
                            send_rb_back = int(round(current_rb_back))
                            send_rb_front = int(round(current_rb_front))
                            send_lb_back = int(round(current_lb_back))
                            send_lb_front = int(round(current_lb_front))
                            if (
                                send_x != last_sent_x
                                or send_y != last_sent_y
                                or send_lid != last_sent_lid
                                or send_neck != last_sent_neck
                                or send_rb_back != last_sent_rb_back
                                or send_rb_front != last_sent_rb_front
                                or send_lb_back != last_sent_lb_back
                                or send_lb_front != last_sent_lb_front
                            ):
                                self.send_line(
                                    f"t{send_x},{send_y},{send_lid},{send_neck},"
                                    f"{send_rb_back},{send_rb_front},{send_lb_back},{send_lb_front}"
                                )
                                last_sent_x = send_x
                                last_sent_y = send_y
                                last_sent_lid = send_lid
                                last_sent_neck = send_neck
                                last_sent_rb_back = send_rb_back
                                last_sent_rb_front = send_rb_front
                                last_sent_lb_back = send_lb_back
                                last_sent_lb_front = send_lb_front
                                last_send = now

                if cfg.show_window:
                    with suppress_native_stderr():
                        cv2.imshow(cfg.window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._stop_event.set()
                        break
        finally:
            cap.release()
            if cfg.show_window:
                try:
                    with suppress_native_stderr():
                        cv2.destroyWindow(cfg.window_name)
                except Exception:
                    pass
            face_mesh.close()
            print("Eye tracker stopped.")
