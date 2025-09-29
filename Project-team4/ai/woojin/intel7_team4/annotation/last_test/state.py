# state.py
from dataclasses import dataclass, field
import threading
from typing import Optional

@dataclass
class AppState:
    latest_jpeg: Optional[bytes] = None
    frame_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    status_lock: threading.Lock = field(default_factory=threading.Lock)

    # status fields
    current_face_state: str = "NONE"
    current_slp_state: str = "NONE"
    current_neck_angle: Optional[float] = None
    current_eye_status: str = "UNKNOWN"
    current_fps: int = 0

    # ===== new flag used by guarded sends =====
    attendance_ok: bool = False
