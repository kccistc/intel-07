# net.py
import socket
import os
from typing import Optional

def connect_ctrl_server(host: str, port: int, timeout: float = 5.0) -> Optional[socket.socket]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.settimeout(None)
        print(f"[NET] connected to {host} {port}")
        return s
    except Exception as e:
        print("[NET] connect failed:", e)
        return None

def send_only(sock: Optional[socket.socket], msg: str):
    """기존 즉시 전송 함수 — 테스트용으로 남겨둠."""
    if not sock:
        print("[NET] send_only: sock is None -> drop", msg)
        return
    try:
        sock.sendall(msg.encode('utf-8'))
        print("[NET SEND]", msg)
    except Exception as e:
        print("[NET] send_only failed:", e)

def guarded_send(app_state, sock: Optional[socket.socket], msg: str):
    """
    실제로는 app_state.attendance_ok 이 True 일 때만 전송.
    sock이 없거나 attendance_ok False면 전송 억제하고 디버그 로그만 출력.
    """
    if sock and getattr(app_state, "attendance_ok", False):
        try:
            sock.sendall(msg.encode('utf-8'))
            print("[NET SEND]", msg)
        except Exception as e:
            print("[NET] guarded_send failed:", e)
    else:
        # 억제 로그 (필요 없으면 지워도 됨)
        print(f"[NET] suppressed send -> msg={msg!r} attendance_ok={getattr(app_state,'attendance_ok',False)} sock={bool(sock)}")
