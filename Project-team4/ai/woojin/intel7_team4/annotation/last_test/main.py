#!/usr/bin/env python3
# main.py - 전체 실행 스크립트 (FLAG/VALUE 처리, FACE auth, ATTENDANCE 대기 포함)

import argparse
import threading
import time
import os
import signal
import re
import socket

from state import AppState
from processing import processing_loop, run_auth_local
from server import create_app
from net import connect_ctrl_server, send_only
from mjpg import start_mjpg_streamer_if_needed, kill_process_group

# 기본값
SRC = os.environ.get("SRC", "AI")
DEFAULT_USER = "seol"

# 서버 응답용 정규식
_flag_re = re.compile(r"^SERVER(?::[^:]+)*:FLAG:(?P<face>[01])$")
_value_re = re.compile(
    r"^SERVER(?::[^:]+)*:VALUE:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$"
)
_att_ok_re = re.compile(r"(?:SERVER:)?(?:[^:]+:)?ATTENDANCE:OK")


def wait_for_pattern(sock: socket.socket, regex: re.Pattern, timeout: float = None):
    """socket에서 regex가 매칭될 때까지(또는 timeout) 대기. 매칭된 문자열 반환 또는 None."""
    if sock is None:
        return None
    end = None if timeout is None else (time.time() + timeout)
    buf = b""
    sock.settimeout(1.0)
    while True:
        if end is not None and time.time() > end:
            return None
        try:
            data = sock.recv(4096)
        except socket.timeout:
            continue
        except Exception:
            return None
        if not data:
            return None
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            sline = line.decode("utf-8", errors="ignore").strip()
            if regex.search(sline) or regex.match(sline):
                return sline
        # 버퍼에 부분 문자열만 있는 경우에도 검사
        try:
            s = buf.decode("utf-8", errors="ignore")
            if regex.search(s) or regex.match(s):
                return s
        except Exception:
            pass


def run_flask_in_thread(app, host="0.0.0.0", port=8081):
    def run():
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(description="Run neck/eye processing + mjpg + flask (FLAG/VALUE/ATTENDANCE flow)")
    parser.add_argument("--no-mjpg", action="store_true")
    parser.add_argument("--mjpg-port", type=int, default=8080)
    parser.add_argument("--proc-port", type=int, default=8081)
    parser.add_argument("--mjpg-bin", default="/usr/local/bin/mjpg_streamer")
    parser.add_argument("--mjpg-input", default="input_uvc.so")
    parser.add_argument("--mjpg-output", default="output_http.so")
    parser.add_argument("--cam-index", type=int, default=0)

    # control behavior
    parser.add_argument("--force-local-auth", action="store_true",
                        help="Even if server says FLAG==1, force local auth before continuing")
    parser.add_argument("--user", default=DEFAULT_USER,
                        help="User id to use in messages (default: %(default)s)")

    args = parser.parse_args()

    app_state = AppState()

    USER = args.user
    LOGIN_MSG = os.environ.get("LOGIN_MSG", f"{SRC}:PASSWD")

    # 제어 서버 연결 시도
    ctrl_host = os.environ.get("CTRL_HOST", "192.168.0.158")
    ctrl_port = int(os.environ.get("CTRL_PORT", "5000"))
    sock = connect_ctrl_server(ctrl_host, ctrl_port)

    face_flag = None
    # VALUE 기본 값들 (서버에서 오면 덮어씀)
    forehead_chin = None
    nose_lips = None
    eye_eye = None
    ear_threshold = None

    if sock is not None:
        print(f"[MAIN] control socket connected to {ctrl_host}:{ctrl_port}")
        # LOGIN 시도 (best-effort)
        try:
            send_only(sock, LOGIN_MSG)
        except Exception as e:
            print("[MAIN] send LOGIN_MSG failed (continuing):", e)

        # FLAG 요청
        try:
            send_only(sock, f"{SRC}:{USER}:AUTH:FLAG")
            line = wait_for_pattern(sock, _flag_re, timeout=3.0)
            if line:
                m = _flag_re.match(line)
                if m:
                    face_flag = int(m.group("face"))
                    print(f"[MAIN] got FLAG from server: FACE_FLAG={face_flag}")
                else:
                    print("[MAIN] FLAG reply didn't match regex:", line)
            else:
                print("[MAIN] FLAG request timed out")
        except Exception as e:
            print("[MAIN] FLAG request failed:", e)

        # VALUE 요청
        try:
            send_only(sock, f"{SRC}:{USER}:AUTH:VALUE")
            line = wait_for_pattern(sock, _value_re, timeout=3.0)
            if line:
                m = _value_re.match(line)
                if m:
                    forehead_chin = float(m.group("fore"))
                    nose_lips = float(m.group("nose"))
                    eye_eye = float(m.group("eye"))
                    ear_threshold = float(m.group("ear"))
                    print("[MAIN] VALUE set from server:", forehead_chin, nose_lips, eye_eye, ear_threshold)
                else:
                    print("[MAIN] VALUE reply didn't match regex:", line)
            else:
                print("[MAIN] VALUE request timed out (using defaults)")
        except Exception as e:
            print("[MAIN] VALUE request failed:", e)

        # 강제 로컬 인증 옵션이 있으면 face_flag를 0으로 취급
        if args.force_local_auth:
            print("[MAIN] --force-local-auth set -> forcing local auth (treat as FLAG==0)")
            face_flag = 0

        # FLAG == 0 이면 로컬 인증 수행하고 FACE:OK 전송
        if face_flag == 0:
            print("[MAIN] FACE_FLAG == 0 -> performing local auth before starting mjpg/processing")
            mp_face = __import__("mediapipe").solutions.face_mesh
            face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.05, min_tracking_confidence=0.05)
            try:
                ok = run_auth_local(face_mesh, args.cam_index, timeout=60.0)
                print("[MAIN] local auth result:", ok)
                if ok:
                    try:
                        send_only(sock, f"{SRC}:{USER}:FACE:OK")
                        print("[MAIN] sent FACE:OK to server")
                    except Exception as e:
                        print("[MAIN] failed to send FACE:OK:", e)
                else:
                    print("[MAIN] local auth failed or timed out; continuing but server may reject")
            finally:
                try:
                    face_mesh.close()
                except Exception:
                    pass
        else:
            print("[MAIN] FACE_FLAG != 0 -> skipping forced local auth")

        # **여기서부터: 서버가 연결되어 있으면 반드시 ATTENDANCE:OK 를 기다립니다**
        print("[MAIN] waiting for ATTENDANCE:OK from server (blocking). Use Ctrl+C to cancel.")
        try:
            while not app_state.stop_event.is_set():
                line = wait_for_pattern(sock, _att_ok_re, timeout=5.0)
                if line:
                    print("[MAIN] ATTENDANCE received:", line)
                    app_state.attendance_ok = True
                    break
                else:
                    # 주기적으로 VALUE 재요청해서 서버와 통신을 유지하게 함 (선택적)
                    try:
                        send_only(sock, f"{SRC}:{USER}:AUTH:VALUE")
                    except Exception:
                        pass
            if app_state.stop_event.is_set():
                print("[MAIN] stop requested while waiting for ATTENDANCE -> exiting main")
                # 정리 후 종료
                try:
                    if sock:
                        sock.close()
                except Exception:
                    pass
                return
        except Exception as e:
            print("[MAIN] error while waiting for ATTENDANCE:", e)
            # 예외 발생 시에는 proceed 하지 않고, 필요에 따라 계속 진행시키려면 여기서 app_state.attendance_ok를 설정하거나 not.
            app_state.attendance_ok = False

    else:
        # 서버 없음: FLAG/VALUE/ATTENDANCE 단계 전부 건너뜀(로컬 실행)
        print("[MAIN] no control socket available -> skipping FLAG/VALUE/ATTENDANCE steps (local only)")
        app_state.attendance_ok = False

    # (선택) processing 모듈에 서버에서 받은 VALUE들을 주입
    try:
        import processing as _processing
        _processing.SRC = SRC
        _processing.USER = USER
        if forehead_chin is not None:
            _processing.FOREHEAD_CHIN = forehead_chin
        if nose_lips is not None:
            _processing.NOSE_LIPS = nose_lips
        if eye_eye is not None:
            _processing.EYE_EYE = eye_eye
        if ear_threshold is not None:
            _processing.EAR_THRESHOLD = ear_threshold
    except Exception:
        pass

    # mjpg-streamer 시작 또는 확인
    mjpg_proc = None
    if not args.no_mjpg:
        print("[MAIN] starting/confirming mjpg-streamer...")
        mjpg_proc = start_mjpg_streamer_if_needed(
            args.mjpg_bin, args.mjpg_input, args.mjpg_output,
            f"/dev/video{args.cam_index}", args.mjpg_port
        )
        if mjpg_proc:
            print("[MAIN] mjpg process handle returned")
    else:
        print("[MAIN] --no-mjpg set, skipping mjpg-streamer")

    # processing thread 시작 (sock 전달 -> processing에서 send_only 사용 가능)
    mjpg_url = f"http://127.0.0.1:{args.mjpg_port}/?action=stream"
    proc_thread = threading.Thread(
        target=processing_loop,
        args=(app_state, sock, mjpg_url, args.cam_index, True),
        daemon=True
    )
    proc_thread.start()
    print("[MAIN] processing thread started")

    # Flask 시작
    app = create_app(app_state)
    flask_thread = run_flask_in_thread(app, port=args.proc_port)
    print(f"[HTTP] processed MJPEG available at http://0.0.0.0:{args.proc_port}/processed")

    # 시그널 처리
    def sig_handler(sig, frame):
        print("[MAIN] signal received, stopping...")
        app_state.stop_event.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 메인 루프(정지 신호 대기)
    try:
        while not app_state.stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        app_state.stop_event.set()

    # 종료 처리
    print("[MAIN] shutting down")
    try:
        if mjpg_proc:
            kill_process_group(mjpg_proc)
    except Exception as e:
        print("[MAIN] kill mjpg error:", e)

    try:
        proc_thread.join(timeout=2.0)
    except Exception:
        pass

    try:
        if sock:
            sock.close()
    except Exception:
        pass

    print("[MAIN] exited cleanly")


if __name__ == "__main__":
    main()
