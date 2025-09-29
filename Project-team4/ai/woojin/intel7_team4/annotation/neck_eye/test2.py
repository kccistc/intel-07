#!/usr/bin/env python3
# main_attendance_blocking.py
# Server-listen (blocking) for ATTENDANCE:OK -> then run monitoring

import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import time
import re
import sys
import math

# ---------------- CONFIG ----------------
HOST = "192.168.0.158"
PORT = 5000
SOCK_TIMEOUT = None   # Listener uses blocking recv -> use None
LOGIN_MSG = "SEOL_SQL:PASSWD"

SRC = "AI"
USER = "seol"
AUTH_CMD = "AUTH"
TAG_VALUE = "VALUE"
TAG_FLAG  = "FLAG"

state_on = "ON"
state_off = "OFF"

src = "AI"
user = "seol"
type = "SLP"

SERVER_MSG_ON  = f"{src}:{user}:{type}:{state_on}"
SERVER_MSG_OFF = f"{src}:{user}:{type}:{state_off}"
# posture alert 메시지 (요구한 형식)
POSTURE_MSG_BAD = "AI:SEOL_SQL:POSTURE:BAD:neck"

CAM_INDEX = 0

# matching tolerances (ratio)
TOL_EYE_EYE_RATIO   = 0.25
TOL_NOSE_LIPS_RATIO = 0.25

# EAR default (may be overwritten by server VALUE)
EAR_THRESHOLD = 0.20

# turtle neck
ANGLE_THRESHOLD = 15.0
VIS_THRESH_POSE = 0.35
POSE_PERIOD = 10.0
POSE_WINDOW = 3.0
POSE_SAMPLE_INTERVAL = 0.5

# drowsy timing
DESIRED_CLOSED_SECS = 0.5
DESIRED_ALARM_SECS = 5.0

# ---------------- globals updated from server --------------
EYE_EYE = 0.0
NOSE_LIPS = 0.0
FOREHEAD_CHIN = 1.0
FACE_FLAG = 0

# events & socket
_sock = None
listener_thread_obj = None
listener_stop_event = threading.Event()
flag_event = threading.Event()
value_event = threading.Event()
attendance_event = threading.Event()

# -------------------------------------------
# regexes
_value_re = re.compile(r"^SERVER:VALUE:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$")
_flag_re  = re.compile(r"^SERVER:FLAG:(?P<face>[01])$")
# -------------------------------------------

def connect_sock(host=HOST, port=PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # blocking socket (no timeout) for listener
    s.settimeout(5.0)  # short timeout for initial connect/recv; we'll set blocking later for listener
    s.connect((host, port))
    return s

def send_only(sock, msg: str):
    try:
        print(f"[NET SEND] {msg}")
        sock.sendall(msg.encode('utf-8'))
    except Exception as e:
        print(f"[NET] send error: {e}")

def parse_server_line(line: str):
    """Parse server line and update globals/events as needed."""
    global EYE_EYE, NOSE_LIPS, FOREHEAD_CHIN, EAR_THRESHOLD, FACE_FLAG
    line = line.strip()
    if not line:
        return
    print(f"[NET RX_LINE] {line}")

    # check VALUE
    m = _value_re.match(line)
    if m:
        try:
            EYE_EYE = float(m.group("eye"))
            NOSE_LIPS = float(m.group("nose"))
            FOREHEAD_CHIN = float(m.group("fore"))
            EAR_THRESHOLD = float(m.group("ear"))
            print(f"[VALUE SET] eye={EYE_EYE}, nose={NOSE_LIPS}, fore={FOREHEAD_CHIN}, ear_thr={EAR_THRESHOLD}")
            value_event.set()
        except Exception as e:
            print(f"[VALUE parse err] {e}")
        return

    # check FLAG (face only)
    m2 = _flag_re.match(line)
    if m2:
        try:
            FACE_FLAG = int(m2.group("face"))
            print(f"[FLAG SET] face={FACE_FLAG}")
            flag_event.set()
        except Exception as e:
            print(f"[FLAG parse err] {e}")
        return

    # check ATTENDANCE OK (any form containing ATTENDANCE:OK)
    if "ATTENDANCE:OK" in line:
        print("[ATTENDANCE] OK detected")
        attendance_event.set()
        return

    # else: other messages - just log
    print(f"[NET] unhandled message: {line}")


def listener_loop(sock, stop_ev: threading.Event):
    """
    Blocking recv listener thread.
    Reads from sock.recv (blocking) and splits by lines to parse.
    """
    print("[Listener] started (blocking recv)...")
    # switch socket to blocking
    try:
        sock.settimeout(None)
    except:
        pass

    buf = b""
    while not stop_ev.is_set():
        try:
            data = sock.recv(4096)
            if not data:
                print("[Listener] socket closed by remote or empty recv")
                break
            buf += data
            # split by newline(s) OR handle everything as lines by '\n'
            # server may not include newline; still handle per-line by splitting on '\n'
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    sline = line.decode('utf-8', errors='ignore')
                except:
                    sline = repr(line)
                parse_server_line(sline)
            # if no newline, try to parse whole buffer if it's reasonable text
            # some servers return single-line messages without newline; try to parse if buffer grows
            if len(buf) > 0 and b"\n" not in buf and len(buf) < 4096:
                try:
                    s = buf.decode('utf-8', errors='ignore').strip()
                    # attempt parse if looks complete (contains SERVER:... or ATTENDANCE)
                    if s.startswith("SERVER:") or "ATTENDANCE:OK" in s:
                        parse_server_line(s)
                        buf = b""
                except:
                    pass
        except Exception as e:
            if stop_ev.is_set():
                break
            print(f"[Listener] recv error: {e}. Sleeping briefly and retrying.")
            time.sleep(0.2)
            continue

    print("[Listener] exiting")


# ---------------- face measurement / match (indices fixed) ----------------
mp_face = mp.solutions.face_mesh
_face_mesh_global = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

IDX_LEFT_EYE_OUT  = 133
IDX_RIGHT_EYE_OUT = 362
IDX_NOSE_TIP      = 1
IDX_MOUTH_CENTER  = 13
IDX_FOREHEAD      = 10
IDX_CHIN          = 152

def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def measure_face_metrics(landmarks, h, w):
    def pt(i): return (landmarks[i].x * w, landmarks[i].y * h)
    try:
        eye_eye = _dist(pt(IDX_LEFT_EYE_OUT), pt(IDX_RIGHT_EYE_OUT))
        nose_lips = _dist(pt(IDX_NOSE_TIP), pt(IDX_MOUTH_CENTER))
        forehead_chin = _dist(pt(IDX_FOREHEAD), pt(IDX_CHIN))
    except Exception as e:
        return None
    if forehead_chin < 1e-6:
        return None
    return {
        "eye_eye": eye_eye,
        "nose_lips": nose_lips,
        "forehead_chin": forehead_chin,
        "eye_over_fc": eye_eye / forehead_chin,
        "nose_over_fc": nose_lips / forehead_chin,
    }

def simple_face_match(server_vals, obs_vals):
    sv_eye, sv_nose, sv_fc = server_vals
    if sv_fc < 1e-6:
        return False
    sv_eye_ratio = sv_eye / sv_fc
    sv_nose_ratio = sv_nose / sv_fc
    ov_eye_ratio = obs_vals["eye_over_fc"]
    ov_nose_ratio = obs_vals["nose_over_fc"]
    def within(a,b,tol): return abs(a-b) <= tol * b
    ok_eye  = within(ov_eye_ratio, sv_eye_ratio, TOL_EYE_EYE_RATIO)
    ok_nose = within(ov_nose_ratio, sv_nose_ratio, TOL_NOSE_LIPS_RATIO)
    print(f"[MATCH] sv_eye_r={sv_eye_ratio:.3f}, sv_nose_r={sv_nose_ratio:.3f}, obs_eye={ov_eye_ratio:.3f}, obs_nose={ov_nose_ratio:.3f} -> eye_ok={ok_eye}, nose_ok={ok_nose}")
    return ok_eye and ok_nose

# ---------------- auth worker (face capture & match) ----------------
def auth_worker(sock, stop_ev):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Auth] camera open failed"); stop_ev.set(); return
    win = "Auth (ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[Auth] started - performing face matching (press ESC to cancel)...")
    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = _face_mesh_global.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # draw debug points
            for i in (IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT, IDX_NOSE_TIP, IDX_MOUTH_CENTER, IDX_FOREHEAD, IDX_CHIN):
                cx, cy = int(lm[i].x*w), int(lm[i].y*h)
                cv2.circle(img, (cx,cy), 3, (0,255,255), -1)
            # Only try match if FACE_FLAG == 0
            if FACE_FLAG == 0:
                obs = measure_face_metrics(lm, h, w)
                if obs:
                    matched = simple_face_match((EYE_EYE, NOSE_LIPS, FOREHEAD_CHIN), obs)
                    cv2.putText(img, f"match={'OK' if matched else 'NO'}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if matched else (0,0,255), 2)
                    if matched:
                        msg = f"{SRC}:{USER}:FACE:OK"
                        print(f"[Auth] match -> sending: {msg}")
                        send_only(sock, msg)
                        # after sending FACE:OK we still wait for attendance_event (listener will set)
        else:
            cv2.putText(img, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)

        cv2.imshow(win, img)
        if cv2.waitKey(1) & 0xFF == 27:
            print("[Auth] cancelled by user")
            stop_ev.set()
            break

    cap.release()
    cv2.destroyWindow(win)
    print("[Auth] ended")

# ---------------- monitoring (drowsy + turtle-neck) ----------------
LEFT = [33,160,158,133,153,144]
RIGHT= [263,387,385,362,380,373]

def compute_ear(lm, idxs, h, w):
    try:
        pts = [(lm[i].x*w, lm[i].y*h) for i in idxs]
    except:
        return None
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
    return (A+B)/(2.0*C)

def send_alert(sock, msg):
    try:
        print(f"[ALERT SEND] {msg}")
        send_only(sock, msg)
    except Exception as e:
        print(f"[ALERT] send err: {e}")

def run_monitoring(sock):
    print("[Monitor] start")
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp.solutions.pose.Pose(model_complexity=1, static_image_mode=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Monitor] camera open failed"); return

    win = "Monitoring (ESC to exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    THRESH = float(EAR_THRESHOLD)
    ema = None; alpha = 0.35
    closed_cnt = 0; closed_total = 0; alarm_sent = False
    fps = 30; prev = time.time(); framec = 0

    period_anchor = time.time()
    last_sample_t = 0.0
    neck_samples = []
    last_window_max = None
    posture_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        img = cv2.flip(frame,1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        framec += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = framec; framec = 0; prev = now
            closed_thr = max(1, int(round(DESIRED_CLOSED_SECS * fps)))
            alarm_thr  = max(1, int(round(DESIRED_ALARM_SECS * fps)))

        # face mesh & EAR
        status = "NO FACE"; scol=(150,150,150)
        res = face_mesh.process(rgb)
        raw = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            le = compute_ear(lm, LEFT, h, w)
            re = compute_ear(lm, RIGHT, h, w)
            if le is not None and re is not None: raw = (le+re)/2.0
            elif le is not None: raw = le
            elif re is not None: raw = re
            if raw is not None:
                ema = raw if ema is None else alpha*raw + (1-alpha)*ema
                if ema < THRESH:
                    closed_cnt += 1; closed_total += 1
                else:
                    closed_cnt = 0; closed_total = 0
                if closed_total >= alarm_thr:
                    status = "DROWSY ALERT!"; scol=(0,0,255)
                elif closed_cnt >= closed_thr:
                    status = "EYES CLOSED"; scol=(0,255,255)
                else:
                    status = "EYES OPEN"; scol=(0,255,0)
        else:
            closed_cnt = 0; closed_total = 0; status="NO FACE"; scol=(180,180,180)

        # send drowsy alerts
        if status == "DROWSY ALERT!" and not alarm_sent:
            send_alert(sock, SERVER_MSG_ON)
            alarm_sent = True
        if alarm_sent and status != "DROWSY ALERT!":
            send_alert(sock, SERVER_MSG_OFF)
            alarm_sent = False

        # pose sampling window
        elapsed = now - period_anchor
        if elapsed >= POSE_PERIOD:
            n = int(elapsed // POSE_PERIOD)
            period_anchor += n * POSE_PERIOD
            neck_samples = []
            elapsed = now - period_anchor
        in_window = (0.0 <= elapsed < POSE_WINDOW)

        running_max = None
        if in_window:
            if now - last_sample_t >= POSE_SAMPLE_INTERVAL:
                last_sample_t = now
                pr = pose.process(rgb)
                if pr.pose_landmarks:
                    pl = pr.pose_landmarks.landmark
                    cand = []
                    for idx in (mp.solutions.pose.PoseLandmark.NOSE.value,
                                mp.solutions.pose.PoseLandmark.LEFT_EAR.value,
                                mp.solutions.pose.PoseLandmark.RIGHT_EAR.value):
                        if pl[idx].visibility > VIS_THRESH_POSE:
                            cand.append((pl[idx].x, pl[idx].y))
                    used = len(cand)
                    if used>0:
                        sx = sum(p[0] for p in cand)/used
                        sy = sum(p[1] for p in cand)/used
                        ls = pl[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                        rs = pl[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                        if ls.visibility>VIS_THRESH_POSE and rs.visibility>VIS_THRESH_POSE:
                            nx = (ls.x+rs.x)/2.0; ny=(ls.y+rs.y)/2.0
                            vx = sx-nx; vy = sy-ny
                            if math.hypot(vx,vy)>1e-6:
                                ang = abs(math.degrees(math.atan2(vx,-vy)))
                                neck_samples.append(ang)
                                running_max = max(neck_samples)
                                cv2.line(img,(int(nx*w),int(ny*h)),(int(sx*w),int(sy*h)),(255,0,255),3)
                                cv2.circle(img,(int(nx*w),int(ny*h)),5,(0,255,255),-1)
                                cv2.circle(img,(int(sx*w),int(sy*h)),5,(255,0,0),-1)
        else:
            if neck_samples:
                last_window_max = max(neck_samples)
                neck_samples = []
                if last_window_max > ANGLE_THRESHOLD and not posture_sent:
                    send_alert(sock, f"{SRC}:{USER}:POSTURE:BAD:neck")
                    posture_sent = True
                if last_window_max <= ANGLE_THRESHOLD and posture_sent:
                    posture_sent = False

        # overlay
        cv2.putText(img, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, scol, 2)
        neck_disp = "Neck: -"
        nc = (150,150,150)
        if running_max is not None:
            neck_disp = f"Neck: {running_max:.1f}deg (win)"; nc = (0,255,0) if running_max<=ANGLE_THRESHOLD else (0,0,255)
        elif last_window_max is not None:
            neck_disp = f"Neck: {last_window_max:.1f}deg"; nc = (0,255,0) if last_window_max<=ANGLE_THRESHOLD else (0,0,255)
        cv2.putText(img, neck_disp, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, nc, 2)
        cv2.putText(img, f"FPS:{fps}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50), 2)

        cv2.imshow(win, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    face_mesh.close()
    pose.close()
    cv2.destroyWindow(win)
    print("[Monitor] stopped")

# ---------------- main flow ----------------
def main():
    global _sock, listener_thread_obj

    # connect
    try:
        _sock = connect_sock(HOST, PORT)
        print(f"[NET] connected to {HOST}:{PORT}")
    except Exception as e:
        print(f"[NET] connect error: {e}")
        sys.exit(1)

    # login (send only)
    try:
        send_only(_sock, LOGIN_MSG)
        # attempt to read immediate ack briefly (non-blocking way)
        try:
            _sock.settimeout(1.0)
            data = _sock.recv(4096)
            if data:
                print(f"[NET LOGIN ACK] {data.decode('utf-8',errors='ignore').strip()}")
        except socket.timeout:
            pass
        finally:
            _sock.settimeout(None)  # listener wants blocking
    except Exception as e:
        print(f"[NET] login send err: {e}")
        _sock.close(); sys.exit(1)

    # start listener thread (blocking recv)
    listener_stop_event.clear()
    listener_thread_obj = threading.Thread(target=listener_loop, args=(_sock, listener_stop_event), daemon=True)
    listener_thread_obj.start()

    # 1) request FLAG (send)
    send_only(_sock, f"{SRC}:{USER}:{AUTH_CMD}:{TAG_FLAG}")
    # wait for flag_event (listener will set when receives SERVER:FLAG)
    print("[Main] waiting for FLAG from server...")
    flag_event.wait(timeout=10.0)  # wait up to 10s for a FLAG response (non-mandatory)
    if flag_event.is_set():
        print(f"[Main] FACE_FLAG now = {FACE_FLAG}")
    else:
        print("[Main] FLAG not received within 10s (continue waiting for attendance via listener)")

    # If face already authenticated (FACE_FLAG==1) -> wait only for ATTENDANCE:OK (blocking via attendance_event)
    if FACE_FLAG == 1:
        print("[Main] FACE_FLAG==1. Now block until ATTENDANCE:OK received from server (listener will set).")
        print("[Main] (Press Ctrl+C to abort.)")
        attendance_event.wait()  # block until listener sets attendance_event
        print("[Main] ATTENDANCE:OK received -> starting monitoring.")
        run_monitoring(_sock)
        cleanup()
        return

    # FACE_FLAG == 0 -> request VALUE to get face params
    print("[Main] FACE_FLAG==0 -> requesting VALUE from server.")
    send_only(_sock, f"{SRC}:{USER}:{AUTH_CMD}:{TAG_VALUE}")

    print("[Main] waiting for VALUE from server...")
    value_event.wait(timeout=15.0)  # wait up to 15s for VALUE
    if not value_event.is_set():
        print("[Main] VALUE not received (timeout). Proceeding with defaults.")
    else:
        print("[Main] VALUE received and set.")

    # start auth worker thread (user performs face matching)
    stop_ev = threading.Event()
    t_auth = threading.Thread(target=auth_worker, args=(_sock, stop_ev), daemon=True)
    t_auth.start()

    # Now block until attendance_event set by listener (server will set when both face and other checks passed)
    print("[Main] Waiting for ATTENDANCE:OK (listener will set attendance_event).")
    try:
        attendance_event.wait()  # block indefinitely
        print("[Main] ATTENDANCE:OK received -> stopping auth and starting monitoring.")
    except KeyboardInterrupt:
        print("[Main] interrupted by user; cleaning up.")
    finally:
        stop_ev.set()
        t_auth.join(timeout=2.0)

    # start monitoring
    run_monitoring(_sock)

    cleanup()

def cleanup():
    global listener_stop_event, _sock
    print("[Cleanup] shutting down listener and closing socket...")
    try:
        listener_stop_event.set()
    except:
        pass
    try:
        if _sock:
            _sock.close()
    except:
        pass
    try:
        _face_mesh_global.close()
    except:
        pass
    print("[Exit] bye.")

if __name__ == "__main__":
    main()
