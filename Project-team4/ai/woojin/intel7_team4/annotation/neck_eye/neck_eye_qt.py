#!/usr/bin/env python3
# single-threaded attendance + monitoring
# 인증 -> ATTENDANCE -> 모니터링 (no threads)
# 수정: GStreamer/V4L2 기반 카메라 탐색 통합 (try_open_capture / ensure_camera_available)

import os
import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import re
import sys
import math
import argparse
import fcntl
import traceback
import glob
import subprocess

# -------- CONFIG ---------
HOST = "192.168.0.158"
PORT = 5000
SOCK_TIMEOUT = 5.0

LOGIN_MSG = "SEOL_SQL:PASSWD"
SRC = "AI"
USER = "seol"
AUTH_CMD = "AUTH"
TAG_VALUE = "VALUE"
TAG_FLAG  = "FLAG"

state_on = "ON"
state_off = "OFF"
SERVER_MSG_ON  = f"{SRC}:{USER}:SLP:{state_on}"
SERVER_MSG_OFF = f"{SRC}:{USER}:SLP:{state_off}"
POSTURE_MSG_BAD = f"{SRC}:{USER}:POSTURE:BAD:neck"

# 기본 인덱스 (숫자), 하지만 CAM_DEVICE env로 특정 /dev/videoX 지정 가능
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
CAM_DEVICE = os.environ.get("CAM_DEVICE", "")  # ex "/dev/video1"
DEV_NODE = f"/dev/video{CAM_INDEX}"

# tolerances
TOL_EYE_EYE_RATIO   = 0.05
TOL_NOSE_LIPS_RATIO = 0.05

EAR_THRESHOLD = 0.20

ANGLE_THRESHOLD = 15.0
VIS_THRESH_POSE = 0.35
POSE_PERIOD = 10.0
POSE_WINDOW = 3.0
POSE_SAMPLE_INTERVAL = 0.5

DESIRED_CLOSED_SECS = 0.5
DESIRED_ALARM_SECS = 5.0

LOCKFILE = "/tmp/neck_eye.lock"

# -------- global parameters populated from server (defaults) ----------
FOREHEAD_CHIN = 1.0
NOSE_LIPS = 0.0
EYE_EYE = 0.0
FACE_FLAG = 0
EAR_THRESHOLD = EAR_THRESHOLD

# -------- regex for parsing server lines ----------
_value_re = re.compile(r"^SERVER(?::[^:]+)*:VALUE:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$")
_flag_re  = re.compile(r"^SERVER(?::[^:]+)*:FLAG:(?P<face>[01])$")
_att_ok_re = re.compile(r"(?:SERVER:)?(?:[^:]+:)?ATTENDANCE:OK")

# ---------------- camera helpers (NEW) ----------------
def run_cmd(cmd: str):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return 0, out.decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        return e.returncode, (e.output.decode('utf-8', errors='ignore') if e.output else "")

def list_video_devs():
    # returns sorted list of /dev/video* paths (numeric order)
    devs = sorted(glob.glob("/dev/video*"),
                  key=lambda p: int(''.join([c for c in p if c.isdigit()]) or 0))
    return devs

def try_open_capture(timeout=1.0):
    """
    Try to open a VideoCapture in this order for each candidate device:
     1) GStreamer MJPEG pipeline (v4l2src ! image/jpeg ! jpegdec ! appsink) using CAP_GSTREAMER
     2) cv2.VideoCapture(dev, cv2.CAP_V4L2)
     3) cv2.VideoCapture(dev) default
     4) numeric indices fallback (0..4)
    Returns opened cv2.VideoCapture or None.
    """
    candidates = []
    if CAM_DEVICE:
        candidates.append(CAM_DEVICE)
    # add /dev/video* devices
    for d in list_video_devs():
        if d not in candidates:
            candidates.append(d)
    # also try numeric indices (as str/ints) fallback
    for i in range(0, 6):
        s = str(i)
        if s not in candidates:
            candidates.append(s)

    for dev in candidates:
        # A: try GStreamer MJPEG pipeline if dev looks like /dev/video*
        if isinstance(dev, str) and dev.startswith("/dev/"):
            gst_str = (
                f"v4l2src device={dev} ! "
                f"image/jpeg,width=640,height=480,framerate=15/1 ! "
                f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            try:
                cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    # quick read verification
                    start = time.time()
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    while time.time() - start < timeout:
                        ret, _ = cap.read()
                        if ret:
                            print(f"[CAM] opened via GStreamer MJPEG on {dev}")
                            return cap
                        time.sleep(0.02)
                    cap.release()
            except Exception:
                try:
                    cap.release()
                except:
                    pass

        # B: try V4L2 backend explicitly
        try:
            if isinstance(dev, str) and dev.startswith("/dev/"):
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            else:
                # numeric index
                try:
                    idx = int(dev)
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                except Exception:
                    cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                start = time.time()
                while time.time() - start < timeout:
                    ret, _ = cap.read()
                    if ret:
                        print(f"[CAM] opened via V4L2/default on {dev}")
                        return cap
                    time.sleep(0.02)
                cap.release()
        except Exception:
            try:
                cap.release()
            except:
                pass

    print("[CAM] try_open_capture: no working camera found")
    return None

def ensure_camera_available(timeout: float = 12.0):
    """
    Try repeatedly until timeout to get an opened cv2.VideoCapture.
    Returns cap or None.
    """
    print(f"[CAM] ensure_camera_available(timeout={timeout})")
    end = time.time() + timeout
    while time.time() < end:
        cap = try_open_capture(timeout=1.0)
        if cap:
            return cap
        # optional: list holders for debugging
        rc1, out1 = run_cmd(f"sudo lsof {DEV_NODE} || true")
        rc2, out2 = run_cmd(f"sudo fuser -v {DEV_NODE} || true")
        if out1.strip() or out2.strip():
            print("[CAM] holders info (lsof/fuser):")
            print("--- lsof ---\n", out1)
            print("--- fuser ---\n", out2)
        time.sleep(0.8)
    print("[CAM] ensure_camera_available timed out.")
    return None

# ---------------- other utilities ----------------
def acquire_single_instance_lock():
    try:
        fp = open(LOCKFILE, "w")
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fp.write(str(os.getpid()))
        fp.flush()
        print(f"[LOCK] acquired ({LOCKFILE}), pid={os.getpid()}")
        return fp
    except BlockingIOError:
        print("[LOCK] another instance is running. exit.")
        sys.exit(1)

def release_single_instance_lock(fp):
    try:
        if fp:
            fcntl.flock(fp, fcntl.LOCK_UN)
            fp.close()
            try: os.remove(LOCKFILE)
            except: pass
            print("[LOCK] released")
    except Exception:
        pass

def send_only(sock, msg: str):
    try:
        print(f"[NET SEND] {msg}")
        sock.sendall(msg.encode('utf-8'))
    except Exception as e:
        print(f"[NET] send error: {e}")

# recv-lines helper: accumulate until newline and return full lines
def wait_for_pattern(sock, regex, timeout=None):
    end = None if timeout is None else (time.time() + timeout)
    buf = b""
    sock.settimeout(1.0)
    while True:
        if end and time.time() > end:
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
            sline = line.decode('utf-8', errors='ignore').strip()
            if regex.search(sline) or regex.match(sline):
                return sline
        # partial
        try:
            s = buf.decode('utf-8', errors='ignore')
            if regex.search(s) or regex.match(s):
                return s
        except:
            pass

# ---------- face & measure ----------
mp_face = mp.solutions.face_mesh
IDX_LEFT_EYE_OUT  = 133
IDX_RIGHT_EYE_OUT = 362
IDX_NOSE_TIP      = 1
IDX_MOUTH_CENTER  = 13
IDX_FOREHEAD      = 10
IDX_CHIN          = 152

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def measure(lm, w, h):
    def pt(i): return (lm[i].x * w, lm[i].y * h)
    try:
        fore = dist(pt(IDX_FOREHEAD), pt(IDX_CHIN))
        nose = dist(pt(IDX_NOSE_TIP), pt(IDX_MOUTH_CENTER))
        eye  = dist(pt(IDX_LEFT_EYE_OUT), pt(IDX_RIGHT_EYE_OUT))
    except Exception:
        return None
    if fore < 1e-6:
        return None
    return {
        "forehead_chin": fore,
        "nose_lips": nose,
        "eye_eye": eye,
        "eye_over_fc": eye / (fore + 1e-9),
        "nose_over_fc": nose / (fore + 1e-9)
    }

def simple_face_match_server_order(server_fore, server_nose, server_eye, obs):
    if obs is None:
        print("[MATCH] obs None -> cannot match")
        return False
    try:
        if float(server_fore) <= 0 or float(server_nose) <= 0 or float(server_eye) <= 0:
            print("[MATCH] server VALUE invalid (<=0) -> skip matching")
            return False
    except:
        return False

    obs_eye_px = obs["eye_eye"]
    obs_nose_px = obs["nose_lips"]
    obs_fc_px = obs["forehead_chin"]

    sv_fc_px = float(server_fore)
    sv_nose_px = float(server_nose)
    sv_eye_px = float(server_eye)

    sv_eye_ratio = sv_eye_px / sv_fc_px if sv_fc_px != 0 else float('inf')
    sv_nose_ratio = sv_nose_px / sv_fc_px if sv_fc_px != 0 else float('inf')
    ov_eye_ratio = obs_eye_px / obs_fc_px if obs_fc_px != 0 else float('inf')
    ov_nose_ratio = obs_nose_px / obs_fc_px if obs_fc_px != 0 else float('inf')

    def within_ratio(a, b, tol):
        if b == 0:
            return False
        return abs(a-b) <= tol * abs(b)

    eye_ok  = within_ratio(ov_eye_ratio, sv_eye_ratio, TOL_EYE_EYE_RATIO)
    nose_ok = within_ratio(ov_nose_ratio, sv_nose_ratio, TOL_NOSE_LIPS_RATIO)
    overall = eye_ok and nose_ok

    eye_diff_px = obs_eye_px - sv_eye_px
    nose_diff_px = obs_nose_px - sv_nose_px
    eye_pct = (eye_diff_px / sv_eye_px * 100.0) if sv_eye_px != 0 else float('inf')
    nose_pct = (nose_diff_px / sv_nose_px * 100.0) if sv_nose_px != 0 else float('inf')

    print(
        f"[MATCH] sv_fc_px={sv_fc_px:.1f}px sv_nose_px={sv_nose_px:.1f}px sv_eye_px={sv_eye_px:.1f}px | "
        f"obs_fc_px={obs_fc_px:.1f}px obs_nose_px={obs_nose_px:.1f}px obs_eye_px={obs_eye_px:.1f}px || "
        f"eye_diff={eye_diff_px:+.1f}px ({eye_pct:+.1f}%), nose_diff={nose_diff_px:+.1f}px ({nose_pct:+.1f}%) || "
        f"sv_eye_ratio={sv_eye_ratio:.3f} sv_nose_ratio={sv_nose_ratio:.3f} obs_eye_ratio={ov_eye_ratio:.3f} obs_nose_ratio={ov_nose_ratio:.3f} || "
        f"eye_ok={eye_ok} nose_ok={nose_ok} => MATCH={overall}"
    )
    return overall

# ---------- EAR utils ----------
LEFT = [33,160,158,133,153,144]
RIGHT= [263,387,385,362,380,373]

def compute_ear(lm, idxs, h, w):
    try:
        pts = [(lm[i].x*w, lm[i].y*h) for i in idxs]
    except Exception:
        return None
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
    return (A+B)/(2.0*C)

# ---------- Main flow (single-threaded) ----------
def main(force_kill: bool):
    global FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE, FACE_FLAG, EAR_THRESHOLD

    lock_fp = acquire_single_instance_lock()
    sock = None
    face_mesh = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SOCK_TIMEOUT)
        sock.connect((HOST, PORT))
        print(f"[NET] connected to {HOST}:{PORT}")
        # login
        send_only(sock, LOGIN_MSG)
        try:
            sock.settimeout(1.0)
            data = sock.recv(4096)
            if data:
                print(f"[NET LOGIN ACK] {data.decode('utf-8',errors='ignore').strip()}")
        except socket.timeout:
            pass
        finally:
            sock.settimeout(None)

        # request FLAG
        send_only(sock, f"{SRC}:{USER}:{AUTH_CMD}:{TAG_FLAG}")
        # wait for FLAG in-line
        line = wait_for_pattern(sock, _flag_re, timeout=5.0)
        if line:
            m = _flag_re.match(line)
            if m:
                FACE_FLAG = int(m.group("face"))
                print(f"[Main] initial FACE_FLAG = {FACE_FLAG}")
        else:
            print("[Main] FLAG not received (timeout). Assuming FACE_FLAG==0")
            FACE_FLAG = 0

        # request VALUE
        send_only(sock, f"{SRC}:{USER}:{AUTH_CMD}:{TAG_VALUE}")
        line = wait_for_pattern(sock, _value_re, timeout=10.0)
        if line:
            m = _value_re.match(line)
            if m:
                FOREHEAD_CHIN = float(m.group("fore"))
                NOSE_LIPS = float(m.group("nose"))
                EYE_EYE = float(m.group("eye"))
                EAR_THRESHOLD = float(m.group("ear"))
                print(f"[VALUE SET] fore={FOREHEAD_CHIN}, nose={NOSE_LIPS}, eye={EYE_EYE}, ear_thr={EAR_THRESHOLD}")
        else:
            print("[Main] VALUE not received (timeout). Using defaults.")

        # create FaceMesh once and reuse (important on Jetson/TFLite)
        print("[Main] initializing FaceMesh (single instance reuse)")
        face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                     refine_landmarks=True, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)

        # AUTH step if needed
        auth_cap = None
        if FACE_FLAG == 0:
            print("[AUTH] start camera for face matching (try_open_capture)...")
            auth_cap = try_open_capture(timeout=2.0)
            if auth_cap is None:
                print("[AUTH] camera open failed for auth. aborting auth (will continue to monitoring).")
            else:
                win = "Auth (ESC cancel)"
                headless = False
                try:
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                except Exception:
                    headless = True; win = None
                MATCH_REQUIRED = 3
                match_count = 0
                sent_face_ok = False
                while not sent_face_ok:
                    ret, frame = auth_cap.read()
                    if not ret or frame is None:
                        time.sleep(0.02); continue
                    img = cv2.flip(frame, 1)
                    h, w = img.shape[:2]
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = face_mesh.process(rgb)
                    if res and res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        obs = measure(lm, w, h)
                        if obs:
                            matched = simple_face_match_server_order(FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE, obs)
                            if not headless:
                                cv2.putText(img, f"match={'OK' if matched else 'NO'} ({match_count})", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if matched else (0,0,255), 2)
                            if matched:
                                match_count += 1
                                if match_count >= MATCH_REQUIRED:
                                    msg = f"{SRC}:{USER}:FACE:OK"
                                    send_only(sock, msg)
                                    print("[AUTH] sent FACE:OK")
                                    sent_face_ok = True
                                    FACE_FLAG = 1
                                    break
                            else:
                                if match_count > 0:
                                    print(f"[AUTH] match broken, resetting counter (was {match_count})")
                                match_count = 0
                        else:
                            match_count = 0
                    else:
                        match_count = 0
                        if not headless:
                            try:
                                cv2.putText(img, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)
                            except:
                                pass
                    if not headless:
                        cv2.imshow(win, img)
                        if cv2.waitKey(1) & 0xFF == 27:
                            print("[AUTH] cancelled by user")
                            break
                try:
                    auth_cap.release()
                except:
                    pass
                try:
                    if not headless and win is not None:
                        cv2.destroyWindow(win)
                except:
                    pass

            # after sending FACE:OK, wait for ATTENDANCE
            print("[AUTH] waiting for ATTENDANCE:OK (server)")
            line = wait_for_pattern(sock, _att_ok_re, timeout=15.0)
            if line:
                print("[Main] ATTENDANCE received:", line)
            else:
                print("[Main] ATTENDANCE not received in time; continuing to monitoring")

        else:
            print("[Main] FACE_FLAG==1 -> already authenticated, waiting ATTENDANCE:OK")
            line = wait_for_pattern(sock, _att_ok_re, timeout=15.0)
            if line:
                print("[Main] ATTENDANCE received:", line)
            else:
                print("[Main] ATTENDANCE not received (timeout). continuing to monitoring")

        # ---------- Monitoring ----------
        print("[MON] entering monitoring")
        cap = ensure_camera_available(timeout=12.0)
        if cap is None:
            print("[MON] camera not available -> exiting monitoring")
            return

        # pose model (create here)
        pose = mp.solutions.pose.Pose(model_complexity=0, static_image_mode=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[MON] cannot read first frame -> abort")
            try: cap.release()
            except: pass
            pose.close()
            return

        win = "Monitoring (ESC exit)"
        headless = False
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except Exception:
            headless = True; win = None

        THRESH = float(EAR_THRESHOLD)
        ema = None; alpha = 0.35
        closed_cnt = 0; closed_total = 0; alarm_sent = False
        fps = 30; prev = time.time(); framec = 0
        period_anchor = time.time()
        last_sample_t = 0.0
        neck_samples = []
        last_window_max = None
        posture_sent = False
        last_hb = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                img = cv2.flip(frame,1)
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                framec += 1
                now = time.time()
                if now - prev >= 1.0:
                    fps = framec; framec = 0; prev = now

                if now - last_hb >= 2.0:
                    print(f"[HB] monitoring alive fps={fps} THRESH={THRESH:.2f}")
                    last_hb = now

                status = "NO FACE"; scol=(150,150,150)
                try:
                    res = face_mesh.process(rgb)
                except Exception as e:
                    print(f"[MON] face_mesh.process exc: {e}")
                    res = None

                raw = None
                if res and getattr(res, "multi_face_landmarks", None):
                    lm = res.multi_face_landmarks[0].landmark
                    le = compute_ear(lm, LEFT, h, w)
                    re = compute_ear(lm, RIGHT, h, w)
                    if le is not None and re is not None:
                        raw = (le+re)/2.0
                    elif le is not None:
                        raw = le
                    elif re is not None:
                        raw = re
                    if raw is not None:
                        ema = raw if ema is None else alpha*raw + (1-alpha)*ema
                        closed_thr = max(1, int(round(DESIRED_CLOSED_SECS * fps)))
                        alarm_thr  = max(1, int(round(DESIRED_ALARM_SECS * fps)))
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

                if status == "DROWSY ALERT!" and not alarm_sent:
                    send_only(sock, SERVER_MSG_ON)
                    alarm_sent = True
                if alarm_sent and status != "DROWSY ALERT!":
                    send_only(sock, SERVER_MSG_OFF)
                    alarm_sent = False

                # posture window
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
                        try:
                            pr = pose.process(rgb)
                        except Exception:
                            pr = None
                        if pr and getattr(pr, "pose_landmarks", None):
                            pl = pr.pose_landmarks.landmark
                            cand = []
                            for idx in (mp.solutions.pose.PoseLandmark.NOSE.value,
                                        mp.solutions.pose.PoseLandmark.LEFT_EAR.value,
                                        mp.solutions.pose.PoseLandmark.RIGHT_EAR.value):
                                if pl[idx].visibility > VIS_THRESH_POSE:
                                    cand.append((pl[idx].x, pl[idx].y))
                            used = len(cand)
                            if used > 0:
                                sx = sum(p[0] for p in cand)/used
                                sy = sum(p[1] for p in cand)/used
                                ls = pl[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                                rs = pl[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                                if ls.visibility>VIS_THRESH_POSE and rs.visibility>VIS_THRESH_POSE:
                                    nx = (ls.x+rs.x)/2.0; ny=(ls.y+rs.y)/2.0
                                    vx = sx-nx; vy = sy-ny
                                    if math.hypot(vx,vy) > 1e-6:
                                        ang = abs(math.degrees(math.atan2(vx,-vy)))
                                        neck_samples.append(ang)
                                        running_max = max(neck_samples)
                                        if win is not None:
                                            cv2.line(img,(int(nx*w),int(ny*h)),(int(sx*w),int(sy*h)),(255,0,255),3)
                                            cv2.circle(img,(int(nx*w),int(ny*h)),5,(0,255,255),-1)
                                            cv2.circle(img,(int(sx*w),int(sy*h)),5,(255,0,0),-1)
                else:
                    if neck_samples:
                        last_window_max = max(neck_samples)
                        neck_samples = []
                        if last_window_max > ANGLE_THRESHOLD and not posture_sent:
                            send_only(sock, POSTURE_MSG_BAD)
                            posture_sent = True
                        if last_window_max <= ANGLE_THRESHOLD and posture_sent:
                            posture_sent = False

                # draw & show
                if win is not None:
                    try:
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
                            print("[MON] ESC pressed -> exit monitoring")
                            break
                    except Exception as e:
                        print(f"[MON] imshow failed -> headless mode: {e}")
                        win = None
                else:
                    time.sleep(0.002)
        finally:
            try: cap.release()
            except: pass
            pose.close()
            try:
                if win is not None:
                    cv2.destroyWindow(win)
            except:
                pass

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt - exiting")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            if face_mesh:
                face_mesh.close()
        except:
            pass
        try:
            if sock:
                sock.close()
        except:
            pass
        release_single_instance_lock(lock_fp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-kill", action="store_true", help="not used in single-threaded version")
    args = ap.parse_args()
    main(force_kill=args.force_kill)
