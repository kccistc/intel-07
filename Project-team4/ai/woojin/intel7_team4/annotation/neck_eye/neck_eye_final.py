#!/usr/bin/env python3
# single-threaded attendance + monitoring
# 인증 -> ATTENDANCE -> 모니터링 (no threads)

import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import re
import sys
import math
import argparse
import os
import fcntl
import traceback

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

CAM_INDEX = 0
DEV_NODE = "/dev/video0"

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

# -------- regex for parsing server lines ----------
_value_re = re.compile(r"^SERVER(?::[^:]+)*:VALUE:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$")
_flag_re  = re.compile(r"^SERVER(?::[^:]+)*:FLAG:(?P<face>[01])$")
_att_ok_re = re.compile(r"(?:SERVER:)?(?:[^:]+:)?ATTENDANCE:OK")

# -------- utilities ----------
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
def recv_lines(sock, timeout=None):
    """
    Generator yielding lines (str) as they arrive.
    Blocks until at least one line arrives or timeout.
    """
    if timeout is not None:
        sock.settimeout(timeout)
    else:
        sock.settimeout(None)

    buf = b""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                # remote closed
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode('utf-8', errors='ignore').strip()
            # also try to flush partial lines if they look complete (no newline)
            # but prefer waiting for explicit newline
        except socket.timeout:
            return
        except Exception:
            traceback.print_exc()
            return

def wait_for_pattern(sock, regex, timeout=None):
    """
    Wait until a received line matches regex.search (or pattern found in line).
    Returns the matched line or None on timeout/close.
    """
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
            # immediate match?
            if regex.search(sline):
                return sline
            # if regex is anchored groups, allow full-line match too
            if regex.match(sline):
                return sline
        # also check partial buffer for patterns without newline (rare)
        try:
            s = buf.decode('utf-8', errors='ignore')
            if regex.search(s):
                return s
        except:
            pass

# ---------- camera quick testers ----------
def try_open_camera(idx=CAM_INDEX, backend=None, timeout=1.0) -> bool:
    if backend is None:
        cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx, backend)
    try:
        ok = cap.isOpened()
        if not ok:
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        start = time.time()
        while time.time() - start < timeout:
            ret, _ = cap.read()
            if ret:
                return True
            time.sleep(0.02)
        return False
    finally:
        try: cap.release()
        except: pass

def ensure_camera_available(timeout: float = 12.0) -> bool:
    print(f"[CAM] ensure_camera_available(timeout={timeout})")
    end = time.time() + timeout
    while time.time() < end:
        if try_open_camera(CAM_INDEX):
            print("[CAM] camera opened successfully (default backend).")
            return True
        try:
            if try_open_camera(CAM_INDEX, backend=cv2.CAP_V4L2):
                print("[CAM] camera opened successfully (V4L2 backend).")
                return True
        except Exception:
            pass
        time.sleep(0.6)
    print("[CAM] ensure_camera_available timed out.")
    return False

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
        # handle b == 0 gracefully
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
        if FACE_FLAG == 0:
            print("[AUTH] start camera for face matching")
            cap = cv2.VideoCapture(CAM_INDEX)
            if not cap.isOpened():
                print("[AUTH] camera open failed for auth. aborting auth.")
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
                start_auth = time.time()
                while not sent_face_ok:
                    ret, frame = cap.read()
                    if not ret:
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
                cap.release()
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
        # ensure camera
        if not ensure_camera_available(timeout=12.0):
            print("[MON] camera not available -> exiting monitoring")
            return

        # pose model (create here; can be lower complexity)
        pose = mp.solutions.pose.Pose(model_complexity=0, static_image_mode=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print("[MON] camera failed to open for monitoring")
            pose.close()
            return

        ret, frame = cap.read()
        if not ret:
            print("[MON] cannot read first frame -> abort")
            cap.release()
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
                if not ret:
                    time.sleep(0.02)
                    continue
                img = cv2.flip(frame,1)
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                framec += 1
                now = time.time()
                if now - prev >= 1.0:
                    fps = framec; framec = 0; prev = now

                # heartbeat every 2 sec
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

                # DROWSY alerts
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
                        except Exception as e:
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
                    # headless small sleep
                    time.sleep(0.002)
        finally:
            cap.release()
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
