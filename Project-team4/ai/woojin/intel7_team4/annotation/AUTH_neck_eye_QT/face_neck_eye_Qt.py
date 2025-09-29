if 1:
    #!/usr/bin/env python3
    # neck_eye_final_auth_before_mjpg.py
    # 변경: FACE_FLAG==0일 때 먼저 로컬 인증 수행(카메라 직접 사용), 인증 끝난 뒤 mjpg_streamer 시작 및 모니터링
    # 수정: ATTENDANCE:OK가 올 때까지 무한 블로킹(단, SIGINT/SIGTERM로 중단 가능)

    import os
    import sys
    import time
    import math
    import socket
    import argparse
    import threading
    import subprocess
    import signal
    import traceback
    import re
    import fcntl
    import cv2
    import numpy as np
    import mediapipe as mp
    from flask import Flask, Response, render_template_string, jsonify, request

    # ---------------- CONFIG ----------------
    HOST = os.environ.get("CTRL_HOST", "192.168.0.158")
    PORT = int(os.environ.get("CTRL_PORT", "5000"))
    SOCK_TIMEOUT = 5.0
    LOGIN_MSG = os.environ.get("LOGIN_MSG", "AI:PASSWD")
    SRC = os.environ.get("SRC", "AI")
    USER = "seol"

    # messages
    state_on = "ON"; state_off = "OFF"
    SERVER_MSG_ON  = f"{SRC}:{USER}:SLP:{state_on}"
    SERVER_MSG_OFF = f"{SRC}:{USER}:SLP:{state_off}"
    POSTURE_MSG_BAD = f"{SRC}:{USER}:POSTURE:BAD:neck"

    # camera / mjpg-streamer
    CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
    DEV_NODE = f"/dev/video{CAM_INDEX}"
    MJPG_HTTP_PORT = int(os.environ.get("MJPG_PORT", "8080"))
    MJPG_URL = f"http://127.0.0.1:{MJPG_HTTP_PORT}/?action=stream"
    MJPG_STREAMER_BIN = os.environ.get("MJPG_BIN", "/usr/local/bin/mjpg_streamer")
    MJPG_INPUT_PLUGIN = os.environ.get("MJPG_INPUT", "input_uvc.so")
    MJPG_OUTPUT_PLUGIN = os.environ.get("MJPG_OUTPUT", "output_http.so")

    # thresholds
    EAR_THRESHOLD = float(os.environ.get("EAR_THRESHOLD", "0.20"))
    DESIRED_CLOSED_SECS = float(os.environ.get("DESIRED_CLOSED_SECS", "0.5"))
    DESIRED_ALARM_SECS = float(os.environ.get("DESIRED_ALARM_SECS", "5.0"))
    ANGLE_THRESHOLD = float(os.environ.get("ANGLE_THRESHOLD", "15.0"))
    VIS_THRESH_POSE = float(os.environ.get("VIS_THRESH_POSE", "0.35"))
    POSE_PERIOD = float(os.environ.get("POSE_PERIOD", "10.0"))
    POSE_WINDOW = float(os.environ.get("POSE_WINDOW", "3.0"))
    POSE_SAMPLE_INTERVAL = float(os.environ.get("POSE_SAMPLE_INTERVAL", "0.5"))
    FACE_OUT_TIMEOUT = float(os.environ.get("FACE_OUT_TIMEOUT", "5.0"))

    LOCKFILE = "/tmp/neck_eye_processed.lock"

    # face/value regex from server
    _value_re = re.compile(r"^SERVER(?::[^:]+)*:VALUE:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$")
    _flag_re  = re.compile(r"^SERVER(?::[^:]+)*:FLAG:(?P<face>[01])$")
    _att_ok_re = re.compile(r"(?:SERVER:)?(?:[^:]+:)?ATTENDANCE:OK")

    # ---------------- mediapipe indices ----------------
    IDX_LEFT_EYE_OUT  = 133
    IDX_RIGHT_EYE_OUT = 362
    IDX_NOSE_TIP      = 1
    IDX_MOUTH_CENTER  = 13
    IDX_FOREHEAD      = 10
    IDX_CHIN          = 152
    LEFT_EAR_IDX = [33,160,158,133,153,144]
    RIGHT_EAR_IDX = [263,387,385,362,380,373]

    # ---------------- shared state ----------------
    latest_jpeg = None
    frame_event = threading.Event()
    stop_event = threading.Event()
    status_lock = threading.Lock()

    current_face_state = "NONE"
    current_slp_state = "NONE"
    current_neck_angle = None
    current_eye_status = "UNKNOWN"
    current_fps = 0

    mjpg_proc = None
    proc_thread = None
    flask_thread = None

    # ---------------- utilities ----------------
    def acquire_single_instance_lock():
        try:
            fp = open(LOCKFILE, "w")
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fp.write(str(os.getpid())); fp.flush()
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
            print("[NET SEND]", msg)
            sock.sendall(msg.encode('utf-8'))
        except Exception as e:
            print("[NET] send error:", e)

    def wait_for_pattern(sock, regex, timeout=None):
        end = None if timeout is None else (time.time() + timeout)
        buf = b""
        sock.settimeout(1.0)
        while True:
            if stop_event.is_set():
                return None
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
                line, buf = buf.split(b"\n",1)
                sline = line.decode('utf-8',errors='ignore').strip()
                if regex.search(sline) or regex.match(sline):
                    return sline
            # partial
            try:
                s = buf.decode('utf-8',errors='ignore')
                if regex.search(s) or regex.match(s):
                    return s
            except:
                pass

    def is_port_listening(port):
        try:
            out = subprocess.check_output(["ss","-ltnp"]).decode("utf-8","ignore")
            return f":{port} " in out or f":{port}\n" in out
        except Exception:
            return False

    def start_mjpg_streamer_if_needed(dev_node=DEV_NODE, port=MJPG_HTTP_PORT, width=640, height=480, fps=15):
        global mjpg_proc
        if is_port_listening(port):
            print(f"[MJPG] port {port} already listening -> assume already running")
            return None
        cmd = f'{MJPG_STREAMER_BIN} -i "{MJPG_INPUT_PLUGIN} -d {dev_node} -r {width}x{height} -f {fps}" -o "{MJPG_OUTPUT_PLUGIN} -p {port} -w ./www"'
        print("[MJPG] starting mjpg-streamer:", cmd)
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setpgrp)
            for _ in range(25):
                time.sleep(0.2)
                if is_port_listening(port):
                    mjpg_proc = p
                    print("[MJPG] started and listening on port", port)
                    return p
            stderr = p.stderr.read().decode("utf-8","ignore") if p.stderr else ""
            print("[MJPG] failed to start (no listen). stderr:", stderr[:300])
            mjpg_proc = p
            return p
        except Exception as e:
            print("[MJPG] start exception:", e)
            return None

    def kill_process_group(p):
        import os, signal
        if not p: return
        try:
            pgid = os.getpgid(p.pid)
            print(f"[MJPG] killing process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(10):
                if p.poll() is not None:
                    break
                time.sleep(0.2)
            if p.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        except Exception as e:
            print("[MJPG] kill exception:", e)

    # ---------------- simple measure helpers ----------------
    def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
    def measure(lm, w, h):
        def pt(i): return (lm[i].x * w, lm[i].y * h)
        try:
            fore = dist(pt(IDX_FOREHEAD), pt(IDX_CHIN))
            nose = dist(pt(IDX_NOSE_TIP), pt(IDX_MOUTH_CENTER))
            eye  = dist(pt(IDX_LEFT_EYE_OUT), pt(IDX_RIGHT_EYE_OUT))
        except Exception:
            return None
        if fore < 1e-6: return None
        return {"forehead_chin": fore, "nose_lips": nose, "eye_eye": eye,
                "eye_over_fc": eye/(fore+1e-9), "nose_over_fc": nose/(fore+1e-9)}

    def compute_ear(lm, idxs, h, w):
        try:
            pts = [(lm[i].x*w, lm[i].y*h) for i in idxs]
        except Exception:
            return None
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
        return (A+B)/(2.0*C)

    # ---------------- auth routine (direct camera) ----------------
    def run_auth_local(face_mesh, want_landmarks=True, timeout=30.0):
        """
        Open local V4L2 camera and attempt to match face using server-provided VALUE (FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE).
        Returns True if matched (FACE:OK should be sent by caller), False otherwise.
        """
        print("[AUTH] starting local auth (direct camera)")
        try:
            cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        except Exception:
            cap = None
        if not cap or not cap.isOpened():
            print("[AUTH] local camera open failed")
            try:
                if cap: cap.release()
            except: pass
            return False

        MATCH_REQUIRED = 3
        match_count = 0
        sent_ok = False
        headless = False
        win = "Auth (ESC cancel)"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except Exception:
            headless = True; win = None

        start = time.time()
        global FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE
        while time.time() - start < timeout and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02); continue
            img = cv2.flip(frame,1)
            h,w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                res = face_mesh.process(rgb)
            except Exception as e:
                res = None
                print("[AUTH] face_mesh exception:", e)
            if res and getattr(res, "multi_face_landmarks", None):
                lm = res.multi_face_landmarks[0].landmark
                if want_landmarks and not headless:
                    try:
                        for i in (IDX_FOREHEAD, IDX_CHIN, IDX_NOSE_TIP, IDX_MOUTH_CENTER, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT):
                            cx, cy = int(lm[i].x*w), int(lm[i].y*h)
                            cv2.circle(img, (cx,cy), 3, (0,255,255), -1)
                    except: pass
                obs = measure(lm, w, h)
                if obs:
                    matched = False
                    try:
                        if FOREHEAD_CHIN > 0 and NOSE_LIPS > 0 and EYE_EYE > 0:
                            sv_fc = float(FOREHEAD_CHIN); sv_nose=float(NOSE_LIPS); sv_eye=float(EYE_EYE)
                            ov_eye_ratio = obs["eye_eye"]/obs["forehead_chin"]
                            sv_eye_ratio = sv_eye/sv_fc
                            tol = 0.06
                            if abs(ov_eye_ratio - sv_eye_ratio) <= tol*abs(sv_eye_ratio):
                                matched = True
                        else:
                            if obs["forehead_chin"] > 60: matched = True
                    except Exception:
                        matched = False

                    if matched:
                        match_count += 1
                    else:
                        if match_count > 0:
                            match_count = 0
                else:
                    match_count = 0
            else:
                match_count = 0
                if not headless:
                    try:
                        cv2.putText(img, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)
                    except: pass

            if not headless:
                cv2.imshow(win, img)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[AUTH] cancelled by user")
                    break

            if match_count >= MATCH_REQUIRED:
                sent_ok = True
                break

        try:
            cap.release()
        except: pass
        try:
            if not headless and win is not None:
                cv2.destroyWindow(win)
        except: pass

        return sent_ok

    # ---------------- processing loop ----------------
    def processing_loop(sock, mjpg_url, prefer_mjpg=True, show_monitor_landmarks=False):
        global latest_jpeg, frame_event, stop_event
        global current_face_state, current_slp_state, current_neck_angle, current_eye_status, current_fps

        mp_face = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                    refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(model_complexity=0, static_image_mode=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # open capture (prefer MJPEG HTTP if available)
        cap = None
        if prefer_mjpg:
            try:
                cap = cv2.VideoCapture(mjpg_url)
                start = time.time()
                while time.time() - start < 3.0 and not stop_event.is_set():
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret: break
                    time.sleep(0.05)
                if not (cap and cap.isOpened()):
                    try: cap.release()
                    except: pass
                    cap = None
            except Exception as e:
                print("[PROC] mjpg open exc:", e); cap = None

        if cap is None:
            cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
            start = time.time()
            while time.time() - start < 3.0 and not stop_event.is_set():
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret: break
                time.sleep(0.05)
            if not (cap and cap.isOpened()):
                print("[PROC] cannot open capture"); return

        print("[PROC] started processing (headless)")

        fps = 15; prev = time.time(); framec = 0
        alpha = 0.35; ema = None
        closed_cnt = 0; closed_total = 0; alarm_sent = False
        period_anchor = time.time(); last_sample_t = 0.0
        neck_samples = []; last_window_max = None; posture_sent = False
        face_present = False; last_face_seen = 0.0

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02); continue

                framec += 1
                now = time.time()
                if now - prev >= 1.0:
                    fps = framec; framec = 0; prev = now
                    with status_lock:
                        current_fps = fps

                img = cv2.flip(frame, 1)
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                status = "NO FACE"; scol=(150,150,150)
                raw = None

                try:
                    res = face_mesh.process(rgb)
                except Exception as e:
                    res = None
                    print("[PROC] face_mesh exc:", e)

                if res and getattr(res, "multi_face_landmarks", None):
                    last_face_seen = now
                    if not face_present:
                        face_present = True
                        with status_lock:
                            current_face_state = "IN"
                        try: send_only(sock, f"{SRC}:{USER}:FACE:IN")
                        except: pass

                    lm = res.multi_face_landmarks[0].landmark
                    le = compute_ear(lm, LEFT_EAR_IDX, h, w)
                    re = compute_ear(lm, RIGHT_EAR_IDX, h, w)
                    if le is not None and re is not None: raw = (le + re)/2.0
                    elif le is not None: raw = le
                    elif re is not None: raw = re

                    if show_monitor_landmarks:
                        try:
                            for i in (IDX_FOREHEAD, IDX_CHIN, IDX_NOSE_TIP, IDX_MOUTH_CENTER, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT):
                                cx, cy = int(lm[i].x * w), int(lm[i].y * h)
                                cv2.circle(img, (cx, cy), 2, (0,255,255), -1)
                        except: pass

                    if raw is not None:
                        ema = raw if ema is None else alpha*raw + (1-alpha)*ema
                        closed_thr = max(1, int(round(DESIRED_CLOSED_SECS * fps)))
                        alarm_thr  = max(1, int(round(DESIRED_ALARM_SECS * fps)))
                        if ema < EAR_THRESHOLD:
                            closed_cnt += 1; closed_total += 1
                        else:
                            closed_cnt = 0; closed_total = 0
                        if closed_total >= alarm_thr:
                            status = "DROWSY ALERT!"; scol=(0,0,255)
                        elif closed_cnt >= closed_thr:
                            status = "EYES CLOSED"; scol=(0,255,255)
                        else:
                            status = "EYES OPEN"; scol=(0,255,0)
                        with status_lock:
                            current_eye_status = status
                else:
                    if face_present and (now - last_face_seen) >= FACE_OUT_TIMEOUT:
                        face_present = False
                        with status_lock:
                            current_face_state = "OUT"
                        try: send_only(sock, f"{SRC}:{USER}:FACE:OUT")
                        except: pass
                    closed_cnt = 0; closed_total = 0; status="NO FACE"; scol=(180,180,180)
                    with status_lock:
                        current_eye_status = status

                # SLP message logic
                if status == "DROWSY ALERT!" and not alarm_sent:
                    try: send_only(sock, SERVER_MSG_ON)
                    except: pass
                    alarm_sent = True
                    with status_lock: current_slp_state = "ON"
                if alarm_sent and status != "DROWSY ALERT!":
                    try: send_only(sock, SERVER_MSG_OFF)
                    except: pass
                    alarm_sent = False
                    with status_lock: current_slp_state = "OFF"

                # posture sampling
                elapsed = now - period_anchor
                if elapsed >= POSE_PERIOD:
                    n = int(elapsed // POSE_PERIOD)
                    period_anchor += n * POSE_PERIOD
                    neck_samples = []
                    elapsed = now - period_anchor
                in_window = (0.0 <= elapsed < POSE_WINDOW)
                running_max = None
                if in_window and (now - last_sample_t >= POSE_SAMPLE_INTERVAL):
                    last_sample_t = now
                    try:
                        pr = pose.process(rgb)
                    except Exception:
                        pr = None
                    if pr and getattr(pr, "pose_landmarks", None):
                        pl = pr.pose_landmarks.landmark
                        cand = []
                        for idx in (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value):
                            if pl[idx].visibility > VIS_THRESH_POSE:
                                cand.append((pl[idx].x, pl[idx].y))
                        used = len(cand)
                        if used > 0:
                            sx = sum(p[0] for p in cand)/used
                            sy = sum(p[1] for p in cand)/used
                            ls = pl[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            rs = pl[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            if ls.visibility>VIS_THRESH_POSE and rs.visibility>VIS_THRESH_POSE:
                                nx = (ls.x+rs.x)/2.0; ny=(ls.y+rs.y)/2.0
                                vx = sx-nx; vy = sy-ny
                                if math.hypot(vx,vy) > 1e-6:
                                    ang = abs(math.degrees(math.atan2(vx,-vy)))
                                    neck_samples.append(ang)
                                    running_max = max(neck_samples)
                                    with status_lock:
                                        current_neck_angle = running_max
                                    try:
                                        cv2.line(img, (int(nx*w),int(ny*h)), (int(sx*w),int(sy*h)), (255,0,255), 3)
                                    except: pass
                else:
                    if neck_samples:
                        last_window_max = max(neck_samples)
                        neck_samples = []
                        with status_lock:
                            current_neck_angle = last_window_max
                        if last_window_max > ANGLE_THRESHOLD and not posture_sent:
                            try: send_only(sock, POSTURE_MSG_BAD)
                            except: pass
                            posture_sent = True
                        if last_window_max <= ANGLE_THRESHOLD and posture_sent:
                            posture_sent = False

                # overlay
                try:
                    cv2.putText(img, f"Eye:{status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scol, 2)
                    with status_lock:
                        na = current_neck_angle
                        fs = current_face_state
                        ss = current_slp_state
                        es = current_eye_status
                        fps_loc = current_fps
                    neck_text = f"Neck:-" if na is None else f"Neck:{na:.1f}deg"
                    cv2.putText(img, neck_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                    cv2.putText(img, f"FACE:{fs}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                    cv2.putText(img, f"SLP:{ss}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                    cv2.putText(img, f"FPS:{fps_loc}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)
                except Exception:
                    pass

                # encode JPEG
                try:
                    ret2, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret2:
                        latest_jpeg = jpg.tobytes()
                        frame_event.set(); frame_event.clear()
                except Exception as e:
                    print("[PROC] jpeg encode err:", e)

        except Exception as e:
            print("[PROC] unhandled exc:", e)
            traceback.print_exc()
        finally:
            try: cap.release()
            except: pass
            try: face_mesh.close()
            except: pass
            try: pose.close()
            except: pass
            print("[PROC] exiting")

    # ---------------- Flask app ----------------
    app = Flask(__name__)
    PROC_ROUTE = "/processed"
    INDEX_HTML = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"/><title>Processed Stream & Status</title>
    <style>body{background:#111;color:#eee;font-family:Arial}.wrap{max-width:900px;margin:10px auto}.stream{border:1px solid #444}.status{margin-top:10px;background:#222;padding:10px;border-radius:6px}.k{color:#9cf}</style>
    </head>
    <body>
    <div class="wrap">
        <h2>Processed Stream</h2>
        <div><img class="stream" id="stream" src="{{route}}" width="800"></div>
        <div class="status">
        <div>FACE: <span id="face" class="k">-</span></div>
        <div>SLP: <span id="slp" class="k">-</span></div>
        <div>Neck: <span id="neck" class="k">-</span></div>
        <div>EyeStatus: <span id="eye" class="k">-</span></div>
        <div>FPS: <span id="fps" class="k">-</span></div>
        </div>
    </div>
    <script>
    async function fetchStatus(){
    try{
        let r = await fetch('/status');
        if(!r.ok) return;
        let j = await r.json();
        document.getElementById('face').textContent = j.face;
        document.getElementById('slp').textContent = j.slp;
        document.getElementById('neck').textContent = j.neck === null ? '-' : j.neck.toFixed(1) + ' deg';
        document.getElementById('eye').textContent = j.eye;
        document.getElementById('fps').textContent = j.fps;
    }catch(e){ console.log('status fetch err', e); }
    }
    setInterval(fetchStatus, 1000); fetchStatus();
    </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return render_template_string(INDEX_HTML, route=PROC_ROUTE)

    @app.route(PROC_ROUTE)
    def proc_stream():
        def generator():
            global latest_jpeg, stop_event, frame_event
            while not stop_event.is_set():
                if not frame_event.wait(timeout=1.0):
                    continue
                data = latest_jpeg
                if not data: continue
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
        return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=--frame')

    @app.route('/status')
    def status():
        with status_lock:
            return jsonify({
                "face": current_face_state,
                "slp": current_slp_state,
                "neck": (None if current_neck_angle is None else float(current_neck_angle)),
                "eye": current_eye_status,
                "fps": current_fps
            })

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return "Not running with the Werkzeug Server", 500
        func()
        return 'Server shutting down...', 200

    # ---------------- main ----------------
    def connect_ctrl_server():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(SOCK_TIMEOUT)
            s.connect((HOST, PORT))
            print("[NET] connected to", HOST, PORT)
            return s
        except Exception as e:
            print("[NET] connect failed:", e); return None

    def signal_handler(sig, frame):
        print(f"[SIGNAL] {sig} received -> stopping")
        stop_event.set()
        # try to shutdown flask
        try:
            import urllib.request
            urllib.request.urlopen('http://127.0.0.1:8081/shutdown', data=b'', timeout=1.0)
        except:
            pass

    def main():
        global mjpg_proc, proc_thread, flask_thread
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-mjpg", action="store_true", help="do not attempt to start mjpg-streamer")
        parser.add_argument("--mjpg-port", type=int, default=MJPG_HTTP_PORT)
        parser.add_argument("--proc-port", type=int, default=8081)
        parser.add_argument("--show-monitor-landmarks", action="store_true")
        args = parser.parse_args()

        lock_fp = acquire_single_instance_lock()

        sock = connect_ctrl_server()
        face_flag = 0
        face_mesh = None
        if sock:
            try:
                send_only(sock, LOGIN_MSG)
                try:
                    sock.settimeout(1.0)
                    data = sock.recv(4096)
                    if data: print("[NET ACK]", data.decode(errors='ignore').strip())
                except socket.timeout:
                    pass
                finally:
                    sock.settimeout(None)
            except Exception as e:
                print("[NET] init err", e)

            # request FLAG
            try:
                send_only(sock, f"{SRC}:{USER}:AUTH:FLAG")
                line = wait_for_pattern(sock, _flag_re, timeout=5.0)
                if line:
                    m = _flag_re.match(line)
                    if m:
                        face_flag = int(m.group("face"))
                        print("[Main] initial FACE_FLAG =", face_flag)
                    else:
                        face_flag = 0
                else:
                    print("[Main] FLAG not received (timeout) -> assume 0")
                    face_flag = 0
            except Exception as e:
                print("[Main] FLAG request error:", e)
                face_flag = 0
            
            # request VALUE
            try:
                send_only(sock, f"{SRC}:{USER}:AUTH:VALUE")
                line = wait_for_pattern(sock, _value_re, timeout=10.0)
                if line:
                    m = _value_re.match(line)
                    if m:
                        global FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE
                        FOREHEAD_CHIN = float(m.group("fore"))
                        NOSE_LIPS = float(m.group("nose"))
                        EYE_EYE = float(m.group("eye"))
                        global EAR_THRESHOLD
                        EAR_THRESHOLD = float(m.group("ear"))
                        print("[VALUE SET]", FOREHEAD_CHIN, NOSE_LIPS, EYE_EYE, EAR_THRESHOLD)
                else:
                    print("[Main] VALUE not received (timeout). Using defaults.")
            except Exception as e:
                print("[Main] VALUE request error:", e)
            
            # initialize FaceMesh for auth usage
            mp_face = mp.solutions.face_mesh
            face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                        refine_landmarks=True, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

            # If face_flag == 0 => perform local auth BEFORE starting mjpg
            if face_flag == 0:
                print("[Main] FACE_FLAG == 0 -> perform local auth BEFORE starting mjpg")
                auth_ok = False
                try:
                    auth_ok = run_auth_local(face_mesh, want_landmarks=True, timeout=30.0)
                    if auth_ok:
                        try:
                            send_only(sock, f"{SRC}:{USER}:FACE:OK")
                            print("[AUTH] sent FACE:OK")
                        except: pass
                    else:
                        print("[AUTH] not successful or cancelled -> proceeding to start mjpg & monitoring anyway")
                except Exception as e:
                    print("[AUTH] exception during local auth:", e)
            else:
                print("[Main] FACE_FLAG == 1 -> skip local auth")
            

            # -------------- wait for ATTENDANCE:OK -> BLOCKING --------------
            # Per request, block here until server sends ATTENDANCE:OK (or until stop_event set by SIGINT).
            print("[Main] waiting for ATTENDANCE:OK (blocking until received). Use Ctrl+C to cancel.")
            try:
                while not stop_event.is_set():
                    line = wait_for_pattern(sock, _att_ok_re, timeout=5.0)
                    if line:
                        print("[Main] ATTENDANCE received:", line)
                        break
                    else:
                        print("[Main] still waiting for ATTENDANCE:OK...")
                        send_only(sock, f"{SRC}:{USER}:AUTH:VALUE")
                if stop_event.is_set():
                    print("[Main] stop_event set while waiting for ATTENDANCE -> proceeding to shutdown")
                    # graceful exit
                    release_single_instance_lock(lock_fp)
                    return
            except Exception as e:
                print("[Main] error while waiting for ATTENDANCE:", e)
                # proceed anyway if unexpected error
            # ----------------------------------------------------------------

        else:
            face_flag = 0
            face_mesh = None
            print("[Main] No control socket; proceeding without server-side flags (will run auth locally then start mjpg)")

        # now start mjpg-streamer (only AFTER auth attempt if face_flag was 0)
        if not args.no_mjpg:
            start_mjpg_streamer_if_needed(dev_node=DEV_NODE, port=args.mjpg_port, width=640, height=480, fps=15)
        else:
            print("[MJPG] --no-mjpg set, not starting mjpg-streamer")

        # start processing loop in thread (read mjpg stream preferably)
        proc_thread = threading.Thread(target=processing_loop, args=(sock, f'http://127.0.0.1:{args.mjpg_port}/?action=stream', True, args.show_monitor_landmarks), daemon=True)
        proc_thread.start()

        # start Flask app in separate thread
        def run_flask():
            try:
                app.run(host='0.0.0.0', port=args.proc_port, threaded=True, use_reloader=False)
            except Exception as e:
                print("[FLASK] run exception:", e)
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

        print(f"[HTTP] processed MJPEG available at http://0.0.0.0:{args.proc_port}/")
        try:
            while not stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[MAIN] KeyboardInterrupt -> stopping")
            stop_event.set()

        # cleanup
        print("[MAIN] stopping threads and cleaning up...")
        # shutdown flask
        try:
            import urllib.request
            urllib.request.urlopen(f'http://127.0.0.1:{args.proc_port}/shutdown', data=b'', timeout=1.0)
        except:
            pass
        if proc_thread:
            proc_thread.join(timeout=3.0)
        if flask_thread:
            flask_thread.join(timeout=3.0)

        # terminate mjpg we started
        try:
            if mjpg_proc and mjpg_proc.poll() is None:
                kill_process_group(mjpg_proc)
        except Exception as e:
            print("[MAIN] mjpg kill exc:", e)

        try:
            if sock:
                sock.close()
        except: pass

        release_single_instance_lock(lock_fp)
        print("[MAIN] exited cleanly")

    if __name__ == "__main__":
        signal.signal(signal.SIGINT, lambda s,f: (print("[SIGNAL] SIGINT"), stop_event.set()))
        signal.signal(signal.SIGTERM, lambda s,f: (print("[SIGNAL] SIGTERM"), stop_event.set()))
        main()

