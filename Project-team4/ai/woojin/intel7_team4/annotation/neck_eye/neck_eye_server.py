


if 0:
    # merged_drowsy_and_turtleneck_max_window_posture_alert_overlay_fixed.py
    # 화면 표시 정리: 1행(눈 상태 3개), 2행(목 각도), 3행(FPS) — 나머지 로직은 기존과 동일

    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import json
    import os
    import socket
    import sys
    import math

    # ====== SETTINGS (changeable) ======
    ANGLE_THRESHOLD = 15.0       # degree threshold for turtle neck
    VISIBILITY_THRESH = 0.35     # landmark visibility threshold
    CAM_INDEX = 0                # camera index

    # POSE scheduling: every POSE_PERIOD start a POSE_WINDOW of length POSE_WINDOW
    POSE_PERIOD = 10.0           # seconds
    POSE_WINDOW = 3.0            # seconds (window length)
    POSE_SAMPLE_INTERVAL = 0.5   # seconds between samples inside window
    # ===================================

    # -------------------------
    # drowsy settings
    CALIB_JSON = None
    USE_BOTH_EYES = True
    EAR_DEFAULT = 0.20
    REL_DROP = 0.75
    STD_K = 1.5
    ABS_MIN = 0.12
    EMA_ALPHA = 0.35

    # 판정 시간 (초)
    DESIRED_CLOSED_SECS = 0.50
    DESIRED_OPEN_SECS   = 0.10
    DESIRED_ALARM_SECS  = 5.00

    FPS_ASSUMED = 30

    # 서버 설정
    SEND_TO_SERVER = True
    SERVER_HOST = "192.168.0.158"
    SERVER_PORT = 5000
    SERVER_CONNECT_RETRY_SECS = 2.0
    SERVER_SOCKET_TIMEOUT = 3.0

    # 로그인/전송 문자열 (개행 없이 전송)
    LOGIN_MSG = "SEOL_SQL"
    src = "AI"
    user = "seol"
    type = "SLP"
    state_on = "ON"
    state_off = "OFF"
    SERVER_MSG_ON  = f"{src}:{user}:{type}:{state_on}"
    SERVER_MSG_OFF = f"{src}:{user}:{type}:{state_off}"
    # posture alert 메시지 (요구한 형식)
    POSTURE_MSG_BAD = "AI:SEOL_SQL:POSTURE:BAD:neck"
    # -------------------------

    # Face mesh eye indices (MediaPipe)
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    # -------------------------------------------------
    def compute_ear(landmarks, idx_list, h, w):
        try:
            pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx_list]
        except Exception:
            return None
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
        return (A + B) / (2.0 * C)

    def compute_threshold_from_json(path):
        try:
            j = json.load(open(path, 'r'))
        except Exception as e:
            print("Calib JSON load error:", e)
            return None
        stats = j.get('with_glasses') or j.get('without_glasses')
        if not stats:
            return None
        mean = stats.get('mean')
        std  = stats.get('std', 0.0) or 0.0
        if mean is None:
            return None
        t1 = mean * REL_DROP
        t2 = mean - STD_K * std
        thr = min(t1, t2) if std > 0 else t1
        return float(max(thr, ABS_MIN))

    def frames_from_secs(secs, fps_now, fps_fallback):
        f = fps_now if fps_now > 0 else fps_fallback
        return max(1, int(round(secs * f)))

    def connect_blocking(host, port, retry_secs=2.0, timeout=3.0):
        print(f"[Network] Trying to connect to {host}:{port} (retry every {retry_secs}s)...")
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((host, port))
                s.settimeout(timeout)
                print(f"[Network] Connected to {host}:{port}")
                return s
            except KeyboardInterrupt:
                print("[Network] Interrupted by user during connect.")
                raise
            except Exception as e:
                print(f"[Network] Connect failed: {e}. Retrying in {retry_secs}s...")
                try:
                    s.close()
                except:
                    pass
                time.sleep(retry_secs)

    # -------------------------------------------------
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, static_image_mode=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # -------------------------------------------------
    # State
    ema_ear = None
    consec_closed = 0
    consec_open = 0
    frame_count = 0
    prev_time = time.time()
    fps = FPS_ASSUMED
    frame_idx = 0

    closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
    open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS,   fps, FPS_ASSUMED)
    alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS,  fps, FPS_ASSUMED)

    total_consec_closed_for_alarm = 0
    alarm_sent = False

    # Pose variables
    neck_status_text = "No pose yet"
    neck_color = (128,128,128)
    last_window_max_angle = None
    last_window_used_sum = 0

    # Pose scheduling trackers
    pose_period_anchor = time.time()
    last_pose_sample_time = 0.0

    # collect samples during window (store angles and used counts)
    neck_samples = []   # list of (angle_deg, used_points)

    # posture alert flag (one-shot until recovery)
    posture_sent = False

    # -------------------------------------------------
    # Network connect + login
    sock = None
    if SEND_TO_SERVER:
        try:
            sock = connect_blocking(SERVER_HOST, SERVER_PORT, SERVER_CONNECT_RETRY_SECS, SERVER_SOCKET_TIMEOUT)
        except KeyboardInterrupt:
            sys.exit(0)

        try:
            sock.sendall(LOGIN_MSG.encode('utf-8'))   # no newline
            print(f"[Network] Sent login -> {LOGIN_MSG}")
        except Exception as e:
            print(f"[Network] Failed to send login: {e}")
            try: sock.close()
            except: pass
            sys.exit(1)

        try:
            sock.settimeout(SERVER_SOCKET_TIMEOUT)
            data = sock.recv(1024)
            if data:
                try:
                    s = data.decode('utf-8', errors='ignore').strip()
                except:
                    s = repr(data)
                print(f"[Network] Login response: {s}")
            else:
                print("[Network] Login response: <empty>")
        except socket.timeout:
            print("[Network] No login ACK (timeout). Proceeding.")
        except Exception as e:
            print(f"[Network] Error waiting login ACK: {e}")
            try: sock.close()
            except: pass
            sys.exit(1)

    # -------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Camera] camera open failed.")
        if sock:
            try: sock.close()
            except: pass
        sys.exit(1)

    win = "Drowsy + TurtleNeck (max-window + posture alert)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    # threshold load
    if CALIB_JSON and os.path.exists(CALIB_JSON):
        THRESHOLD = compute_threshold_from_json(CALIB_JSON) or EAR_DEFAULT
        print(f"[Config] Using calibrated EAR threshold = {THRESHOLD:.4f}")
    else:
        THRESHOLD = EAR_DEFAULT
        print(f"[Config] Using default EAR threshold = {THRESHOLD:.4f}")

    try:
        print("[Run] Starting detection loop. ESC to exit, 'r' to reset alarm/posture flags.")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            now = time.time()
            frame_idx += 1
            frame_count += 1

            # fps calc each 1s
            if now - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = now
                closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
                open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS,   fps, FPS_ASSUMED)
                alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS,  fps, FPS_ASSUMED)

            img = cv2.flip(frame, 1)
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # -------------------------
            # Face mesh + EAR (drowsy) - every frame
            # -------------------------
            raw_ear = None
            ear_for_display = None
            status_text = "NO FACE"
            status_color = (200,200,200)

            results = face_mesh.process(img_rgb)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                mesh = results.multi_face_landmarks[0]
                lm = mesh.landmark

                left_ear_val = compute_ear(lm, LEFT_EYE_IDX, h, w)
                right_ear_val = compute_ear(lm, RIGHT_EYE_IDX, h, w) if USE_BOTH_EYES else None

                if USE_BOTH_EYES:
                    if left_ear_val is not None and right_ear_val is not None:
                        raw_ear = (left_ear_val + right_ear_val) / 2.0
                    elif left_ear_val is not None:
                        raw_ear = left_ear_val
                    elif right_ear_val is not None:
                        raw_ear = right_ear_val
                else:
                    raw_ear = left_ear_val if left_ear_val is not None else right_ear_val

                if raw_ear is not None:
                    if ema_ear is None:
                        ema_ear = raw_ear
                    else:
                        ema_ear = EMA_ALPHA * raw_ear + (1.0 - EMA_ALPHA) * ema_ear

                    ear_for_display = ema_ear

                    if ema_ear < THRESHOLD:
                        consec_closed += 1
                        consec_open = 0
                        total_consec_closed_for_alarm += 1
                    else:
                        consec_open += 1
                        total_consec_closed_for_alarm = 0
                        if consec_open >= open_thr_frames:
                            consec_closed = 0

                    if total_consec_closed_for_alarm >= alarm_thr_frames:
                        status_text = "DROWSY ALERT!"
                        status_color = (0,0,255)
                    elif consec_closed >= closed_thr_frames:
                        status_text = "EYES CLOSED"
                        status_color = (0,255,255)
                    else:
                        status_text = "EYES OPEN"
                        status_color = (0,255,0)
                else:
                    status_text = "LANDMARK ERROR"
                    status_color = (100,100,100)

                # --- eye debug drawing: (주석처리됨) ---
                # for i in LEFT_EYE_IDX:
                #     x, y = int(lm[i].x * w), int(lm[i].y * h)
                #     cv2.circle(img, (x,y), 1, (0,255,255), -1)
                # for i in RIGHT_EYE_IDX:
                #     x, y = int(lm[i].x * w), int(lm[i].y * h)
                #     cv2.circle(img, (x,y), 1, (0,255,255), -1)
                # ----------------------------------------
            else:
                consec_open = 0
                consec_closed = 0
                total_consec_closed_for_alarm = 0

            # -------------------------
            # Pose (turtle-neck) - PERIODIC sampling -> compute per-sample angle from raw centroid
            # -------------------------
            # running_max will be used to show in-window running max; initialize None
            running_max = None

            elapsed_since_anchor = now - pose_period_anchor
            if elapsed_since_anchor >= POSE_PERIOD:
                n = int(elapsed_since_anchor // POSE_PERIOD)
                pose_period_anchor += n * POSE_PERIOD
                elapsed_since_anchor = now - pose_period_anchor
                # when new period starts, reset neck_samples so window starts fresh
                neck_samples = []

            in_window = (0.0 <= elapsed_since_anchor < POSE_WINDOW)

            if in_window:
                if now - last_pose_sample_time >= POSE_SAMPLE_INTERVAL:
                    last_pose_sample_time = now
                    # perform pose processing for this sample
                    pose_results = pose.process(img_rgb)
                    if pose_results.pose_landmarks:
                        lm_p = pose_results.pose_landmarks.landmark
                        candidates = []
                        n = lm_p[mp_pose.PoseLandmark.NOSE.value]
                        if n.visibility > VISIBILITY_THRESH:
                            candidates.append((n.x, n.y))
                        le = lm_p[mp_pose.PoseLandmark.LEFT_EAR.value]
                        if le.visibility > VISIBILITY_THRESH:
                            candidates.append((le.x, le.y))
                        re = lm_p[mp_pose.PoseLandmark.RIGHT_EAR.value]
                        if re.visibility > VISIBILITY_THRESH:
                            candidates.append((re.x, re.y))

                        used = len(candidates)
                        if used > 0:
                            sx = sum([p[0] for p in candidates]) / used
                            sy = sum([p[1] for p in candidates]) / used
                            # compute angle directly from this sample's centroid and shoulders midpoint (no EMA)
                            l_sh = lm_p[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            r_sh = lm_p[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            if l_sh.visibility > VISIBILITY_THRESH and r_sh.visibility > VISIBILITY_THRESH:
                                neck_x = (l_sh.x + r_sh.x) / 2.0
                                neck_y = (l_sh.y + r_sh.y) / 2.0
                                vx = sx - neck_x
                                vy = sy - neck_y
                                norm = math.hypot(vx, vy)
                                if norm > 1e-6:
                                    angle_rad = math.atan2(vx, -vy)
                                    angle_deg = abs(math.degrees(angle_rad))
                                    neck_samples.append((angle_deg, used))
                                    # update running max for display during window
                                    running_max = max(a for a,_ in neck_samples)
                                    total_used = sum(u for _,u in neck_samples)
                                    neck_status_text = f"Sampling... max={running_max:.1f}° samples={len(neck_samples)} used_sum={total_used}"
                                    neck_color = (0,255,0) if running_max <= ANGLE_THRESHOLD else (0,0,255)
                                    # draw viz using this sample centroid & neck point
                                    hx, hy = int(sx * w), int(sy * h)
                                    nx, ny = int(neck_x * w), int(neck_y * h)
                                    cv2.line(img, (nx, ny), (hx, hy), (255, 0, 255), 3)
                                    cv2.circle(img, (nx, ny), 6, (0, 255, 255), -1)
                                    cv2.circle(img, (hx, hy), 6, (255, 0, 0), -1)
                                else:
                                    neck_status_text = "Head-Neck too close"
                                    neck_color = (100,100,100)
                            else:
                                neck_status_text = "Shoulders not visible"
                                neck_color = (100,100,100)
                        else:
                            neck_status_text = "Head landmarks not visible"
                            neck_color = (128,128,128)
                    else:
                        neck_status_text = "No pose detected"
                        neck_color = (128,128,128)
            else:
                # not in window: if we have samples from the just-finished window, compute final max and keep it displayed
                if len(neck_samples) > 0:
                    max_angle = max(a for a,_ in neck_samples)
                    total_used = sum(u for _,u in neck_samples)
                    last_window_max_angle = max_angle
                    last_window_used_sum = total_used
                    neck_status_text = f"Window MAX: {max_angle:.1f}°  samples:{len(neck_samples)} used_sum:{total_used}"
                    neck_color = (0,255,0) if max_angle <= ANGLE_THRESHOLD else (0,0,255)
                    # posture alert send logic (send once until recovery)
                    if SEND_TO_SERVER and sock:
                        if max_angle > ANGLE_THRESHOLD and not posture_sent:
                            try:
                                sock.sendall(POSTURE_MSG_BAD.encode('utf-8'))  # no newline
                                posture_sent = True
                                print(f"[Notify] Sent POSTURE BAD -> {POSTURE_MSG_BAD}")
                            except Exception as e:
                                print(f"[Notify] POSTURE send failed: {e}")
                                try: sock.close()
                                except: pass
                                sock = None
                                print("[Run] Server connection lost — stopping detection loop.")
                                break
                        # reset flag when posture recovered (allow next violation to send again)
                        if max_angle <= ANGLE_THRESHOLD and posture_sent:
                            posture_sent = False
                    # clear samples so next window starts fresh when it arrives
                    neck_samples = []
                # otherwise keep previous neck_status_text (last window's value)

            # -------------------------
            # Server: DROWSY ON/OFF (unchanged)
            # -------------------------
            if SEND_TO_SERVER and sock:
                if status_text == "DROWSY ALERT!" and not alarm_sent:
                    try:
                        sock.sendall(SERVER_MSG_ON.encode('utf-8'))
                        alarm_sent = True
                        print(f"[Notify] Sent ON -> {SERVER_MSG_ON}")
                    except Exception as e:
                        print(f"[Notify] ON send failed: {e}")
                        try: sock.close()
                        except: pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break
                elif alarm_sent and status_text != "DROWSY ALERT!":
                    try:
                        sock.sendall(SERVER_MSG_OFF.encode('utf-8'))
                        alarm_sent = False
                        print(f"[Notify] Sent OFF -> {SERVER_MSG_OFF}")
                    except Exception as e:
                        print(f"[Notify] OFF send failed: {e}")
                        try: sock.close()
                        except: pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break

            # -------------------------
            # Overlay : (1) 한 줄에 눈 상태 3개 (현재 상태만 색상 강조)
            #           (2) 그 아래 한 줄에 Neck: xx.x°
            #           (3) 그 아래 한 줄에 FPS
            # -------------------------
            # Prepare eye-state labels (Korean)
            labels = [("Eye Closed", (0,255,255)), ("Eye Open", (0,255,0)), ("Drowsy", (0,0,255))]
            # decide active index
            if status_text == "EYES CLOSED":
                active_idx = 0
            elif status_text == "EYES OPEN":
                active_idx = 1
            elif status_text == "SLP!":
                active_idx = 2
            else:
                active_idx = None

            # draw the three labels in one line (left-aligned, spaced)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2
            x = 10
            y = 30
            gap = 30
            for i, (txt, clr) in enumerate(labels):
                color = clr if (active_idx == i) else (180,180,180)
                cv2.putText(img, txt, (x, y), font, scale, color, thickness)
                (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
                x += tw + gap

            # Neck: use running_max(if in_window and samples exist) else last_window_max_angle
            neck_angle_to_show = None
            if in_window and len(neck_samples) > 0:
                neck_angle_to_show = running_max
            elif last_window_max_angle is not None:
                neck_angle_to_show = last_window_max_angle

            neck_y = y + 28
            if neck_angle_to_show is None:
                neck_txt = "Neck: -"
                neck_clr = (180,180,180)
            else:
                neck_txt = f"Neck: {neck_angle_to_show:.1f}deg"
                neck_clr = (0,255,0) if neck_angle_to_show <= ANGLE_THRESHOLD else (0,0,255)
            cv2.putText(img, neck_txt, (10, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, neck_clr, 2)

            # FPS line (below neck)
            fps_y = neck_y + 28
            cv2.putText(img, f"FPS: {fps}", (10, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50), 2)

            # -------------------------
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[Run] User exit (ESC).")
                break
            elif key == ord('r'):
                alarm_sent = False
                posture_sent = False
                print("[Run] alarm_sent and posture_sent manually reset.")

    finally:
        print("[Cleanup] Releasing resources...")
        try:
            if sock:
                sock.close()
        except:
            pass
        try:
            face_mesh.close()
        except:
            pass
        try:
            pose.close()
        except:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("[Exit] Bye.")


if 0:
    # merged_drowsy_and_turtleneck_max_window_posture_alert_overlay_face-first.py
    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import json
    import os
    import socket
    import sys
    import math

    # ====== SETTINGS (changeable) ======
    ANGLE_THRESHOLD = 15.0       # degree threshold for turtle neck
    VISIBILITY_THRESH = 0.35     # landmark visibility threshold
    CAM_INDEX = 0                # camera index

    # POSE scheduling: every POSE_PERIOD start a POSE_WINDOW of length POSE_WINDOW
    POSE_PERIOD = 10.0           # seconds
    POSE_WINDOW = 3.0            # seconds (window length)
    POSE_SAMPLE_INTERVAL = 0.5   # seconds between samples inside window
    # ===================================

    # -------------------------
    # drowsy settings
    CALIB_JSON = None
    USE_BOTH_EYES = True
    EAR_DEFAULT = 0.20
    REL_DROP = 0.75
    STD_K = 1.5
    ABS_MIN = 0.12
    EMA_ALPHA = 0.35

    # 판정 시간 (초)
    DESIRED_CLOSED_SECS = 0.50
    DESIRED_OPEN_SECS   = 0.10
    DESIRED_ALARM_SECS  = 5.00

    FPS_ASSUMED = 30

    # 서버 설정
    SEND_TO_SERVER = True
    SERVER_HOST = "192.168.0.158"
    SERVER_PORT = 5000
    SERVER_CONNECT_RETRY_SECS = 2.0
    SERVER_SOCKET_TIMEOUT = 3.0

    # 로그인/전송 문자열 (개행 없이 전송)
    LOGIN_MSG = "SEOL_SQL"
    src = "AI"
    user = "seol"
    type = "SLP"
    state_on = "ON"
    state_off = "OFF"
    SERVER_MSG_ON  = f"{src}:{user}:{type}:{state_on}"
    SERVER_MSG_OFF = f"{src}:{user}:{type}:{state_off}"
    # posture alert 메시지 (요구한 형식)
    POSTURE_MSG_BAD = "AI:SEOL_SQL:POSTURE:BAD:neck"

    # New: NO FACE notification settings
    NO_FACE_SECS = 5.0  # if face not seen for this many seconds -> then check pose and possibly notify
    SERVER_MSG_NO_FACE = f"{src}:{user}:ABSENT"
    SERVER_MSG_FACE_BACK = f"{src}:{user}:PRESENT"
    # -------------------------

    # Face mesh eye indices (MediaPipe)
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    # -------------------------------------------------
    def compute_ear(landmarks, idx_list, h, w):
        try:
            pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx_list]
        except Exception:
            return None
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
        return (A + B) / (2.0 * C)

    def compute_threshold_from_json(path):
        try:
            j = json.load(open(path, 'r'))
        except Exception as e:
            print("Calib JSON load error:", e)
            return None
        stats = j.get('with_glasses') or j.get('without_glasses')
        if not stats:
            return None
        mean = stats.get('mean')
        std  = stats.get('std', 0.0) or 0.0
        if mean is None:
            return None
        t1 = mean * REL_DROP
        t2 = mean - STD_K * std
        thr = min(t1, t2) if std > 0 else t1
        return float(max(thr, ABS_MIN))

    def frames_from_secs(secs, fps_now, fps_fallback):
        f = fps_now if fps_now > 0 else fps_fallback
        return max(1, int(round(secs * f)))

    def connect_blocking(host, port, retry_secs=2.0, timeout=3.0):
        print(f"[Network] Trying to connect to {host}:{port} (retry every {retry_secs}s)...")
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((host, port))
                s.settimeout(timeout)
                print(f"[Network] Connected to {host}:{port}")
                return s
            except KeyboardInterrupt:
                print("[Network] Interrupted by user during connect.")
                raise
            except Exception as e:
                print(f"[Network] Connect failed: {e}. Retrying in {retry_secs}s...")
                try:
                    s.close()
                except:
                    pass
                time.sleep(retry_secs)

    # -------------------------------------------------
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, static_image_mode=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # -------------------------------------------------
    # State
    ema_ear = None
    consec_closed = 0
    consec_open = 0
    frame_count = 0
    prev_time = time.time()
    fps = FPS_ASSUMED
    frame_idx = 0

    closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
    open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS,   fps, FPS_ASSUMED)
    alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS,  fps, FPS_ASSUMED)

    total_consec_closed_for_alarm = 0
    alarm_sent = False

    # Pose variables
    neck_status_text = "No pose yet"
    neck_color = (128,128,128)
    last_window_max_angle = None
    last_window_used_sum = 0

    # Pose scheduling trackers
    pose_period_anchor = time.time()
    last_pose_sample_time = 0.0

    # collect samples during window (store angles and used counts)
    neck_samples = []   # list of (angle_deg, used_points)

    # posture alert flag (one-shot until recovery)
    posture_sent = False

    # face/presence trackers (face-first logic)
    last_face_seen_time = time.time()
    no_face_sent = False

    # -------------------------------------------------
    # Network connect + login
    sock = None
    if SEND_TO_SERVER:
        try:
            sock = connect_blocking(SERVER_HOST, SERVER_PORT, SERVER_CONNECT_RETRY_SECS, SERVER_SOCKET_TIMEOUT)
        except KeyboardInterrupt:
            sys.exit(0)

        try:
            sock.sendall(LOGIN_MSG.encode('utf-8'))   # no newline
            print(f"[Network] Sent login -> {LOGIN_MSG}")
        except Exception as e:
            print(f"[Network] Failed to send login: {e}")
            try: sock.close()
            except: pass
            sys.exit(1)

        try:
            sock.settimeout(SERVER_SOCKET_TIMEOUT)
            data = sock.recv(1024)
            if data:
                try:
                    s = data.decode('utf-8', errors='ignore').strip()
                except:
                    s = repr(data)
                print(f"[Network] Login response: {s}")
            else:
                print("[Network] Login response: <empty>")
        except socket.timeout:
            print("[Network] No login ACK (timeout). Proceeding.")
        except Exception as e:
            print(f"[Network] Error waiting login ACK: {e}")
            try: sock.close()
            except: pass
            sys.exit(1)

    # -------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Camera] camera open failed.")
        if sock:
            try: sock.close()
            except: pass
        sys.exit(1)

    win = "Drowsy + TurtleNeck (face-first presence check)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # threshold load
    if CALIB_JSON and os.path.exists(CALIB_JSON):
        THRESHOLD = compute_threshold_from_json(CALIB_JSON) or EAR_DEFAULT
        print(f"[Config] Using calibrated EAR threshold = {THRESHOLD:.4f}")
    else:
        THRESHOLD = EAR_DEFAULT
        print(f"[Config] Using default EAR threshold = {THRESHOLD:.4f}")

    try:
        print("[Run] Starting detection loop. ESC to exit, 'r' to reset flags.")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            now = time.time()
            frame_idx += 1
            frame_count += 1

            # fps calc each 1s
            if now - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = now
                closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
                open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS, fps, FPS_ASSUMED)
                alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS, fps, FPS_ASSUMED)

            img = cv2.flip(frame, 1)
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # -------------------------
            # Face mesh + EAR (drowsy) - every frame
            # -------------------------
            raw_ear = None
            ear_for_display = None
            status_text = "NO FACE"
            status_color = (200,200,200)

            face_results = face_mesh.process(img_rgb)

            face_seen = False

            if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
                face_seen = True
                last_face_seen_time = now
                if no_face_sent:
                    # someone returned after NO_FACE -> notify server (one-shot)
                    if SEND_TO_SERVER and sock:
                        try:
                            sock.sendall(SERVER_MSG_FACE_BACK.encode('utf-8'))
                            print(f"[Notify] Sent FACE BACK -> {SERVER_MSG_FACE_BACK}")
                        except Exception as e:
                            print(f"[Notify] FACE BACK send failed: {e}")
                            try: sock.close()
                            except: pass
                            sock = None
                    no_face_sent = False

                mesh = face_results.multi_face_landmarks[0]
                lm = mesh.landmark

                left_ear_val = compute_ear(lm, LEFT_EYE_IDX, h, w)
                right_ear_val = compute_ear(lm, RIGHT_EYE_IDX, h, w) if USE_BOTH_EYES else None

                if USE_BOTH_EYES:
                    if left_ear_val is not None and right_ear_val is not None:
                        raw_ear = (left_ear_val + right_ear_val) / 2.0
                    elif left_ear_val is not None:
                        raw_ear = left_ear_val
                    elif right_ear_val is not None:
                        raw_ear = right_ear_val
                else:
                    raw_ear = left_ear_val if left_ear_val is not None else right_ear_val

                if raw_ear is not None:
                    if ema_ear is None:
                        ema_ear = raw_ear
                    else:
                        ema_ear = EMA_ALPHA * raw_ear + (1.0 - EMA_ALPHA) * ema_ear

                    ear_for_display = ema_ear

                    if ema_ear < THRESHOLD:
                        consec_closed += 1
                        consec_open = 0
                        total_consec_closed_for_alarm += 1
                    else:
                        consec_open += 1
                        total_consec_closed_for_alarm = 0
                        if consec_open >= open_thr_frames:
                            consec_closed = 0

                    if total_consec_closed_for_alarm >= alarm_thr_frames:
                        status_text = "DROWSY ALERT!"
                        status_color = (0,0,255)
                    elif consec_closed >= closed_thr_frames:
                        status_text = "EYES CLOSED"
                        status_color = (0,255,255)
                    else:
                        status_text = "EYES OPEN"
                        status_color = (0,255,0)
                else:
                    status_text = "LANDMARK ERROR"
                    status_color = (100,100,100)
            else:
                # face not seen this frame: reset EAR counters (no face landmarks available)
                consec_open = 0
                consec_closed = 0
                total_consec_closed_for_alarm = 0

            # -------------------------
            # If face not seen for NO_FACE_SECS, check pose ONCE and notify only if pose also not seen
            # -------------------------
            if (now - last_face_seen_time) >= NO_FACE_SECS and not no_face_sent:
                # call pose.process only at this check moment to verify presence
                pose_check = pose.process(img_rgb)
                pose_seen = False
                if pose_check.pose_landmarks:
                    lm_pq = pose_check.pose_landmarks.landmark
                    nose_vis = lm_pq[mp_pose.PoseLandmark.NOSE.value].visibility
                    lsh_vis  = lm_pq[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                    rsh_vis  = lm_pq[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                    lear_vis = lm_pq[mp_pose.PoseLandmark.LEFT_EAR.value].visibility
                    rear_vis = lm_pq[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility
                    if (nose_vis > VISIBILITY_THRESH) or (lsh_vis > VISIBILITY_THRESH) or (rsh_vis > VISIBILITY_THRESH) or (lear_vis > VISIBILITY_THRESH) or (rear_vis > VISIBILITY_THRESH):
                        pose_seen = True

                if pose_seen:
                    # Pose visible -> treat as present; reset face timer so we won't notify
                    last_face_seen_time = now
                    no_face_sent = False
                    neck_status_text = "Pose-only presence"
                else:
                    # Not seen by pose either -> send NO_FACE (one-shot)
                    if SEND_TO_SERVER and sock:
                        try:
                            sock.sendall(SERVER_MSG_NO_FACE.encode('utf-8'))
                            print(f"[Notify] Sent NO FACE -> {SERVER_MSG_NO_FACE}")
                        except Exception as e:
                            print(f"[Notify] NO FACE send failed: {e}")
                            try: sock.close()
                            except: pass
                            sock = None
                    no_face_sent = True
                    status_text = "NO PERSON"
                    status_color = (0,165,255)

            # -------------------------
            # Pose (turtle-neck) - PERIODIC sampling -> compute per-sample angle from raw centroid
            # (We still perform windowed pose sampling as before; pose.process is called inside window when sampling)
            # -------------------------
            running_max = None

            elapsed_since_anchor = now - pose_period_anchor
            if elapsed_since_anchor >= POSE_PERIOD:
                n = int(elapsed_since_anchor // POSE_PERIOD)
                pose_period_anchor += n * POSE_PERIOD
                elapsed_since_anchor = now - pose_period_anchor
                neck_samples = []

            in_window = (0.0 <= elapsed_since_anchor < POSE_WINDOW)

            if in_window:
                if now - last_pose_sample_time >= POSE_SAMPLE_INTERVAL:
                    last_pose_sample_time = now
                    # call pose.process for sampling (this is expected)
                    pose_results = pose.process(img_rgb)
                    if pose_results.pose_landmarks:
                        lm_p = pose_results.pose_landmarks.landmark
                        candidates = []
                        n = lm_p[mp_pose.PoseLandmark.NOSE.value]
                        if n.visibility > VISIBILITY_THRESH:
                            candidates.append((n.x, n.y))
                        le = lm_p[mp_pose.PoseLandmark.LEFT_EAR.value]
                        if le.visibility > VISIBILITY_THRESH:
                            candidates.append((le.x, le.y))
                        re = lm_p[mp_pose.PoseLandmark.RIGHT_EAR.value]
                        if re.visibility > VISIBILITY_THRESH:
                            candidates.append((re.x, re.y))

                        used = len(candidates)
                        if used > 0:
                            sx = sum([p[0] for p in candidates]) / used
                            sy = sum([p[1] for p in candidates]) / used
                            l_sh = lm_p[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            r_sh = lm_p[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            if l_sh.visibility > VISIBILITY_THRESH and r_sh.visibility > VISIBILITY_THRESH:
                                neck_x = (l_sh.x + r_sh.x) / 2.0
                                neck_y = (l_sh.y + r_sh.y) / 2.0
                                vx = sx - neck_x
                                vy = sy - neck_y
                                norm = math.hypot(vx, vy)
                                if norm > 1e-6:
                                    angle_rad = math.atan2(vx, -vy)
                                    angle_deg = abs(math.degrees(angle_rad))
                                    neck_samples.append((angle_deg, used))
                                    running_max = max(a for a,_ in neck_samples)
                                    total_used = sum(u for _,u in neck_samples)
                                    neck_status_text = f"Sampling... max={running_max:.1f}° samples={len(neck_samples)} used_sum={total_used}"
                                    neck_color = (0,255,0) if running_max <= ANGLE_THRESHOLD else (0,0,255)
                                    hx, hy = int(sx * w), int(sy * h)
                                    nx, ny = int(neck_x * w), int(neck_y * h)
                                    cv2.line(img, (nx, ny), (hx, hy), (255, 0, 255), 3)
                                    cv2.circle(img, (nx, ny), 6, (0, 255, 255), -1)
                                    cv2.circle(img, (hx, hy), 6, (255, 0, 0), -1)
                                else:
                                    neck_status_text = "Head-Neck too close"
                                    neck_color = (100,100,100)
                            else:
                                neck_status_text = "Shoulders not visible"
                                neck_color = (100,100,100)
                        else:
                            neck_status_text = "Head landmarks not visible"
                            neck_color = (128,128,128)
                    else:
                        neck_status_text = "No pose detected"
                        neck_color = (128,128,128)
            else:
                if len(neck_samples) > 0:
                    max_angle = max(a for a,_ in neck_samples)
                    total_used = sum(u for _,u in neck_samples)
                    last_window_max_angle = max_angle
                    last_window_used_sum = total_used
                    neck_status_text = f"Window MAX: {max_angle:.1f}°  samples:{len(neck_samples)} used_sum:{total_used}"
                    neck_color = (0,255,0) if max_angle <= ANGLE_THRESHOLD else (0,0,255)
                    if SEND_TO_SERVER and sock:
                        if max_angle > ANGLE_THRESHOLD and not posture_sent:
                            try:
                                sock.sendall(POSTURE_MSG_BAD.encode('utf-8'))  # no newline
                                posture_sent = True
                                print(f"[Notify] Sent POSTURE BAD -> {POSTURE_MSG_BAD}")
                            except Exception as e:
                                print(f"[Notify] POSTURE send failed: {e}")
                                try: sock.close()
                                except: pass
                                sock = None
                                print("[Run] Server connection lost — stopping detection loop.")
                                break
                        if max_angle <= ANGLE_THRESHOLD and posture_sent:
                            posture_sent = False
                    neck_samples = []

            # -------------------------
            # Server: DROWSY ON/OFF (unchanged)
            # -------------------------
            if SEND_TO_SERVER and sock:
                if status_text == "DROWSY ALERT!" and not alarm_sent:
                    try:
                        sock.sendall(SERVER_MSG_ON.encode('utf-8'))
                        alarm_sent = True
                        print(f"[Notify] Sent ON -> {SERVER_MSG_ON}")
                    except Exception as e:
                        print(f"[Notify] ON send failed: {e}")
                        try: sock.close()
                        except: pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break
                elif alarm_sent and status_text != "DROWSY ALERT!":
                    try:
                        sock.sendall(SERVER_MSG_OFF.encode('utf-8'))
                        alarm_sent = False
                        print(f"[Notify] Sent OFF -> {SERVER_MSG_OFF}")
                    except Exception as e:
                        print(f"[Notify] OFF send failed: {e}")
                        try: sock.close()
                        except: pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break

            # -------------------------
            # Overlay : one line Eye labels, one line Neck, one line FPS
            labels = [("Eye Closed", (0,255,255)), ("Eye Open", (0,255,0)), ("Drowsy", (0,0,255))]
            if status_text == "EYES CLOSED":
                active_idx = 0
            elif status_text == "EYES OPEN":
                active_idx = 1
            elif status_text == "SLP!":
                active_idx = 2
            else:
                active_idx = None

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2
            x = 10
            y = 30
            gap = 30
            for i, (txt, clr) in enumerate(labels):
                color = clr if (active_idx == i) else (180,180,180)
                cv2.putText(img, txt, (x, y), font, scale, color, thickness)
                (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
                x += tw + gap

            neck_angle_to_show = None
            if in_window and len(neck_samples) > 0:
                neck_angle_to_show = running_max
            elif last_window_max_angle is not None:
                neck_angle_to_show = last_window_max_angle

            neck_y = y + 28
            if neck_angle_to_show is None:
                neck_txt = "Neck: -"
                neck_clr = (180,180,180)
            else:
                neck_txt = f"Neck: {neck_angle_to_show:.1f}deg"
                neck_clr = (0,255,0) if neck_angle_to_show <= ANGLE_THRESHOLD else (0,0,255)
            cv2.putText(img, neck_txt, (10, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, neck_clr, 2)

            fps_y = neck_y + 28
            cv2.putText(img, f"FPS: {fps}", (10, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50), 2)

            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[Run] User exit (ESC).")
                break
            elif key == ord('r'):
                alarm_sent = False
                posture_sent = False
                no_face_sent = False
                print("[Run] flags manually reset.")

    finally:
        print("[Cleanup] Releasing resources...")
        try:
            if sock:
                sock.close()
        except:
            pass
        try:
            face_mesh.close()
        except:
            pass
        try:
            pose.close()
        except:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("[Exit] Bye.")
