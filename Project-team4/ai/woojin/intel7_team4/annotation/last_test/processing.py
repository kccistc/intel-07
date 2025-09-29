# processing.py
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from state import AppState
from net import guarded_send

# Provide sensible defaults so module can be used standalone
SRC = "AI"
USER = "seol"
FOREHEAD_CHIN = 0.0
NOSE_LIPS = 0.0
EYE_EYE = 0.0
EAR_THRESHOLD = 0.20

# indices used by the logic
IDX_LEFT_EYE_OUT  = 133
IDX_RIGHT_EYE_OUT = 362
IDX_NOSE_TIP      = 1
IDX_MOUTH_CENTER  = 13
IDX_FOREHEAD      = 10
IDX_CHIN          = 152
LEFT_EAR_IDX = [33,160,158,133,153,144]
RIGHT_EAR_IDX = [263,387,385,362,380,373]

def dist(a,b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

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

def run_auth_local(face_mesh, cam_index:int, timeout=30.0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap or not cap.isOpened():
        print("[AUTH] camera open failed")
        return False
    match_count = 0
    start = time.time()
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        img = cv2.flip(frame,1)
        h,w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            res = face_mesh.process(rgb)
        except Exception:
            res = None
        if res and getattr(res, "multi_face_landmarks", None):
            lm = res.multi_face_landmarks[0].landmark
            obs = measure(lm, w, h)
            # use server-provided measurement if available (FOREHEAD_CHIN global)
            thresh = FOREHEAD_CHIN if FOREHEAD_CHIN and FOREHEAD_CHIN > 1e-6 else 60
            if obs and obs["forehead_chin"] > thresh:
                match_count += 1
            else:
                match_count = 0
        else:
            match_count = 0
        if match_count >= 3:
            cap.release()
            return True
    cap.release()
    return False

def processing_loop(app_state: AppState, sock, mjpg_url: str, cam_index:int, mjpg_prefer=True, EAR_THRESHOLD_LOCAL=0.20):
    global EAR_THRESHOLD
    if EAR_THRESHOLD_LOCAL:
        EAR_THRESHOLD = EAR_THRESHOLD_LOCAL

    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=True, min_detection_confidence=0.05, min_tracking_confidence=0.05)
    pose = mp_pose.Pose(model_complexity=0, static_image_mode=False,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = None
    if mjpg_prefer:
        try:
            cap = cv2.VideoCapture(mjpg_url)
            start = time.time()
            while time.time() - start < 3.0:
                if cap.isOpened():
                    ret,_ = cap.read()
                    if ret: break
                time.sleep(0.05)
            if not (cap and cap.isOpened()):
                try: cap.release()
                except: pass
                cap = None
        except Exception:
            cap = None

    if cap is None:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        start = time.time()
        while time.time() - start < 3.0:
            if cap.isOpened():
                ret,_ = cap.read()
                if ret: break
            time.sleep(0.05)
        if not (cap and cap.isOpened()):
            print("[PROC] cannot open capture")
            return

    fps = 15; prev = time.time(); framec = 0
    alpha = 0.35; ema = None
    closed_cnt = 0; closed_total = 0; alarm_sent = False
    period_anchor = time.time(); last_sample_t = 0.0
    neck_samples = []; posture_sent = False
    face_present = False; last_face_seen = 0.0

    while not app_state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02); continue
        framec += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = framec; framec = 0; prev = now
            with app_state.status_lock:
                app_state.current_fps = fps

        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        status = "NO FACE"; scol=(150,150,150)
        raw = None

        res = face_mesh.process(rgb)
        if res and getattr(res, "multi_face_landmarks", None):
            last_face_seen = now
            if not face_present:
                face_present = True
                with app_state.status_lock:
                    app_state.current_face_state = "IN"
                # use guarded_send so main can control whether sending is allowed
                guarded_send(app_state, sock, f"{SRC}:{USER}:FACE:IN")

            lm = res.multi_face_landmarks[0].landmark
            le = compute_ear(lm, LEFT_EAR_IDX, h, w)
            re = compute_ear(lm, RIGHT_EAR_IDX, h, w)
            if le is not None and re is not None: raw = (le + re)/2.0
            elif le is not None: raw = le
            elif re is not None: raw = re

            if raw is not None:
                ema = raw if ema is None else alpha*raw + (1-alpha)*ema
                closed_thr = max(1, int(round(0.5 * fps)))
                alarm_thr  = max(1, int(round(5.0 * fps)))
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
                with app_state.status_lock:
                    app_state.current_eye_status = status

                # send alert messages through guarded_send
                if status == "DROWSY ALERT!" and not alarm_sent:
                    guarded_send(app_state, sock, f"{SRC}:{USER}:SLP:ON")
                    alarm_sent = True
                    with app_state.status_lock:
                        app_state.current_slp_state = "ON"
                if alarm_sent and status != "DROWSY ALERT!":
                    guarded_send(app_state, sock, f"{SRC}:{USER}:SLP:OFF")
                    alarm_sent = False
                    with app_state.status_lock:
                        app_state.current_slp_state = "OFF"
        else:
            if face_present and (now - last_face_seen) >= 5.0:
                face_present = False
                with app_state.status_lock:
                    app_state.current_face_state = "OUT"
                guarded_send(app_state, sock, f"{SRC}:{USER}:FACE:OUT")
            closed_cnt = 0; closed_total = 0; status="NO FACE"; scol=(180,180,180)
            with app_state.status_lock:
                app_state.current_eye_status = status

        # posture sampling (simplified)
        elapsed = now - period_anchor
        if elapsed >= 10.0:
            n = int(elapsed // 10.0)
            period_anchor += n * 10.0
            neck_samples = []
            elapsed = now - period_anchor
        in_window = (0.0 <= elapsed < 3.0)
        if in_window and (now - last_sample_t >= 0.5):
            last_sample_t = now
            pr = pose.process(rgb)
            if pr and getattr(pr, "pose_landmarks", None):
                pl = pr.pose_landmarks.landmark
                cand = []
                for idx in (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value):
                    if pl[idx].visibility > 0.35:
                        cand.append((pl[idx].x, pl[idx].y))
                if cand:
                    sx = sum(p[0] for p in cand)/len(cand)
                    sy = sum(p[1] for p in cand)/len(cand)
                    ls = pl[mp_pose.PoseLandmark.LEFT_SHOULDER.value]; rs = pl[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    if ls.visibility>0.35 and rs.visibility>0.35:
                        nx = (ls.x+rs.x)/2.0; ny=(ls.y+rs.y)/2.0
                        vx = sx-nx; vy = sy-ny
                        if (vx*vx+vy*vy) > 1e-6:
                            import math
                            ang = abs(math.degrees(math.atan2(vx,-vy)))
                            neck_samples.append(ang)
                            with app_state.status_lock:
                                app_state.current_neck_angle = max(neck_samples)
                            # posture alert
                            if app_state.current_neck_angle is not None and app_state.current_neck_angle > 15.0 and not posture_sent:
                                guarded_send(app_state, sock, f"{SRC}:{USER}:POSTURE:BAD:neck")
                                posture_sent = True
                            if app_state.current_neck_angle is not None and app_state.current_neck_angle <= 15.0 and posture_sent:
                                posture_sent = False

        # overlay & JPEG encode
        try:
            cv2.putText(img, f"Eye:{status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scol, 2)
            with app_state.status_lock:
                na = app_state.current_neck_angle
                fs = app_state.current_face_state
                ss = app_state.current_slp_state
                es = app_state.current_eye_status
                fps_loc = app_state.current_fps
            neck_text = f"Neck:-" if na is None else f"Neck:{na:.1f}deg"
            cv2.putText(img, neck_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(img, f"FACE:{fs}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(img, f"SLP:{ss}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(img, f"FPS:{fps_loc}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)
        except Exception:
            pass

        ret2, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret2:
            app_state.latest_jpeg = jpg.tobytes()
            app_state.frame_event.set()
            app_state.frame_event.clear()

    cap.release()
    face_mesh.close()
    pose.close()
    print("[PROC] exiting")
