# main_auth_pipeline_fixed.py
# 로그인 → VALUE/FLAG 수신 → (필요 시) 얼굴인증 대기 → 둘 다 OK면 모니터링(졸음+거북목)

import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import time
import re
import sys
import math

# =================== 네트워크/프로토콜 설정 ===================
HOST = "192.168.0.158"
PORT = 5000
SOCK_TIMEOUT_SEC = 3.0

LOGIN_MSG = "SEOL_SQL:PASSWD"   # ★ 서버 로그인 문자열(개행 없이 전송)

SRC      = "AI"
USER     = "seol"
AUTH_CMD = "AUTH"
TAG_VALUE = "VALUE"
TAG_FLAG  = "FLAG"

# =================== 카메라/로직 설정 ===================
CAM_INDEX = 0

# 얼굴 매칭 허용 오차(정규화 비율 기준)
TOL_EYE_EYE_RATIO       = 0.25
TOL_NOSE_LIPS_RATIO     = 0.35

# 거북목 샘플링 주기
ANGLE_THRESHOLD   = 15.0
VIS_THRESH_POSE   = 0.35
POSE_PERIOD       = 10.0     # 매 10초마다
POSE_WINDOW       = 3.0      # 3초간 샘플링
POSE_SAMPLE_INTV  = 0.5      # 0.5초 간격

# =================== 전역 상태 ===================
EYE_EYE = 0.0
NOSE_LIPS = 0.0
FOREHEAD_CHIN = 0.0
EAR_THRESHOLD_DEFAULT = 0.20   # 서버 값으로 갱신됨

FACE_FLAG = 0
AUTH_FLAG = 0

_sock = None
_stop_flag_thread = False
_flag_thread = None

# =================== 네트워크 유틸 ===================
def connect_blocking():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(SOCK_TIMEOUT_SEC)
    s.connect((HOST, PORT))
    s.settimeout(SOCK_TIMEOUT_SEC)
    return s

def send_login(sock):
    print(f"[Network] Sending login -> {LOGIN_MSG}")
    sock.sendall(LOGIN_MSG.encode("utf-8"))
    try:
        data = sock.recv(1024)
        msg = (data or b"").decode("utf-8", errors="ignore").strip()
        print(f"[Network] Login ACK: {msg if msg else '<empty>'}")
    except socket.timeout:
        print("[Network] Login ACK timeout (continue)")

def _send_and_recv_line(sock, msg: str) -> str:
    print(msg)  # 보낸 명령 로그
    sock.sendall(msg.encode("utf-8"))
    try:
        data = sock.recv(1024)
    except socket.timeout:
        return ""
    resp = (data or b"").decode("utf-8", errors="ignore").strip()
    print(resp)  # 받은 결과 로그
    return resp

_value_regex = re.compile(
    r"^SERVER:VALUE:(?P<eye>[+-]?\d+(?:\.\d+)?)\:(?P<nose>[+-]?\d+(?:\.\d+)?)\:(?P<fore>[+-]?\d+(?:\.\d+)?)\:(?P<ear>[+-]?\d+(?:\.\d+)?)$"
)
def request_value_and_update(sock):
    global EYE_EYE, NOSE_LIPS, FOREHEAD_CHIN, EAR_THRESHOLD_DEFAULT
    cmd = f"{SRC}:{USER}:{AUTH_CMD}:{TAG_VALUE}"
    resp = _send_and_recv_line(sock, cmd)
    m = _value_regex.match(resp)
    if not m:
        return False
    EYE_EYE       = float(m.group("eye"))
    NOSE_LIPS     = float(m.group("nose"))
    FOREHEAD_CHIN = float(m.group("fore"))
    EAR_THRESHOLD_DEFAULT = float(m.group("ear"))
    return True

_flag_regex = re.compile(r"^SERVER:FLAG:(?P<face>[01])\:(?P<rfid>[01])$")
def request_flag_and_update(sock):
    global FACE_FLAG, RFID_FLAG
    cmd = f"{SRC}:{USER}:{AUTH_CMD}:{TAG_FLAG}"
    resp = _send_and_recv_line(sock, cmd)
    m = _flag_regex.match(resp)
    if not m:
        return False
    FACE_FLAG = int(m.group("face"))
    RFID_FLAG = int(m.group("rfid"))
    return True

def start_flag_polling(sock):
    global _flag_thread, _stop_flag_thread
    if _flag_thread and _flag_thread.is_alive():
        return
    _stop_flag_thread = False
    _flag_thread = threading.Thread(target=_flag_poll_loop, args=(sock,), daemon=True)
    _flag_thread.start()

def _flag_poll_loop(sock):
    global _stop_flag_thread
    while not _stop_flag_thread:
        try:
            request_flag_and_update(sock)
        except Exception as e:
            print(f"[FLAG-POLL] error: {e}")
        for _ in range(50):  # 5초
            if _stop_flag_thread:
                break
            time.sleep(0.1)

def stop_flag_polling():
    global _flag_thread, _stop_flag_thread
    _stop_flag_thread = True
    if _flag_thread:
        _flag_thread.join(timeout=2.0)
        _flag_thread = None

# =================== 얼굴/피처 측정 ===================
mp_face = mp.solutions.face_mesh
_face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5)

def _dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

IDX_LEFT_EYE_OUT  = 133
IDX_RIGHT_EYE_OUT = 362
IDX_NOSE_TIP      = 1
IDX_MOUTH_CENTER  = 13
IDX_FOREHEAD      = 10
IDX_CHIN          = 152

def measure_face_metrics(landmarks, h, w):
    def pt(i): return (landmarks[i].x * w, landmarks[i].y * h)
    eye_eye = _dist(pt(IDX_LEFT_EYE_OUT), pt(IDX_RIGHT_EYE_OUT))
    nose_lips = _dist(pt(IDX_NOSE_TIP), pt(IDX_MOUTH_CENTER))
    forehead_chin = _dist(pt(IDX_FOREHEAD), pt(IDX_CHIN))
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
    sv_eye_ratio  = sv_eye  / sv_fc
    sv_nose_ratio = sv_nose / sv_fc
    ov_eye_ratio  = obs_vals["eye_over_fc"]
    ov_nose_ratio = obs_vals["nose_over_fc"]

    def within(a, b, tol):  # a≈b ?
        return abs(a-b) <= tol * b

    ok_eye  = within(ov_eye_ratio,  sv_eye_ratio,  TOL_EYE_EYE_RATIO)
    ok_nose = within(ov_nose_ratio, sv_nose_ratio, TOL_NOSE_LIPS_RATIO)
    return ok_eye and ok_nose

# =================== 인증 스레드 ===================
def auth_worker(sock, stop_event):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Camera] open failed")
        stop_event.set()
        return

    print("[Auth] waiting face... (ESC to stop)")
    win = "Auth waiting"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        res = _face.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 디버그 표시
            for i in (IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT, IDX_NOSE_TIP, IDX_MOUTH_CENTER, IDX_FOREHEAD, IDX_CHIN):
                cx, cy = int(lm[i].x * w), int(lm[i].y * h)
                cv2.circle(img, (cx,cy), 3, (0,255,255), -1)

            if FACE_FLAG == 0:
                obs = measure_face_metrics(lm, h, w)
                if obs:
                    matched = simple_face_match((EYE_EYE, NOSE_LIPS, FOREHEAD_CHIN), obs)
                    cv2.putText(img, f"match={'OK' if matched else 'NO'}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if matched else (0,0,255), 2)
                    if matched:
                        try:
                            msg = f"{SRC}:{USER}:FACE:OK"
                            print(msg)
                            sock.sendall(msg.encode("utf-8"))
                        except Exception as e:
                            print(f"[Auth] FACE:OK send error: {e}")

        if RFID_FLAG == 0:
            cv2.putText(img, "Please tap RFID...", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow(win, img)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()
            break

    cap.release()
    cv2.destroyWindow(win)

# =================== 모니터링(졸음+거북목) ===================
def compute_ear(lm, idxs, h, w):
    try:
        pts = [(lm[i].x*w, lm[i].y*h) for i in idxs]
    except Exception:
        return None
    A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0])-np.array(pts[3])) + 1e-8
    return (A+B)/(2.0*C)

LEFT = [33,160,158,133,153,144]
RIGHT= [263,387,385,362,380,373]

def run_monitoring():
    print("\n[Monitor] start (EAR + turtle-neck)")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    pose = mp.solutions.pose.Pose(
        model_complexity=1, static_image_mode=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[Camera] open failed"); return

    win = "Monitoring"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # drowsy state
    THRESH = float(EAR_THRESHOLD_DEFAULT)
    ema = None
    alpha = 0.35
    closed_cnt = 0
    open_cnt   = 0
    fps = 30
    prev = time.time(); framec = 0

    # pose window sched
    period_anchor = time.time()
    last_sample_t = 0.0
    neck_samples = []
    neck_text = "No pose yet"; neck_color = (180,180,180)
    running_max = None
    last_window_max = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # fps
        framec += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = framec
            framec = 0
            prev = now

        # Face/EAR
        status = "NO FACE"; scol=(200,200,200)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            le = compute_ear(lm, LEFT, h, w)
            re = compute_ear(lm, RIGHT, h, w)
            raw = None
            if le is not None and re is not None: raw = (le+re)/2.0
            elif le is not None: raw = le
            elif re is not None: raw = re
            if raw is not None:
                ema = raw if ema is None else alpha*raw + (1-alpha)*ema
                if ema < THRESH:
                    closed_cnt += 1; open_cnt = 0
                else:
                    open_cnt += 1
                    if open_cnt >= max(1,int(0.1*fps)): closed_cnt = 0
                if closed_cnt >= max(1,int(0.5*fps)):
                    status = "EYES CLOSED"; scol=(0,255,255)
                else:
                    status = "EYES OPEN";   scol=(0,255,0)

        # Pose(turtle-neck) 샘플링 윈도우
        elapsed = now - period_anchor
        if elapsed >= POSE_PERIOD:
            n = int(elapsed // POSE_PERIOD)
            period_anchor += n*POSE_PERIOD
            neck_samples = []
            elapsed = now - period_anchor
        in_window = (0.0 <= elapsed < POSE_WINDOW)

        if in_window:
            if now - last_sample_t >= POSE_SAMPLE_INTV:
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
                            vx = sx-nx; vy=sy-ny
                            if math.hypot(vx,vy)>1e-6:
                                ang = abs(math.degrees(math.atan2(vx,-vy)))
                                neck_samples.append(ang)
                                running_max = max(neck_samples)
                                neck_text = f"Sampling... max={running_max:.1f}°"
                                neck_color=(0,255,0) if running_max<=ANGLE_THRESHOLD else (0,0,255)
                                cv2.line(img,(int(nx*w),int(ny*h)),(int(sx*w),int(sy*h)),(255,0,255),3)
                                cv2.circle(img,(int(nx*w),int(ny*h)),6,(0,255,255),-1)
                                cv2.circle(img,(int(sx*w),int(sy*h)),6,(255,0,0),-1)
                        else:
                            neck_text="Shoulders not visible"; neck_color=(150,150,150)
                else:
                    neck_text="No pose"; neck_color=(150,150,150)
        else:
            if neck_samples:
                last_window_max = max(neck_samples)
                neck_text = f"Window MAX: {last_window_max:.1f}°"
                neck_color=(0,255,0) if last_window_max<=ANGLE_THRESHOLD else (0,0,255)
                neck_samples = []

        # Overlay
        cv2.putText(img, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, scol, 2)
        cv2.putText(img, neck_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, neck_color, 2)
        cv2.putText(img, f"FPS:{fps}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50),2)

        cv2.imshow(win, img)
        if cv2.waitKey(1)&0xFF==27:
            break

    cap.release()
    face_mesh.close()
    pose.close()
    cv2.destroyWindow(win)
    print("[Monitor] end")

# =================== main ===================
def main():
    global _sock
    # 1) 서버 연결 + 로그인
    try:
        _sock = connect_blocking()
        print(f"[NET] connected to {HOST}:{PORT}")
        send_login(_sock)  # ★ 로그인 먼저
    except Exception as e:
        print(f"[NET] connect/login failed: {e}")
        sys.exit(1)

    # 2) 카메라 점검
    tmp = cv2.VideoCapture(CAM_INDEX)
    if not tmp.isOpened():
        print("[Camera] open failed")
        sys.exit(1)
    tmp.release()

    # 3) VALUE/FLAG 1회 요청
    try: request_value_and_update(_sock)
    except Exception as e: print(f"[NET] VALUE err: {e}")
    try: request_flag_and_update(_sock)
    except Exception as e: print(f"[NET] FLAG err: {e}")

    # >>> 여기서 이미 1:1이면 인증 스킵하고 바로 모니터링 <<<
    if FACE_FLAG == 1 and RFID_FLAG == 1:
        print("[Main] flags already 1:1 → skip auth and start monitoring.")
        run_monitoring()
        _sock.close()
        _face.close()
        print("[Exit] bye.")
        return

    # 4) FLAG 폴링 시작 + 인증 스레드 시작
    start_flag_polling(_sock)
    stop_event = threading.Event()
    t_auth = threading.Thread(target=auth_worker, args=(_sock, stop_event), daemon=True)
    t_auth.start()

    # 5) 두 flag가 1이 될 때까지 대기
    print("[Main] waiting FACE_FLAG & RFID_FLAG == 1 ...")
    try:
        while not stop_event.is_set():
            if FACE_FLAG == 1 and RFID_FLAG == 1:
                print("[Main] both flags are 1. Proceed.")
                stop_event.set()
                break
            time.sleep(0.2)
    finally:
        stop_flag_polling()
        t_auth.join(timeout=2.0)

    # 6) 모니터링 시작
    run_monitoring()

    try:
        _sock.close()
    except: pass
    try:
        _face.close()
    except: pass
    print("[Exit] bye.")

if __name__ == "__main__":
    main()
