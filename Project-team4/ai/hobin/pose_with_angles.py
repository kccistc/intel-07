#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------설명주석----------------------
# [개요]
# - YOLO(.pt) 포즈 추론 + 각도 계산 + 스켈레톤 렌더링
# - Flask MJPEG 웹스트림 + Jetson 로컬 모니터 동시 출력
# - IoT 서버 연결 (옵션)
# - 실행 커맨드 단순화: python3 pose_with_angles.py
# ----------------------------------------------------

import os, cv2, math, time, torch, threading, socket, subprocess
import numpy as np
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response
try:
    from iot_client_HB import IoTClient
except ImportError:
    IoTClient = None

# -----------------------설명주석----------------------
# [런타임 환경변수/카메라 설정]
# ----------------------------------------------------
os.environ["ULTRALYTICS_REQUIREMENTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    subprocess.run([
        "v4l2-ctl",
        "--set-fmt-video=width=640,height=480,pixelformat=MJPG"
    ], check=True)
    subprocess.run(["v4l2-ctl", "--set-parm=30"], check=True)
    print("[INFO] Camera forced to MJPG 640x480 @30fps")
except Exception as e:
    print("[WARN] v4l2-ctl setting failed:", e)

# -----------------------설명주석----------------------
# [모델/서버 기본값]
# ----------------------------------------------------
MODEL_PATH   = "/home/jetson/custom_yolo/best.pt"
CAMERA_INDEX = 0
IMG_SIZE     = 160
CONF_THRESH  = 0.40
MIN_KPT_CONF = 0.60

IOT_IP       = "192.168.0.158"
IOT_PORT     = 5000
LOGIN_USER   = "AI2"
LOGIN_PASS   = "PASSWD"
USER_NAME    = "seol"

WEB_HOST     = "0.0.0.0"
WEB_PORT     = 8081

JPEG_QUALITY = 60
CAM_FPS      = 30
BUFFER_SIZE  = 1

# -----------------------설명주석----------------------
# [키포인트 매핑 & 스켈레톤 정의]
# ----------------------------------------------------
keypoint_index = {
    'neck1': 0,
    'neck2': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'right_arm': 4,   # <-- 템플릿상 4가 오른팔(팔꿈치/팔)입니다
    'left_arm': 5,    # <-- 5가 왼팔
    'back1': 6,
    'back2': 7,
    'waist': 8,
    'left_hip': 9,
    'right_hip': 10,
    'left_knee': 11,
    'left_ankle': 12,
    'right_knee': 13,
    'right_ankle': 14,
    'left_eye': 15,
    'nose': 16,
    'right_eye': 17,
    'right_ear': 18,
    'left_ear': 19
}
skeleton = [
    # 얼굴/머리
    ('left_ear', 'left_eye'),
    ('left_eye', 'nose'),
    ('nose', 'right_eye'),
    ('right_eye', 'right_ear'),
    ('nose', 'neck1'),
    ('neck1', 'neck2'),

    # 어깨-목
    ('left_shoulder', 'neck2'),
    ('right_shoulder', 'neck2'),

    # 팔
    ('left_shoulder', 'left_arm'),
    ('right_shoulder', 'right_arm'),

    # 척추 라인
    ('neck2', 'back1'),
    ('back1', 'back2'),
    ('back2', 'waist'),

    # 골반/다리
    ('waist', 'left_hip'),
    ('waist', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]
angle_targets = {
    'Neck bent': ('neck1','neck2','back1'),
    'Back bent': ('neck2','waist','back2'),
    'Leg twist': ('left_hip','right_hip','right_knee')
}

# -----------------------설명주석----------------------
# [헬퍼 함수/스무딩]
# ----------------------------------------------------
def calculate_angle(a,b,c):
    if None in (a,b,c): return None
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                       math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(ang)
    return ang if ang<180 else 360-ang

def get_center_person_index(results, frame_width):
    if not results: return None
    kobj = getattr(results[0],"keypoints",None)
    if kobj is None: return None
    xy = getattr(kobj,"xy",None)
    centers_x = None
    if xy is not None and hasattr(xy,"shape") and xy.shape[0]>0:
        centers_x = xy[...,0].mean(dim=1).detach().cpu().numpy()
    if centers_x is None:
        data = getattr(kobj,"data",None)
        if data is None or (hasattr(data,"shape") and data.shape[0]==0): return None
        centers_x = data[...,0].mean(dim=1).detach().cpu().numpy()
    if centers_x is None: return None
    centers_x = np.asarray(centers_x,dtype=float)
    if centers_x.size==0 or np.all(np.isnan(centers_x)): return None
    valid = ~np.isnan(centers_x)
    if not np.any(valid): return None
    frame_center = frame_width/2.0
    idx = int(np.argmin(np.abs(centers_x[valid]-frame_center)))
    return int(np.arange(len(centers_x))[valid][idx])

_smooth_pts={}
_SMOOTH_ALPHA=0.5
_DIST_RATIO=0.35
def _ema(name,pt):
    if pt is None: return _smooth_pts.get(name)
    prev=_smooth_pts.get(name)
    if prev is None: _smooth_pts[name]=pt
    else:
        _smooth_pts[name]=(int(prev[0]*(1-_SMOOTH_ALPHA)+pt[0]*_SMOOTH_ALPHA),
                           int(prev[1]*(1-_SMOOTH_ALPHA)+pt[1]*_SMOOTH_ALPHA))
    return _smooth_pts[name]

# -----------------------설명주석----------------------
# [Flask MJPEG 서버]
# ----------------------------------------------------
app = Flask(__name__)
_latest_frame=None
_frame_lock=threading.Lock()

def _set_latest_frame(bgr):
    global _latest_frame
    with _frame_lock:
        _latest_frame = bgr.copy()

def _mjpeg_stream():
    while True:
        with _frame_lock:
            f=None if _latest_frame is None else _latest_frame.copy()
        if f is None:
            time.sleep(0.01); continue
        ok,jpg=cv2.imencode('.jpg',f,[int(cv2.IMWRITE_JPEG_QUALITY),JPEG_QUALITY])
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n')

@app.get('/')
def _root():
    return '<img src="/video" style="max-width:100vw;max-height:100vh">'

@app.get('/video')
def _video():
    return Response(_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------설명주석----------------------
# [렌더링 색상]
# ----------------------------------------------------
SKEL_COLOR=(255,200,0)
JOINT_COLOR=(30,144,255)
WARN_COLOR=(0,0,255)
OK_COLOR=(0,200,0)
JOINT_RADIUS=3
THICK_LINE=2

# -----------------------설명주석----------------------
# [메인 루프]
# ----------------------------------------------------
def main():
    # IoT 연결
    client=None; raw_sock=None
    if IoTClient:
        try:
            client=IoTClient(IOT_IP,IOT_PORT,LOGIN_USER,LOGIN_PASS)
            client.connect()
            print('[INFO] IoTClient connected')
        except Exception as e:
            print('[WARN] IoTClient fail',e)
    if client is None:
        try:
            raw_sock=socket.socket(); raw_sock.settimeout(3)
            raw_sock.connect((IOT_IP,IOT_PORT))
            print('[INFO] Raw TCP connected')
        except Exception as e:
            print('[ERR] TCP fail',e)

    # 모델 로드
    device=0 if torch.cuda.is_available() else 'cpu'
    model=YOLO(MODEL_PATH)
    try:
        if device!='cpu': model.fuse()
    except: pass

    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('[ERR] camera open fail'); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE,BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS,CAM_FPS)
    try: cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    except: pass

    frame_q=deque(maxlen=1)
    def _capture_loop():
        while True:
            ok,fr=cap.read()
            if not ok: time.sleep(0.005); continue
            frame_q.append(fr)
    threading.Thread(target=_capture_loop,daemon=True).start()

    while True:
        if not frame_q:
            time.sleep(0.001); continue
        frame=frame_q.pop()

        try:
            res=model.predict(frame,conf=CONF_THRESH,imgsz=IMG_SIZE,
                              verbose=False,device=device,half=(device!='cpu'))
        except TypeError:
            res=model.predict(frame,conf=CONF_THRESH,imgsz=IMG_SIZE,
                              verbose=False,device=device)

        if not res or res[0].keypoints is None:
            _set_latest_frame(frame)
            cv2.imshow('Posture',frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        # ====== 1회성 디버그 & 자동 매핑(17/20) ======
        if not hasattr(main, "_dbg_once"):
            n_det = int(res[0].boxes.shape[0]) if (hasattr(res[0], "boxes") and res[0].boxes is not None) else 0
            kobj = getattr(res[0], "keypoints", None)
            k_shape = None
            if kobj is not None and hasattr(kobj, "data"):
                try:
                    k_shape = tuple(kobj.data.shape)  # (persons, kpts, 3)
                except Exception:
                    pass
            print(f"[DBG] det={n_det}, keypoints_shape={k_shape}")
            # 포즈 헤드가 없으면 바로 알림
            if kobj is None:
                print("[ERR] This .pt seems NOT a pose model (keypoints is None).")
            else:
                # 키포인트 개수 자동 확인
                try:
                    k_per = int(kobj.data.shape[-2])
                    if k_per != len(keypoint_index):
                        print(f"[WARN] KP count mismatch: model={k_per}, code_mapping={len(keypoint_index)}")
                        # COCO17일 가능성 → 자동 매핑 전환
                        if k_per == 17:
                            print("[INFO] Auto-switch keypoint mapping to COCO-17")
                            globals()["keypoint_index"] = {
                                # 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
                                # 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,
                                # 9:left_wrist,10:right_wrist,11:left_hip,12:right_hip,
                                # 13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
                                'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
                                'left_shoulder': 5, 'right_shoulder': 6,
                                'left_arm': 7, 'right_arm': 8,  # elbow 대체
                                'waist': None, 'back1': None, 'back2': None, 'neck1': None, 'neck2': None,
                                'left_hip': 11, 'right_hip': 12,
                                'left_knee': 13, 'right_knee': 14,
                                'left_ankle': 15, 'right_ankle': 16
                            }
                        else:
                            print("[INFO] Keep current mapping; please align with your training YAML.")
                except Exception:
                    pass
            main._dbg_once = True
        # =============================================

        idx=get_center_person_index(res,frame.shape[1])
        if idx is None:
            _set_latest_frame(frame)
            cv2.imshow('Posture',frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        pose=res[0].keypoints.data[idx].cpu().numpy().reshape(-1,3)

        # 안전한 키포인트 접근 (임계값/NaN/범위 체크)
        def gp(name):
            i = keypoint_index.get(name)
            try:
                if i is None:
                    return None
                x, y, s = float(pose[i][0]), float(pose[i][1]), float(pose[i][2])
                if not (0.0 <= s <= 1.0):   # 신뢰도 범위 점검
                    return None
                if s < MIN_KPT_CONF:        # 임계값 미달
                    return None
                if np.isnan(x) or np.isnan(y):
                    return None
                return int(x), int(y)
            except Exception:
                return None

        h,w=frame.shape[:2]; diag2=w*w+h*h
        drew_any=False

        for a,b in skeleton:
            p1=_ema(a,gp(a)); p2=_ema(b,gp(b))
            if p1 and p2:
                dx,dy=p1[0]-p2[0],p1[1]-p2[1]
                if dx*dx+dy*dy<diag2*(_DIST_RATIO**2):
                    cv2.line(frame,p1,p2,SKEL_COLOR,THICK_LINE)
                    drew_any=True

        for n,i in keypoint_index.items():
            if i is not None:
                pt=_ema(n,gp(n))
                if pt:
                    cv2.circle(frame,pt,JOINT_RADIUS,JOINT_COLOR,-1)
                    drew_any=True

        y=30
        for label,(a,b,c) in angle_targets.items():
            ang=calculate_angle(_ema(a,gp(a)),_ema(b,gp(b)),_ema(c,gp(c)))
            if ang is not None:
                color=OK_COLOR if ang>30 else WARN_COLOR
                cv2.putText(frame,f'{label}:{int(ang)}',(10,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                y+=30

        if not drew_any:
            cv2.putText(frame,"No skeleton drawn: low conf or mapping mismatch",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        _set_latest_frame(frame)
        cv2.imshow('Posture',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

# -----------------------설명주석----------------------
# [엔트리포인트]
# ----------------------------------------------------
if __name__=='__main__':
    threading.Thread(target=lambda: app.run(host=WEB_HOST,port=WEB_PORT,
                                            threaded=True,use_reloader=False),
                     daemon=True).start()
    main()
