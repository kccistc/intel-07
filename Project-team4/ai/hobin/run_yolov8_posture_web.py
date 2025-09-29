#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
import numpy as np
import cv2
import threading
import socket

from ultralytics import YOLO
from iot_client_HB import IoTClient

# -------------------- 서버 신호에 따른 카메라 게이트 --------------------
cam_gate = threading.Event()                 # 카메라 on/off 제어
CMD_ON   = "SERVER:seol:ATTENDANCE:OK"
CMD_OFF  = "SERVER:seol:ATTENDANCE:STOP"

def on_server_message(text: str):
    if CMD_ON in text:
        print("[CMD] ATTENDANCE OK → camera ON")
        cam_gate.set()
    elif CMD_OFF in text:
        print("[CMD] ATTENDANCE STOP → camera OFF")
        cam_gate.clear()

# -------------------- 키포인트 인덱스 (YOLOv8 COCO-17) --------------------
KP = {
    "YOLO_COCO17": {
        "NOSE": 0, "NECK": None,
        "R_SHOULDER": 6, "R_ELBOW": 8, "R_WRIST": 10,
        "L_SHOULDER": 5, "L_ELBOW": 7, "L_WRIST": 9,
        "MID_HIP": None,
        "R_HIP": 12, "R_KNEE": 14, "R_ANKLE": 16,
        "L_HIP": 11, "L_KNEE": 13, "L_ANKLE": 15,
    }
}

# -------------------- 임계값 --------------------
BACK_TILT_ON = 26.0
BACK_TILT_OFF = 22.0
SHOULDER_SLOPE_ON = 14.0
SHOULDER_SLOPE_OFF = 10.0
BACK_TILT_HARD_ON = 32.0

LEG_LEGACY_ANKLES_CLOSE_RATIO = 0.07
LEG_LEGACY_ANKLE_YGAP = 0.07
ANKLES_DIST_NORM_THR = 1.10
PROX_RATIO_THR = 1.12
ANKLE_Y_GAP_THR = 0.04
KNEE_SWAP_THR_PIX = 24.0

# -------------------- 유틸 --------------------
def vangle_deg(v1, v2):
    a = np.asarray(v1, np.float32); b = np.asarray(v2, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return None
    cos_th = float(np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_th)))

def dist(p, q): return float(np.linalg.norm(np.asarray(p, np.float32)-np.asarray(q, np.float32)))
def mid(p, q):  return ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)

def take(body, idx, thr=0.2):
    if idx is None: return None
    x,y,c = body[idx]
    return (float(x), float(y)) if c >= thr else None

def median_pose(buf): return np.median(np.stack(buf, axis=0), axis=0) if buf else None

def seg_intersect(p1, p2, p3, p4):
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def on_seg(a,b,p):
        return (min(a[0],b[0])-1e-6 <= p[0] <= max(a[0],b[0])+1e-6 and
                min(a[1],b[1])-1e-6 <= p[1] <= max(a[1],b[1])+1e-6)
    c1,c2 = cross(p1,p2,p3), cross(p1,p2,p4)
    c3,c4 = cross(p3,p4,p1), cross(p3,p4,p2)
    if c1==0 and on_seg(p1,p2,p3): return True
    if c2==0 and on_seg(p1,p2,p4): return True
    if c3==0 and on_seg(p3,p4,p1): return True
    if c4==0 and on_seg(p3,p4,p2): return True
    return (c1>0)!=(c2>0) and (c3>0)!=(c4>0)

def is_profile_view(a):
    rs, ls = a["rs"], a["ls"]
    rh, lh = a["rh"], a["lh"]
    torso = a["torso_len"] or a["img_h"]
    shoulder_w = dist(rs, ls) if (rs and ls) else None
    hip_w      = dist(rh, lh) if (rh and lh) else None
    ratio_s = (shoulder_w / torso) if (shoulder_w and torso) else 1.0
    ratio_h = (hip_w / torso) if (hip_w and torso) else 1.0
    return (ratio_s < 0.60) or (ratio_h < 0.58)

def safe_vec(p_from, p_to):
    if not (p_from and p_to): return None
    return (p_to[0]-p_from[0], p_to[1]-p_from[1])

def leg_vector_for_backcheck(a):
    hip_mid, rk, lk, ra, la = a["hip_mid"], a["rk"], a["lk"], a["ra"], a["la"]
    if not hip_mid: return None
    knee_mid  = mid(rk, lk) if (rk and lk) else None
    ankle_mid = mid(ra, la) if (ra and la) else None
    target = knee_mid if knee_mid else ankle_mid
    return safe_vec(hip_mid, target) if target else None

# -------------------- 기준점 계산 --------------------
def compute_anchors(body, mp, img_h):
    nose = take(body, mp["NOSE"])
    rs, ls = take(body, mp["R_SHOULDER"]), take(body, mp["L_SHOULDER"])
    rh, lh = take(body, mp["R_HIP"]), take(body, mp["L_HIP"])
    rk, lk = take(body, mp["R_KNEE"]), take(body, mp["L_KNEE"])
    ra, la = take(body, mp["R_ANKLE"]), take(body, mp["L_ANKLE"])

    neck = mid(rs, ls) if (rs and ls) else (take(body, mp["NECK"]) if mp.get("NECK") is not None else None)
    mid_hip = mid(rh, lh) if (rh and lh) else (take(body, mp["MID_HIP"]) if mp.get("MID_HIP") is not None else None)
    sh_mid = mid(rs, ls) if (rs and ls) else None
    torso_len = (dist(neck, mid_hip) if (neck and mid_hip) else (dist(sh_mid, mid_hip) if (sh_mid and mid_hip) else None))
    return {"nose":nose,"neck":neck,"sh_mid":sh_mid,"hip_mid":mid_hip,
            "rs":rs,"ls":ls,"rh":rh,"lh":lh,"rk":rk,"lk":lk,"ra":ra,"la":la,
            "torso_len":torso_len,"img_h":img_h}

# -------------------- 판정 --------------------
def judge_slouch(a, latched=False):
    sh_mid, hip_mid, rs, ls = a["sh_mid"], a["hip_mid"], a["rs"], a["ls"]
    if not (sh_mid and hip_mid and rs and ls): return False
    back_vec = (hip_mid[0]-sh_mid[0], hip_mid[1]-sh_mid[1])
    back_tilt = vangle_deg(back_vec, (0.0, 1.0))
    shoulder_slope = vangle_deg((ls[0]-rs[0], ls[1]-rs[1]), (1.0, 0.0))
    if back_tilt is None or shoulder_slope is None: return False

    torso = a["torso_len"] or a["img_h"]
    img_h = a["img_h"]
    if torso / max(img_h, 1.0) < 0.18: return False

    leg_vec = leg_vector_for_backcheck(a)
    if leg_vec is not None:
        hip_leg_angle = vangle_deg(back_vec, leg_vec)
        if hip_leg_angle is not None and 75.0 <= hip_leg_angle <= 105.0:
            return False

    ON_BACK  = max(26.0, BACK_TILT_ON)
    ON_SH    = max(14.0, SHOULDER_SLOPE_ON)
    HARD_ON  = max(32.0, BACK_TILT_HARD_ON)

    if not latched:
        cond_and  = (back_tilt >= ON_BACK) and (shoulder_slope >= ON_SH)
        cond_hard = (back_tilt >= HARD_ON)
        return bool(cond_and or cond_hard)
    else:
        off_relaxed = (back_tilt < BACK_TILT_OFF) or (shoulder_slope < SHOULDER_SLOPE_OFF)
        return not off_relaxed

def judge_leg_cross(a):
    rh, lh, rk, lk, ra, la, hip_mid = a["rh"], a["lh"], a["rk"], a["lk"], a["ra"], a["la"], a["hip_mid"]
    if not (rh and lh and rk and lk and ra and la and hip_mid): return False

    img_h = a["img_h"]; hip_w = dist(rh, lh) or 1.0; hip_mid_x = hip_mid[0]
    crossed_sides = ((ra[0] > hip_mid_x) == (la[0] > hip_mid_x))
    ankles_y_gap  = abs(ra[1]-la[1]) / max(img_h, 1.0)
    ankles_close  = dist(ra, la) < LEG_LEGACY_ANKLES_CLOSE_RATIO * img_h
    legacy_strong = crossed_sides and ankles_close and (ankles_y_gap > max(LEG_LEGACY_ANKLE_YGAP, 0.08))

    same_side = np.sign(ra[0] - hip_mid_x) == np.sign(la[0] - hip_mid_x)
    ankles_dist_norm = dist(ra, la) / hip_w
    condA = bool(same_side and ankles_dist_norm < ANKLES_DIST_NORM_THR)
    order_knee  = rk[0] - lk[0]
    order_ankle = ra[0] - la[0]
    condB = bool(np.sign(order_knee) != np.sign(order_ankle) or abs(order_knee - order_ankle) < KNEE_SWAP_THR_PIX)
    condC = seg_intersect(rk, ra, lk, la)
    prox_r = dist(rk, la) / max(dist(rk, ra), 1e-3)
    prox_l = dist(lk, ra) / max(dist(lk, la), 1e-3)
    condD = bool((prox_r < PROX_RATIO_THR) or (prox_l < PROX_RATIO_THR))
    condE = bool(same_side and ankles_y_gap > ANKLE_Y_GAP_THR)

    profile = is_profile_view(a) or getattr(a, "_side_cam_forced", False)
    if profile:
        if getattr(a, "_side_cam_strict_off", False): return False
        if condC:
            if ankles_dist_norm <= 0.90: return True
            if condD and ankles_dist_norm < 0.85 and ankles_y_gap >= 0.055: return True
            return False
        else:
            if condD and ankles_dist_norm < 0.85 and ankles_y_gap >= 0.055: return True
            return False

    if condC: return True
    score = sum([condA, condB, condD, condE])
    return bool(legacy_strong or score >= 2)

# -------------------- UI --------------------
def draw_text(img, text, org, color=(0,255,0), scale=0.9, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_text_outline(img, text, org, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick, cv2.LINE_AA)

def draw_labels(img, latched):
    if latched == "neck":
        line, color = "WARN: Neck", (0,0,255)
    elif latched == "back":
        line, color = "WARN: Back", (0,0,255)
    elif latched == "leg":
        line, color = "WARN: Leg Cross", (0,0,255)
    else:
        line, color = "Posture OK", (0,200,0)
    draw_text(img, line, (10, 28), color)
    if line.startswith("WARN"):
        draw_text(img, "(latched)", (10, 58), color)

def draw_camera_fps_topright(img, fps_text):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    org = (w - 10 - tw, 22)  # 우측 상단 여백
    draw_text_outline(img, fps_text, org, scale=0.7, thick=2)

# -------------------- IoT 전송 헬퍼 --------------------
def client_send_any(client, line: str) -> bool:
    if not line.endswith("\n"):
        line = line + "\n"
    for name in ("send_line","send_text","send","send_command","send_msg","send_message"):
        if hasattr(client, name):
            try:
                getattr(client, name)(line)
                return True
            except Exception as e:
                print(f"[WARN] client {name} error:", e)
                return False
    try:
        sock = getattr(client, "sock", None)
        if sock:
            sock.sendall(line.encode("utf-8", "ignore"))
            return True
    except Exception as e:
        print("[WARN] direct socket send failed:", e)
    return False

def state_to_msg(state, user):
    if state == "neck": return f"AI:{user}:POSTURE:BAD:neck"
    if state == "back": return f"AI:{user}:POSTURE:BAD:back"
    if state == "leg":  return f"AI:{user}:POSTURE:BAD:leg"
    return f"AI:{user}:POSTURE:OK"

# -------------------- YOLO 결과 → (17,3) --------------------
def yolo_person_to_body17(person_res):
    try:
        kp = getattr(person_res, "keypoints", None)
        if kp is None: return None
        xy = getattr(kp, "xy", None)
        if xy is None or xy.ndim != 3: return None
        nperson, nkpt, _ = tuple(xy.shape)
        if nperson == 0 or nkpt < 17: return None

        xy = xy.cpu().numpy()
        conf = getattr(kp, "conf", None)
        if conf is None:
            c = np.ones((nperson, nkpt), dtype=np.float32)
        else:
            c = conf.cpu().numpy().astype(np.float32)

        k = 0
        body = np.zeros((17, 3), dtype=np.float32)
        body[:, 0:2] = xy[k][:17]
        body[:, 2] = c[k][:17]
        return body
    except Exception:
        return None

# ==================== Flask 웹 스트리머 ====================
from flask import Flask, Response
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

def set_latest_frame(frame_bgr):
    global latest_frame
    with frame_lock:
        latest_frame = frame_bgr.copy()

def _mjpeg_generator():
    import time as _t
    while True:
        with frame_lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            _t.sleep(0.01); continue
        ok, jpg = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

@app.get("/")
def _index():
    return '<img src="/video" style="max-width:100vw;max-height:100vh">'

@app.get("/video")
def _video():
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask(host='192.168.0.111', port=8081):
    app.run(host=host, port=port, threaded=True, use_reloader=False)

# -------------------- 메인 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_cam", action="store_true", default=False, help="카메라 측면 설치 가정")
    parser.add_argument("--ip", default="192.168.0.158")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--login_user", default="AI2")
    parser.add_argument("--passwd",     default="PASSWD")
    parser.add_argument("--user",       default="seol")
    parser.add_argument("--model", default=str(os.path.expanduser("~/yolov8n-pose.engine")))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--vid_stride", type=int, default=3)
    parser.add_argument("--no_gui", action="store_true", default=False, help="창 표시 끄기")
    parser.add_argument("--auto_on", action="store_true", help="IoT 명령 없이도 카메라 바로 켬")
    args = parser.parse_args()

    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
    print(f"[INFO] YOLO model={args.model}, imgsz={args.imgsz}, conf={args.conf}")

    # IoT 연결
    client = IoTClient(args.ip, args.port, args.login_user, args.passwd)
    if hasattr(client, "set_on_message"): client.set_on_message(on_server_message)
    try: client.connect()
    except Exception as e: print("[WARN] IoT connect failed:", e)

    if args.auto_on: cam_gate.set()

    # 모델 로드
    model = YOLO(args.model)

    # 카메라 (게이트 열릴 때만 오픈)
    cap = None
    prime_h, prime_w = 480, 640
    set_latest_frame(np.zeros((prime_h, prime_w, 3), dtype=np.uint8))

    mp = KP["YOLO_COCO17"]
    pose_buf, last_t = [], time.time()
    latched_state = None
    last_sent_msg = None
    frame_idx = 0
    last_vis = None

    # --- Camera Capture FPS (EMA) ---
    cap_fps = 0.0
    cap_alpha = 0.90
    t_prev_cap = time.time()

    try:
        while True:
            # 게이트 닫힘 → 카메라 반납 + 대기
            if not cam_gate.is_set():
                if cap is not None:
                    cap.release(); cap = None
                time.sleep(0.1)
                continue

            # 게이트 열림인데 아직 카메라가 없으면 지금 연다
            if cap is None:
                cap = cv2.VideoCapture(args.cam)
                if not cap.isOpened():
                    print("[WARN] Camera open failed. retry...")
                    time.sleep(0.5)
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  prime_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, prime_h)
                cap.set(cv2.CAP_PROP_FPS, 15)

            # 프레임 읽기
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Capture FPS 업데이트 (순수 카메라 프레임레이트)
            now_cap = time.time()
            inst_cap_fps = 1.0 / max(now_cap - t_prev_cap, 1e-6)
            cap_fps = cap_alpha * cap_fps + (1.0 - cap_alpha) * inst_cap_fps
            t_prev_cap = now_cap

            frame_idx += 1
            do_infer = not (args.vid_stride > 1 and (frame_idx % args.vid_stride) != 0)

            if do_infer:
                res = model.predict(
                    source=frame, imgsz=args.imgsz, conf=args.conf,
                    classes=[0], max_det=1, verbose=False
                )[0]
                annotated = res.plot()
                last_vis = annotated.copy()

                body = yolo_person_to_body17(res)
                if body is not None:
                    pose_buf.append(body)
                    if len(pose_buf) > 60: pose_buf = pose_buf[-60:]

                    now2 = time.time()
                    if now2 - last_t >= args.interval and pose_buf:
                        med = median_pose(pose_buf)
                        a = compute_anchors(med, mp, annotated.shape[0])
                        if args.side_cam: a["_side_cam_forced"] = True

                        back_bad = judge_slouch(a, latched=(latched_state=="back"))
                        leg_bad  = judge_leg_cross(a)

                        ts = time.strftime('%H:%M:%S')
                        print(f"[{ts}]")
                        print(f"slouch={back_bad}")
                        print(f"leg_cross={leg_bad}")

                        new_state = ("back" if back_bad else "leg" if leg_bad else None)
                        if new_state != latched_state:
                            latched_state = new_state
                            msg = state_to_msg(latched_state, args.user)
                            if msg != last_sent_msg:
                                client_send_any(client, msg)
                                last_sent_msg = msg

                        last_t = now2
                        pose_buf.clear()
                else:
                    pose_buf.clear()

                vis = last_vis if last_vis is not None else frame
            else:
                vis = last_vis if last_vis is not None else frame

            # 라벨 + Camera FPS(우측 상단) — 오직 이 한 줄만 표시
            draw_labels(vis, latched_state)
            draw_camera_fps_topright(vis, f"Camera FPS: {cap_fps:.1f}")

            set_latest_frame(vis)

            if not args.no_gui:
                cv2.imshow("YOLOv8 Posture Monitor", vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break

    finally:
        try:
            if cap is not None: cap.release()
        except: pass
        if not args.no_gui:
            cv2.destroyAllWindows()
        try: client.close()
        except: pass

if __name__ == "__main__":
    # --- Flask 스트리머 백그라운드 기동 ---
    t = threading.Thread(target=run_flask, kwargs={'host':'192.168.0.111','port':8081}, daemon=True)
    t.start()
    main()
