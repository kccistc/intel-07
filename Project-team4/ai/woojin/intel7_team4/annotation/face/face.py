# local_face_match_test.py
# 서버 없이 로컬에서 forehead-chin, nose-lips, eye-eye 측정 & 매칭 테스트용 스크립트.

import cv2
import mediapipe as mp
import time
import math
import numpy as np
import json
from pathlib import Path

# ---------------- 설정 (필요시 수정) ----------------
CAM_INDEX = 0
REFINE = True   # face_mesh refine_landmarks
TOL_DEFAULT = 0.30   # 기본 허용오차 (30%)
SAVE_DIR = Path("./captured_server_values")
SAVE_DIR.mkdir(exist_ok=True)
# ---------------------------------------------------

# 고정 landmark 인덱스 (요구하신 스키마)
IDX_FOREHEAD = 10
IDX_CHIN     = 152
IDX_NOSE     = 1
IDX_MOUTH    = 13
IDX_LEFT_EYE_OUT  = 133
IDX_RIGHT_EYE_OUT = 362

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def measure(lm, w, h):
    def pt(i): return (lm[i].x * w, lm[i].y * h)
    try:
        fore = dist(pt(IDX_FOREHEAD), pt(IDX_CHIN))
        nose = dist(pt(IDX_NOSE), pt(IDX_MOUTH))
        eye  = dist(pt(IDX_LEFT_EYE_OUT), pt(IDX_RIGHT_EYE_OUT))
    except Exception as e:
        return None
    return {"fore": fore, "nose": nose, "eye": eye,
            "eye_over_fc": eye / (fore + 1e-9),
            "nose_over_fc": nose / (fore + 1e-9)}

def within(a, b, tol):
    # relative comparison: |a - b| <= tol * b
    return abs(a - b) <= tol * abs(b)

def pretty(v):
    return f"{v:.1f}px"

def save_server_vals(path: Path, vals: dict):
    data = {
        "created_at": time.time(),
        "forehead_chin": vals["fore"],
        "nose_lips": vals["nose"],
        "eye_eye": vals["eye"]
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"[SAVE] server-sim saved -> {path}")

def load_server_vals(path: Path):
    if not path.exists():
        return None
    try:
        j = json.loads(path.read_text())
        return (float(j["forehead_chin"]), float(j["nose_lips"]), float(j["eye_eye"]))
    except Exception as e:
        print("load error:", e)
        return None

def main():
    tol = TOL_DEFAULT
    server_vals = None  # tuple (fore, nose, eye)
    match_mode = False

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=REFINE,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera open failed (index {}).".format(CAM_INDEX))
        return

    win = "Local Face Match Test   (c:capture server vals, m:toggle match, t:+tol, y:-tol, ESC/q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Controls:")
    print("  c : capture current obs as 'server' values (saved to ./captured_server_values/)")
    print("  m : toggle matching display (compare obs vs simulated server vals)")
    print("  t : increase tol by 0.05")
    print("  y : decrease tol by 0.05")
    print("  s : save current server values to json file (prompted filename)")
    print("  ESC/q : quit")

    last_print = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = face_mesh.process(rgb)
        obs = None
        used = False

        if res.multi_face_landmarks and len(res.multi_face_landmarks) > 0:
            lm = res.multi_face_landmarks[0].landmark
            obs = measure(lm, w, h)
            # draw control points
            for idx in (IDX_FOREHEAD, IDX_CHIN, IDX_NOSE, IDX_MOUTH, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT):
                x = int(lm[idx].x * w); y = int(lm[idx].y * h)
                cv2.circle(img, (x,y), 3, (0,255,255), -1)
            used = True

        # overlay measurements
        y0 = 30
        col = (0,200,0)
        if obs:
            fore_s = pretty(obs["fore"])
            nose_s = pretty(obs["nose"])
            eye_s  = pretty(obs["eye"])
            cv2.putText(img, f"forehead-chin: {fore_s}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,230,50), 2)
            cv2.putText(img, f"nose-lips:      {nose_s}", (10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,230,50), 2)
            cv2.putText(img, f"eye-eye:        {eye_s}", (10, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,230,50), 2)
        else:
            cv2.putText(img, "No face detected", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,150,150), 2)

        # show server values if exist
        if server_vals:
            sf, sn, se = server_vals
            cv2.putText(img, f"SERVER(fore,nose,eye): {sf:.1f}, {sn:.1f}, {se:.1f}", (10, y0+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            cv2.putText(img, f"Tolerance: {tol*100:.0f}%", (10, y0+130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        # matching
        match_result = None
        if match_mode and server_vals and obs:
            ok = True
            ok_fore = within(obs["fore"], server_vals[0], tol)
            ok_nose = within(obs["nose"], server_vals[1], tol)
            ok_eye  = within(obs["eye"], server_vals[2], tol)
            ok = ok_fore and ok_nose and ok_eye
            match_result = (ok, ok_fore, ok_nose, ok_eye)
            color = (0,255,0) if ok else (0,0,255)
            cv2.putText(img, f"MATCH: {ok}   (F:{ok_fore} N:{ok_nose} E:{ok_eye})", (10, y0+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # periodic console print for debugging (not too spammy)
        now = time.time()
        if obs and now - last_print > 0.5:
            print(f"[OBS] fore={obs['fore']:.1f}px  nose={obs['nose']:.1f}px  eye={obs['eye']:.1f}px")
            last_print = now

        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == ord('c'):
            if obs:
                # capture current observation as server vals
                server_vals = (obs["fore"], obs["nose"], obs["eye"])
                # save to json with timestamp filename
                fname = SAVE_DIR / f"servervals_{int(time.time())}.json"
                save_server_vals(fname, {"fore": obs["fore"], "nose": obs["nose"], "eye": obs["eye"]})
                print("[CAPTURE] current obs stored as server values.")
            else:
                print("[CAPTURE] no face to capture.")
        elif k == ord('m'):
            match_mode = not match_mode
            print(f"[MODE] match_mode = {match_mode}")
        elif k == ord('t'):
            tol += 0.05
            print(f"[TOL] tol = {tol:.3f}")
        elif k == ord('y'):
            tol = max(0.01, tol - 0.05)
            print(f"[TOL] tol = {tol:.3f}")
        elif k == ord('s'):
            if server_vals:
                fname = SAVE_DIR / f"servervals_manual_{int(time.time())}.json"
                save_server_vals(fname, {"fore": server_vals[0], "nose": server_vals[1], "eye": server_vals[2]})
            else:
                print("[SAVE] no server values to save.")
        elif k == ord('l'):
            # load last saved if any
            files = sorted(SAVE_DIR.glob("servervals_*.json"))
            if files:
                sv = load_server_vals(files[-1])
                if sv:
                    server_vals = sv
                    print(f"[LOAD] loaded {files[-1]}")
                else:
                    print("[LOAD] failed to parse last file.")
            else:
                print("[LOAD] no saved server vals found.")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Bye.")

if __name__ == "__main__":
    main()
