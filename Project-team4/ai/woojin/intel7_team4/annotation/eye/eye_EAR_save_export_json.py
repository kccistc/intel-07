"""
얼굴 인증, 등록용 코드
"""

if 0:
    # calibrate_eyes.py
    # Usage: python3 calibrate_eyes.py
    # 선택: 1 (glasses flow) or 2 (no glasses)
    # 캡처 시간 등은 아래 DEFAULTS에서 조정 가능

    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import json
    import os
    from statistics import mean, median, pstdev

    # ====== Defaults (원하면 여기 값만 수정) ======
    CALIB_SECONDS = 30           # 한 세션 캘리브레이션 초 (기본 30s)
    REST_SECONDS = 10            # 안경 ON/OFF 사이 쉬는 시간
    OUTPUT_JSON = "baseline_capture.json"
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    # ==============================================

    def calc_EAR(landmarks, idx, img_shape):
        h, w = img_shape
        p = [(landmarks[i].x, landmarks[i].y) for i in idx]
        # normalized coords -> use relative distances (no conversion needed)
        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        if C == 0:
            return None
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_center(landmarks, idx):
        xs = [landmarks[i].x for i in idx]
        ys = [landmarks[i].y for i in idx]
        return (mean(xs), mean(ys))

    def capture_session(cap, face_mesh, duration_s, verbose=False):
        start = time.time()
        samples = []
        frame_cnt = 0
        prev_time = start
        fps_cnt = 0
        fps = 0
        while True:
            if time.time() - start >= duration_s:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            fps_cnt += 1
            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            results = face_mesh.process(img_rgb)
            ear_mean = None
            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                lm = mesh.landmark
                left_ear = calc_EAR(lm, LEFT_EYE_IDX, (h, w))
                right_ear = calc_EAR(lm, RIGHT_EYE_IDX, (h, w))
                if left_ear is not None and right_ear is not None:
                    ear_mean = float((left_ear + right_ear) / 2.0)
                    samples.append(ear_mean)

                # draw some UI
                # show points for left eye (small)
                for i in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    cx, cy = int(lm[i].x * w), int(lm[i].y * h)
                    cv2.circle(img, (cx, cy), 1, (0,255,0), -1)

                # draw EAR on frame
                if ear_mean is not None:
                    cv2.putText(img, f"EAR: {ear_mean:.3f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                cv2.putText(img, "NO FACE", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,128,128), 2)

            # fps calc (update every 1s)
            if time.time() - prev_time >= 1.0:
                fps = fps_cnt
                fps_cnt = 0
                prev_time = time.time()
            cv2.putText(img, f"FPS: {fps}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50), 2)
            # progress bar/time
            elapsed = time.time() - start
            remain = max(0, duration_s - elapsed)
            cv2.putText(img, f"Time left: {int(remain)}s", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Calibration - look at camera", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        return samples, fps

    def summarize(samples, fps, img_size=None):
        if not samples:
            return None
        s_mean = float(mean(samples))
        s_median = float(median(samples))
        s_std = float(pstdev(samples)) if len(samples) > 1 else 0.0
        # interocular distance estimate if img_size provided: use last frame's landmark centers (normalized)
        iod = None
        return {
            "mean": s_mean,
            "median": s_median,
            "std": s_std,
            "count": len(samples),
            "fps_at_capture": int(fps),
            "samples": None  # avoid storing all raw samples by default (set None), can change
        }

    def main():
        print("캘리브레이션 스크립트")
        print("옵션을 선택하세요:")
        print("1) 안경 착용 흐름(30초간 안경 착용 상태 저장 -> 10초간 쉬고 -> 30-초간 안경 미착용 상태 저장)")
        print("2) 안경 미착용 흐름(안경 없음, 30초간 한 번만 저장)")
        choice = input("선택지를 입력하세요 (1 또는 2): ").strip()
        if choice not in ("1","2"):
            print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                        refine_landmarks=True, min_detection_confidence=0.5)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        result_data = {
            "created_at": time.time(),
            "flow": "glasses" if choice=="1" else "no_glasses",
            "with_glasses": None,
            "without_glasses": None
        }

        try:
            if choice == "1":
                input("Put ON your glasses and press ENTER to start 1st capture (ON) ...")
                print(f"Starting capture for {CALIB_SECONDS}s (glasses ON). Look at camera.")
                samples_on, fps_on = capture_session(cap, face_mesh, CALIB_SECONDS)
                summary_on = summarize(samples_on, fps_on)
                if summary_on:
                    result_data["with_glasses"] = summary_on
                print("Finished ON capture.")

                print(f"Take {REST_SECONDS}s rest. Remove/put off glasses if you will.")
                for t in range(REST_SECONDS, 0, -1):
                    print(f"...{t}s", end="\r"); time.sleep(1)
                input("\nPut OFF your glasses and press ENTER to start 2nd capture (OFF) ...")
                print(f"Starting capture for {CALIB_SECONDS}s (glasses OFF). Look at camera.")
                samples_off, fps_off = capture_session(cap, face_mesh, CALIB_SECONDS)
                summary_off = summarize(samples_off, fps_off)
                if summary_off:
                    result_data["without_glasses"] = summary_off
                print("Finished OFF capture.")

            else:  # choice == "2"
                input("Make sure you are without glasses. Press ENTER to start capture ...")
                print(f"Starting capture for {CALIB_SECONDS}s. Look at camera.")
                samples, fps = capture_session(cap, face_mesh, CALIB_SECONDS)
                summary = summarize(samples, fps)
                if summary:
                    result_data["without_glasses"] = summary

            # add small metadata and save to file
            result_data["camera_index"] = 0
            result_data["calib_seconds"] = CALIB_SECONDS
            out_fname = OUTPUT_JSON
            # if file exists, add timestamp to avoid overwrite
            if os.path.exists(out_fname):
                base, ext = os.path.splitext(out_fname)
                out_fname = f"{base}_{int(time.time())}{ext}"
            with open(out_fname, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"Saved baseline JSON -> {out_fname}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            face_mesh.close()

    if __name__ == "__main__":
        main()

if 1:
    # eye_EAR_save_export_json.py (updated prompts, camera detection)
    # Run: python3 eye_EAR_save_export_json.py
    # Requires: mediapipe, opencv-python, numpy

    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import json
    import os
    from datetime import datetime

    # ----------------- CONFIG -----------------
    OUT_BASE = "./json"         # base dir to store per-user folders
    CAM_INDEX_DEFAULT = 0       # default camera index
    CALIB_SECONDS_DEFAULT = 30  # default calibration seconds for each session
    REST_SECONDS = 10           # rest seconds between glass/no-glass sessions
    SAMPLE_SAVE_LIMIT = 500     # downsample saved samples to this count (None -> keep all)
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    FPS_WARMUP_SECONDS = 1.0
    MAX_CAMERA_CHECK = 6        # check camera indices 0..MAX_CAMERA_CHECK-1
    # ------------------------------------------

    mp_face = mp.solutions.face_mesh

    def ensure_user_dir(username):
        user_dir = os.path.join(OUT_BASE, username)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def save_json(path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def calc_ear(landmarks, idx, h, w):
        try:
            pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
        except Exception:
            return None
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-8
        ear = (A + B) / (2.0 * C)
        return float(ear)

    def detect_available_cameras(max_check=MAX_CAMERA_CHECK):
        avail = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap is None:
                continue
            opened = cap.isOpened()
            cap.release()
            if opened:
                avail.append(i)
        return avail

    def calibrate_ear(duration_seconds=30, cam_index=0, window_title="Calibration"):
        face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                    refine_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"[ERROR] 카메라 {cam_index} 열기 실패. 다른 인덱스를 시도해주세요.")
            face_mesh.close()
            return None

        start_ts = time.time()
        frame_count = 0
        samples = []
        last_show = 0
        warmup_end = start_ts + FPS_WARMUP_SECONDS
        fps_est = 30.0

        print(f">> {duration_seconds}초간 캘리브레이션을 시작합니다. (ESC 누르면 중단)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_count += 1
                now = time.time()
                elapsed = now - start_ts
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                ear_val = None
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    lm = results.multi_face_landmarks[0].landmark
                    left = calc_ear(lm, LEFT_EYE_IDX, h, w)
                    right = calc_ear(lm, RIGHT_EYE_IDX, h, w)
                    if left is not None and right is not None:
                        ear_val = float((left + right) / 2.0)
                    elif left is not None:
                        ear_val = float(left)
                    elif right is not None:
                        ear_val = float(right)
                samples.append({"t": now, "ear": ear_val})

                # fps estimate after warmup
                if now > warmup_end:
                    fps_est = max(1.0, frame_count / (now - start_ts))

                # UI update ~20 FPS cap
                if now - last_show > 0.05:
                    disp = frame.copy()
                    info_y = 28
                    cv2.putText(disp, f"{window_title}: {int(elapsed)}/{int(duration_seconds)}s (ESC to cancel)", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(disp, f"Samples: {len(samples)}  FPS~{fps_est:.1f}", (10, info_y+26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    if ear_val is not None:
                        cv2.putText(disp, f"EAR: {ear_val:.3f}", (10, info_y+52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow(window_title, disp)
                    last_show = now

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print(">> 캘리브레이션을 사용자가 중단했습니다 (ESC).")
                    face_mesh.close()
                    cap.release()
                    cv2.destroyWindow(window_title)
                    return None

                if elapsed >= duration_seconds:
                    break
        finally:
            face_mesh.close()
            cap.release()
            cv2.destroyWindow(window_title)

        # compute stats
        ear_vals = [s["ear"] for s in samples if s["ear"] is not None]
        count = len(ear_vals)
        if count > 0:
            mean = float(np.mean(ear_vals))
            median = float(np.median(ear_vals))
            std = float(np.std(ear_vals, ddof=0))
        else:
            mean = median = std = None

        # downsample stored samples to avoid huge json
        stored_samples = None
        if SAMPLE_SAVE_LIMIT is None:
            stored_samples = samples
        else:
            if len(samples) <= SAMPLE_SAVE_LIMIT:
                stored_samples = samples
            else:
                step = max(1, len(samples) // SAMPLE_SAVE_LIMIT)
                stored_samples = samples[::step]

        summary = {
            "mean": mean,
            "median": median,
            "std": std,
            "count": count,
            "fps_at_capture": fps_est,
            "samples": stored_samples
        }
        return summary

    def rest_countdown(seconds):
        print(f">> {seconds}초 휴식합니다...")
        for i in range(seconds, 0, -1):
            print(f"  {i}...", end="", flush=True)
            time.sleep(1)
            print("\r", end="", flush=True)
        print("  (휴식 종료)\n")

    def input_int_with_default(prompt, default):
        while True:
            s = input(prompt).strip()
            if s == "":
                return default
            try:
                v = int(s)
                return v
            except ValueError:
                print("  숫자를 입력하거나 Enter로 기본값을 사용하세요.")

    def main():
        print("=== EAR Calibration Multi-mode ===")
        username = input("사용자 이름을 입력하세요: ").strip()
        if not username:
            print("사용자 이름이 필요합니다. 종료합니다.")
            return

        # create user dir
        user_dir = ensure_user_dir(username)

        # ask glasses
        ans = input("안경을 착용하고 있습니까? (y/N) : ").strip().lower()
        wearing_glasses = ans == 'y' or ans == 'yes'

        # durations: show default and allow Enter
        print(f"\n캘리브레이션 시간 기본값은 {CALIB_SECONDS_DEFAULT}초 입니다.")
        dur = input_int_with_default(f"캘리브레이션 시간(초)을 입력하거나 Enter를 눌러 기본값({CALIB_SECONDS_DEFAULT}) 사용: ", CALIB_SECONDS_DEFAULT)

        # camera selection: detect available cameras and show
        avail = detect_available_cameras()
        if len(avail) > 0:
            print(f"\n발견된 사용 가능한 카메라 인덱스: {avail}")
        else:
            print("\n주의: 사용 가능한 카메라가 자동으로 발견되지 않았습니다. (기본 인덱스: 0 사용 시도)")

        cam_idx = input_int_with_default(f"카메라 인덱스 입력 (Enter로 기본값 {CAM_INDEX_DEFAULT} 사용): ", CAM_INDEX_DEFAULT)

        timestamp = time.time()
        created_at = timestamp

        if wearing_glasses:
            input("\n준비되면 Enter를 눌러 '안경 착용' 상태 캘리브레이션을 시작하세요...")
            summary_glass = calibrate_ear(duration_seconds=dur, cam_index=cam_idx, window_title="Calib (with glasses)")
            if summary_glass is None:
                print("안경 캘리브레이션이 중단되었습니다. 저장하지 않습니다.")
                return

            print(f"\n{REST_SECONDS}초간 쉬세요 (안경을 벗을 준비)...")
            rest_countdown(REST_SECONDS)

            input("준비되면 Enter를 눌러 '안경 미착용' 상태 캘리브레이션을 시작하세요...")
            summary_noglass = calibrate_ear(duration_seconds=dur, cam_index=cam_idx, window_title="Calib (no glasses)")
            if summary_noglass is None:
                print("안경 미착용 캘리브레이션이 중단되었습니다. 이미 수집한 데이터만 저장합니다 (선택적으로).")

            # Save two JSONs (if available)
            if summary_glass is not None:
                out_glass = {
                    "created_at": created_at,
                    "flow": "with_glasses",
                    "summary": summary_glass,
                    "camera_index": cam_idx,
                    "calib_seconds": dur
                }
                path_glass = os.path.join(user_dir, f"{username}_glass.json")
                save_json(path_glass, out_glass)
                print(f"Saved: {path_glass}")
            if summary_noglass is not None:
                out_noglass = {
                    "created_at": created_at,
                    "flow": "without_glasses",
                    "summary": summary_noglass,
                    "camera_index": cam_idx,
                    "calib_seconds": dur
                }
                path_noglass = os.path.join(user_dir, f"{username}_no_glass.json")
                save_json(path_noglass, out_noglass)
                print(f"Saved: {path_noglass}")

            print("\n완료.")
        else:
            input("\n준비되면 Enter를 눌러 안경 미착용 상태(또는 기본 상태) 캘리브레이션을 시작하세요...")
            summary = calibrate_ear(duration_seconds=dur, cam_index=cam_idx, window_title="Calib (no glasses)")
            if summary is None:
                print("캘리브레이션이 중단되었습니다. 저장하지 않습니다.")
                return

            out = {
                "created_at": created_at,
                "flow": "without_glasses",
                "summary": summary,
                "camera_index": cam_idx,
                "calib_seconds": dur
            }
            path = os.path.join(user_dir, f"{username}_no_glass.json")
            save_json(path, out)
            print(f"Saved: {path}")
            print("\n완료.")

    if __name__ == "__main__":
        main()
