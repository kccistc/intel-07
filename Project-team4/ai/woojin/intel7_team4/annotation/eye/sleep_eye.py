if 1:
    # drowsy_one_shot_blocking_connect_with_login.py
    # - 시작 시 서버에 연결(연결될 때까지 재시도)
    # - 연결 성공하면 로그인 문자열(SEOL_SQL) 전송하고(응답 대기)
    # - 로그인 후 얼굴/눈 감지 루프 실행
    # - 닫힘이 DESIRED_ALARM_SECS 초 이상이면 서버에 ON 문자열 전송(한 번)
    # - ON 상태에서 눈 뜨면 OFF 문자열 전송(한 번)
    # - 전송 실패 시(서버 닫힘 등) 소켓 닫고 루프 종료

    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import json
    import os
    import socket
    import sys

    # -------------------------
    # 설정 (초 단위 / 네트워크)
    CALIB_JSON = None            # example: './user_calib.json'
    USE_BOTH_EYES = True
    EAR_DEFAULT = 0.20
    REL_DROP = 0.75
    STD_K = 1.5
    ABS_MIN = 0.12
    EMA_ALPHA = 0.35

    # 판정 시간 (초)
    DESIRED_CLOSED_SECS = 0.50   # 닫힘 판정: 연속 0.5초
    DESIRED_OPEN_SECS   = 0.10   # 열림 판정: 연속 0.1초
    DESIRED_ALARM_SECS  = 5.00   # 졸림 알람: 연속 5초 이상이면 알람

    FPS_ASSUMED = 30

    # 서버 설정
    SEND_TO_SERVER = True
    SERVER_HOST = "192.168.0.158"    # <- 여기 바꿔
    SERVER_PORT = 5000               # <- 여기 바꿔
    SERVER_CONNECT_RETRY_SECS = 2.0
    SERVER_SOCKET_TIMEOUT = 3.0

    # 로그인/전송 문자열
    LOGIN_MSG = "SEOL_SQL"                   # 서버 접속 직후 보낼 로그인 문자열
    src = "AI"
    user = "seol"
    type = "SLP"
    state_on = "ON"
    state_off = "OFF"
    SERVER_MSG_ON  = f"{src}:{user}:{type}:{state_on}"
    SERVER_MSG_OFF = f"{src}:{user}:{type}:{state_off}"
    # -------------------------

    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

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

    # -------------------------
    # 단순한 blocking connect 함수 (연결될 때까지 블로킹 재시도)
    # -------------------------
    def connect_blocking(host, port, retry_secs=2.0, timeout=3.0):
        print(f"[Network] Trying to connect to {host}:{port} (will retry every {retry_secs}s until connected)...")
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((host, port))
                s.settimeout(timeout)  # keep timeout for send/recv
                print(f"[Network] Connected to {host}:{port}")
                return s
            except KeyboardInterrupt:
                print("[Network] Interrupted by user during connect. Exiting.")
                raise
            except Exception as e:
                print(f"[Network] Connect failed: {e}. Retrying in {retry_secs}s...")
                try:
                    s.close()
                except:
                    pass
                time.sleep(retry_secs)

    # -------------------------
    # 초기 설정
    # -------------------------
    if CALIB_JSON and os.path.exists(CALIB_JSON):
        THRESHOLD = compute_threshold_from_json(CALIB_JSON) or EAR_DEFAULT
        print(f"[Config] Using calibrated EAR threshold = {THRESHOLD:.4f}")
    else:
        THRESHOLD = EAR_DEFAULT
        print(f"[Config] Using default EAR threshold = {THRESHOLD:.4f}")

    # connect to server (blocking until connected) and perform login handshake
    sock = None
    if SEND_TO_SERVER:
        try:
            sock = connect_blocking(SERVER_HOST, SERVER_PORT, SERVER_CONNECT_RETRY_SECS, SERVER_SOCKET_TIMEOUT)
        except KeyboardInterrupt:
            # user aborted during connect
            sys.exit(0)

        # send LOGIN_MSG immediately, then try to read short ACK (non-blocking w/ timeout)
        try:
            #sock.sendall(LOGIN_MSG.encode('utf-8') + b'\n')
            sock.sendall(LOGIN_MSG.encode('utf-8'))
            print(f"[Network] Sent login -> {LOGIN_MSG}")
        except Exception as e:
            print(f"[Network] Failed to send login: {e}")
            try:
                sock.close()
            except:
                pass
            sys.exit(1)

        # try to receive a short response (ACK) — timeout controlled by SERVER_SOCKET_TIMEOUT
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
            print("[Network] No login ACK received (timeout). Proceeding without ACK.")
        except Exception as e:
            print(f"[Network] Error while waiting login ACK: {e}")
            try:
                sock.close()
            except:
                pass
            sys.exit(1)

    ema_ear = None
    consec_closed = 0
    consec_open = 0
    frame_count = 0
    prev_time = time.time()
    fps = FPS_ASSUMED

    closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
    open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS,   fps, FPS_ASSUMED)
    alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS,  fps, FPS_ASSUMED)

    total_consec_closed_for_alarm = 0
    alarm_sent = False   # 서버로 ON 문자열을 이미 보냈는지 플래그

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] camera open failed. Check index/permissions.")
        if sock:
            try: sock.close()
            except: pass
        sys.exit(1)

    win = "Drowsy Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        # 연결이 성공한 상태에서만 여기로 들어왔다면 감지 루프 실행
        print("[Run] Starting detection loop. Press ESC to exit. Press 'r' to reset alarm_sent.")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            frame_count += 1

            # 1초마다 FPS 갱신 및 임계 프레임 재계산
            curr = time.time()
            if curr - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = curr
                closed_thr_frames = frames_from_secs(DESIRED_CLOSED_SECS, fps, FPS_ASSUMED)
                open_thr_frames   = frames_from_secs(DESIRED_OPEN_SECS,   fps, FPS_ASSUMED)
                alarm_thr_frames  = frames_from_secs(DESIRED_ALARM_SECS,  fps, FPS_ASSUMED)

            img = cv2.flip(frame, 1)
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            raw_ear = None
            ear_for_display = None
            status_text = "NO FACE"
            status_color = (200,200,200)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                lm = results.multi_face_landmarks[0].landmark
                left_ear  = compute_ear(lm, LEFT_EYE_IDX,  h, w)
                right_ear = compute_ear(lm, RIGHT_EYE_IDX, h, w) if USE_BOTH_EYES else None

                if USE_BOTH_EYES:
                    if left_ear is not None and right_ear is not None:
                        raw_ear = (left_ear + right_ear) / 2.0
                    else:
                        raw_ear = left_ear if left_ear is not None else right_ear
                else:
                    raw_ear = left_ear if left_ear is not None else right_ear

                if raw_ear is not None:
                    ema_ear = raw_ear if ema_ear is None else (EMA_ALPHA * raw_ear + (1.0 - EMA_ALPHA) * ema_ear)
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

                    # 상태 판정 (알람 우선)
                    if total_consec_closed_for_alarm >= alarm_thr_frames:
                        status_text = "DROWSY ALERT!"
                        status_color = (0,0,255)
                    elif consec_closed >= closed_thr_frames:
                        status_text = "LEFT EYE CLOSED" if not USE_BOTH_EYES else "EYES CLOSED"
                        status_color = (0,0,255)
                    else:
                        status_text = "EYES OPEN"
                        status_color = (0,255,0)
                else:
                    status_text = "LANDMARK ERROR"
                    status_color = (100,100,100)

                # 키포인트 표시
                for i in LEFT_EYE_IDX:
                    x, y = int(lm[i].x * w), int(lm[i].y * h)
                    cv2.circle(img, (x,y), 1, (0,255,255), -1)
                for i in RIGHT_EYE_IDX:
                    x, y = int(lm[i].x * w), int(lm[i].y * h)
                    cv2.circle(img, (x,y), 1, (0,255,255), -1)
            else:
                consec_open = 0
                consec_closed = 0
                total_consec_closed_for_alarm = 0

            # -------------------------
            # ON 전송: 알람 발생 시 서버로 딱 한 번 전송 (성공하면 alarm_sent=True)
            # OFF 전송: ON 상태였고 '눈 떠짐' 감지되면 서버로 딱 한 번 전송하고 alarm_sent=False
            # 전송 실패 시(예: 서버 닫힘)에는 소켓 닫고 루프 종료
            # -------------------------
            if SEND_TO_SERVER:
                # 전송할 필요가 있는 ON 상태 (아직 안 보냈으면 전송)
                if status_text == "DROWSY ALERT!" and not alarm_sent:
                    try:
                        msg = SERVER_MSG_ON.encode('utf-8') + b'\n'
                        sock.sendall(msg)
                        alarm_sent = True
                        print(f"[Notify] Sent ON -> {SERVER_MSG_ON}")
                    except Exception as e:
                        print(f"[Notify] ON send failed: {e}")
                        try:
                            sock.close()
                        except:
                            pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break

                # ON이 이미 전송된 상태에서 사용자가 눈을 뜨면 OFF 전송
                elif alarm_sent and status_text != "DROWSY ALERT!":
                    try:
                        msg = SERVER_MSG_OFF.encode('utf-8') + b'\n'
                        sock.sendall(msg)
                        alarm_sent = False
                        print(f"[Notify] Sent OFF -> {SERVER_MSG_OFF}")
                    except Exception as e:
                        print(f"[Notify] OFF send failed: {e}")
                        try:
                            sock.close()
                        except:
                            pass
                        sock = None
                        print("[Run] Server connection lost — stopping detection loop.")
                        break

            # 오버레이(상태 + FPS 등)
            info_y = 30
            line_h = 24
            cv2.putText(img, f"Status: {status_text}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            info_y += line_h
            cv2.putText(img, f"FPS: {fps}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,50), 2)
            info_y += line_h
            if ear_for_display is not None:
                cv2.putText(img, f"EMA EAR: {ear_for_display:.3f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.putText(img, f"EMA EAR: -", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            info_y += line_h
            cv2.putText(img, f"Alarm thr: {alarm_thr_frames}f (~{DESIRED_ALARM_SECS:.0f}s)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            info_y += line_h
            send_status = "NOT_SENT" if not alarm_sent else "ALARM_SENT"
            if SEND_TO_SERVER:
                cv2.putText(img, f"Server: {SERVER_HOST}:{SERVER_PORT} [{send_status}]", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
            info_y += line_h
            cv2.putText(img, f"Closed secs (approx): {consec_closed / (fps if fps>0 else FPS_ASSUMED):.2f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC: 종료
                print("[Run] User requested exit (ESC).")
                break
            elif key == ord('r'):
                alarm_sent = False
                print("[Run] alarm_sent manually reset (you may resend on next DROWSY)")

    finally:
        # 정리
        print("[Cleanup] Releasing resources...")
        try:
            if sock:
                sock.close()
        except:
            pass
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print("[Exit] Bye.")
