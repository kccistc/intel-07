# 기능
# 웹캠에서
# 관절표시
# 거북목 판단 ( 왼쪽 귀, 오른쪽 귀, 코의 좌표를 잇는 중간지점과 목을 이어서 거북목을 판단)
# 정면과 측면에서는 충분한 성능이 나옴.

if 1:
    import cv2
    import mediapipe as mp
    import time
    import math

    # ====== SETTINGS (changeable) ======
    ANGLE_THRESHOLD = 15.0       # degree: 이 값 초과면 Turtle Neck 판단
    VISIBILITY_THRESH = 0.35     # landmark visibility threshold to consider a point valid
    SMOOTH_ALPHA = 0.2           # EMA alpha for smoothing head centroid (0..1). 0=no update, 1=raw
    CAM_INDEX = 0                # camera index
    # ===================================

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1,
                        static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open camera {CAM_INDEX}")

    prev_time = time.time()
    frame_count = 0
    fps = 0

    # EMA smoothed centroid (in normalized coords). Start None.
    smoothed_head = None

    winname = "Webcam: Turtle Neck Detection (centroid head-point)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, exiting.")
                break

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = now

            img = cv2.flip(frame, 1)  # mirror for natural webcam behavior
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = pose.process(img_rgb)

            used_points = 0
            centroid = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Collect candidate head points (nose, left_ear, right_ear)
                candidates = []
                # nose
                n = lm[mp_pose.PoseLandmark.NOSE.value]
                if n.visibility > VISIBILITY_THRESH:
                    candidates.append((n.x, n.y))
                # left ear
                le = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
                if le.visibility > VISIBILITY_THRESH:
                    candidates.append((le.x, le.y))
                # right ear
                re = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
                if re.visibility > VISIBILITY_THRESH:
                    candidates.append((re.x, re.y))

                used_points = len(candidates)

                if used_points > 0:
                    # centroid in normalized coords
                    sx = sum([p[0] for p in candidates]) / used_points
                    sy = sum([p[1] for p in candidates]) / used_points
                    centroid = (sx, sy)

                    # EMA smoothing
                    if smoothed_head is None:
                        smoothed_head = centroid
                    else:
                        smoothed_head = (SMOOTH_ALPHA * centroid[0] + (1 - SMOOTH_ALPHA) * smoothed_head[0],
                                        SMOOTH_ALPHA * centroid[1] + (1 - SMOOTH_ALPHA) * smoothed_head[1])

                    # neck point = midpoint of shoulders (normalized)
                    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    if l_sh.visibility > VISIBILITY_THRESH and r_sh.visibility > VISIBILITY_THRESH:
                        neck_x = (l_sh.x + r_sh.x) / 2.0
                        neck_y = (l_sh.y + r_sh.y) / 2.0
                        neck = (neck_x, neck_y)

                        # vector from neck -> head (use smoothed_head if available)
                        head_x, head_y = smoothed_head
                        vx = head_x - neck_x
                        vy = head_y - neck_y
                        norm = math.hypot(vx, vy)
                        angle_deg = 0.0
                        status = "No Angle"
                        color = (0, 255, 0)

                        if norm > 1e-6:
                            # angle between vector and vertical axis (pointing up)
                            # vertical vector (0,-1). angle = atan2(vx, -vy)
                            angle_rad = math.atan2(vx, -vy)
                            angle_deg = abs(math.degrees(angle_rad))

                            if angle_deg > ANGLE_THRESHOLD:
                                status = "Turtle Neck!"
                                color = (0, 0, 255)
                            else:
                                status = "Good"
                                color = (0, 255, 0)

                            # draw line and points on image (convert normalized to pixels)
                            hx, hy = int(head_x * w), int(head_y * h)
                            nx, ny = int(neck_x * w), int(neck_y * h)
                            cv2.line(img, (nx, ny), (hx, hy), (255, 0, 255), 3)
                            cv2.circle(img, (nx, ny), 6, (0, 255, 255), -1)
                            cv2.circle(img, (hx, hy), 6, (255, 0, 0), -1)

                            # also draw small markers for each used landmark
#                            for (px, py) in candidates:
#                                cx, cy = int(px * w), int(py * h)
#                                cv2.circle(img, (cx, cy), 4, (0, 200, 200), -1)

                            # top-left status text: status + angle + used count (English only)
                            text = f"{status}  angle={angle_deg:.1f}deg  used={used_points}"
                            cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        else:
                            cv2.putText(img, "Head-Neck too close", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                    else:
                        cv2.putText(img, "Shoulders not visible", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                else:
                    cv2.putText(img, "Head landmarks not visible", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)
            else:
                cv2.putText(img, "No pose detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)

            # FPS
            cv2.putText(img, f"FPS: {fps}", (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)

            cv2.imshow(winname, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

