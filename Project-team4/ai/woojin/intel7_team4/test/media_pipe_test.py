# 실시간 카메라에서 관절을 잡아줌 */


# mediapipe_pose_example.py
import cv2
import csv
import mediapipe as mp
import numpy as np
from pathlib import Path

# 설정
VIDEO_SOURCE = 0           # 0 = 웹캠, 또는 "video.mp4" 같은 파일 경로
OUTPUT_CSV = "pose_landmarks.csv"
DRAW_LANDMARKS = True

# MediaPipe 준비
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 열 CSV 준비 (header: frame, landmark_index, x_pixel, y_pixel, z_norm, visibility)
csv_file = open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "landmark_index", "x_px", "y_px", "z_norm", "visibility"])

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

# Pose 초기화 (CPU). min_detection_confidence/min_tracking_confidence 조절 가능
with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV는 BGR, MediaPipe는 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 처리 (MediaPipe)
        results = pose.process(image_rgb)

        # 랜드마크가 있으면 처리
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(results.pose_landmarks.landmark):
                # lm.x/lm.y는 정규화된 좌표(0~1). 픽셀 좌표로 변환
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                csv_writer.writerow([frame_idx, i, x_px, y_px, lm.z, lm.visibility])

            # 화면에 랜드마크를 그림
            if DRAW_LANDMARKS:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )

        # 보여주기
        cv2.putText(frame, f"Frame: {frame_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.imshow("MediaPipe Pose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC로 종료
            break

        frame_idx += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()
