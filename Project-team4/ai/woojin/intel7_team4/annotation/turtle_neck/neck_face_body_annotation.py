import os
import cv2
import mediapipe as mp

# 경로 설정
input_dir = "sitting_dataset"  # 입력 이미지 폴더
output_dir = './annotation_neck_face_body'       # 출력 폴더 (없으면 자동 생성)

os.makedirs(output_dir, exist_ok=True)

# mediapipe pose 세팅
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 목-귀 인덱스 (mp_pose.PoseLandmark)
LANDMARK_DICT = mp_pose.PoseLandmark
NECK_IDX = LANDMARK_DICT.NOSE.value       # mediapipe에는 "NECK"이 별도로 없음, 보통 목 대신 어깨 중앙(목 추정)이나 "NOSE" 사용
RIGHT_EAR_IDX = LANDMARK_DICT.RIGHT_EAR.value
LEFT_EAR_IDX = LANDMARK_DICT.LEFT_EAR.value
RIGHT_SHOULDER_IDX = LANDMARK_DICT.RIGHT_SHOULDER.value
LEFT_SHOULDER_IDX = LANDMARK_DICT.LEFT_SHOULDER.value

def get_neck_point(landmarks, image_width, image_height):
    # mediapipe에는 목(NECK)이 없으니, 어깨 중앙을 neck으로 씀
    right_shoulder = landmarks[RIGHT_SHOULDER_IDX]
    left_shoulder = landmarks[LEFT_SHOULDER_IDX]
    x = int((right_shoulder.x + left_shoulder.x) / 2 * image_width)
    y = int((right_shoulder.y + left_shoulder.y) / 2 * image_height)
    return x, y

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print(f"No pose detected in {filename}")
            continue

        annotated_image = image.copy()
        # 기존 mediapipe 관절 라인 그리기
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        # 목 좌표 (어깨 중앙)
        h, w, _ = image.shape
        neck_x, neck_y = get_neck_point(results.pose_landmarks.landmark, w, h)

        # 오른쪽 귀
        right_ear = results.pose_landmarks.landmark[RIGHT_EAR_IDX]
        right_ear_x = int(right_ear.x * w)
        right_ear_y = int(right_ear.y * h)

        # 왼쪽 귀
        left_ear = results.pose_landmarks.landmark[LEFT_EAR_IDX]
        left_ear_x = int(left_ear.x * w)
        left_ear_y = int(left_ear.y * h)

        # 목-오른쪽 귀 선 (빨간색)
        cv2.line(annotated_image, (neck_x, neck_y), (right_ear_x, right_ear_y), (0, 0, 255), 2)
        # 목-왼쪽 귀 선 (빨간색)
        cv2.line(annotated_image, (neck_x, neck_y), (left_ear_x, left_ear_y), (0, 0, 255), 2)

        # 저장
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, annotated_image)
        print(f"Annotated: {filename}")

print("Done! 모든 이미지에 관절 + 목-귀 라인 추가 완료.")
