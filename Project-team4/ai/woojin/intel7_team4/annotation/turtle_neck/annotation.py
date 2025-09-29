import cv2
import mediapipe as mp
import os

# === 경로 설정 ===
input_dir = "./dataset/sitting_dataset"
output_dir = "annotation_face_body"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

os.makedirs(output_dir, exist_ok=True)

# sitting_dataset 안에 있는 '파일'만 가져옴 (하위 디렉토리는 제외)
for fname in os.listdir(input_dir):
    fpath = os.path.join(input_dir, fname)
    # 하위 폴더, 디렉터리 파일 무시
    if not os.path.isfile(fpath):
        continue
    # 이미지 확장자만 처리
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image = cv2.imread(fpath)
    if image is None:
        print(f"이미지 읽기 실패: {fpath}")
        continue

    with mp_pose.Pose(static_image_mode=True) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

    outpath = os.path.join(output_dir, fname)
    cv2.imwrite(outpath, image)
    print(f"저장 완료: {outpath}")

print("완료! (COCO 폴더/파일 무시, sitting_dataset의 이미지만 처리)")
