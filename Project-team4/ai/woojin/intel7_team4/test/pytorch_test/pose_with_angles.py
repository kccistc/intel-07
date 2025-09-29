# pose_with_angles.py
import cv2
import math
import numpy as np
from ultralytics import YOLO

# 📌 모델 경로와 카메라 소스
model_path = '/home/jetson/intel7_team4/test/pytorch_test/best.pt'
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

# 📐 각도 계산 함수
def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

# 🔗 연결할 keypoint 쌍 (skeleton)
skeleton = [
    ('left_ear', 'left_eye'), ('left_eye', 'nose'), ('nose', 'right_eye'),
    ('right_eye', 'right_ear'), ('nose', 'neck1'),
    ('left_shoulder', 'left_arm'), ('right_shoulder', 'right_arm'),
    ('left_shoulder', 'neck2'), ('right_shoulder', 'neck2'),
    ('neck1', 'neck2'), ('neck2', 'back1'), ('back1', 'back2'), ('back2', 'waist'),
    ('waist', 'left_hip'), ('waist', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

# 🔢 YOLOv8 기반 keypoint 인덱스 매핑 (사용자 모델에 맞게 조정하세요)
keypoint_index = {
    'neck1': 0,
    'neck2': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_arm': 4,
    'right_arm': 5,
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

# 📏 측정할 각도 타겟들 (자세 분석)
angle_targets = {
    'Neck bent': ('neck1', 'neck2', 'back1'),
    'Back bent': ('neck2', 'waist', 'back2'),
    'Leg twist': ('left_hip', 'right_hip', 'right_knee')
}

# 사람 여러명일 때, 프레임 중앙에 가장 가까운 사람 인덱스 선택 헬퍼
def get_center_person_index(result, frame_width):
    """
    result: ultralytics result for one frame (results[0])
    frame_width: frame width in pixels
    """
    # 안전 체크
    if result is None:
        return None
    # prefer keypoints if available
    kps_obj = getattr(result, "keypoints", None)
    if kps_obj is None:
        # fallback: try boxes
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        # use box centers
        centers = []
        for b in boxes:
            try:
                # b.xyxy or b.xywh? try xyxy
                xyxy = b.xyxy[0].cpu().numpy() if hasattr(b, "xyxy") else None
            except Exception:
                xyxy = None
            if xyxy is not None:
                cx = (xyxy[0] + xyxy[2]) / 2.0
                centers.append(cx)
        if not centers:
            return 0
        frame_center = frame_width / 2.0
        idx = int(np.argmin([abs(c - frame_center) for c in centers]))
        return idx

    # keypoints present
    # keypoints.data is expected: (n_persons, k*3) or similar; try safe extraction
    try:
        data = kps_obj.data  # tensor maybe
    except Exception:
        # some versions: kps_obj.xy or kps_obj.xys
        try:
            data = kps_obj.xyn  # try normalized
        except Exception:
            return 0

    try:
        # data shape (n, k, 3) or (n, k*3)
        arr = data.cpu().numpy()
        if arr.ndim == 3:
            persons = arr.shape[0]
            mean_x = []
            for i in range(persons):
                # arr[i,:,0] are x coords
                xs = arr[i,:,0]
                mean_x.append(xs.mean())
        elif arr.ndim == 2:
            # flatten form: (n, k*3)
            persons = arr.shape[0]
            k3 = arr.shape[1]
            k = k3 // 3
            mean_x = []
            for i in range(persons):
                xs = arr[i, 0::3]
                mean_x.append(xs.mean())
        else:
            return 0
        # convert normalized -> pixel
        mean_x_px = [mx * frame_width for mx in mean_x]
        frame_center = frame_width / 2.0
        idx = int(np.argmin([abs(mx - frame_center) for mx in mean_x_px]))
        return idx
    except Exception:
        return 0

# 🔍 실시간 분석 루프 (수정: 들여쓰기 고침)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 모델에 프레임 전달 (pose 모델이라면 결과에 keypoints 포함)
    # model(frame) -> list of Results; take first
    try:
        results = model(frame)  # returns list-like
        if results is None or len(results) == 0:
            cv2.imshow('Posture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        res0 = results[0]
    except Exception as e:
        print("model inference error:", e)
        break

    frame_width = frame.shape[1]
    target_idx = get_center_person_index(res0, frame_width)

    pose_kps = None
    # 안전하게 keypoints 추출
    if hasattr(res0, "keypoints") and getattr(res0, "keypoints") is not None:
        try:
            # .data expected
            pose_kps = res0.keypoints.data  # tensor (n, k, 3) or (n, k*3)
        except Exception:
            try:
                pose_kps = res0.keypoints.cpu().numpy()
            except Exception:
                pose_kps = None

    if pose_kps is not None:
        # get chosen person's keypoints as numpy (k,3)
        try:
            arr = pose_kps.cpu().numpy() if hasattr(pose_kps, "cpu") else np.array(pose_kps)
            # normalize shape
            if arr.ndim == 3:
                person_kp = arr[target_idx]
            elif arr.ndim == 2:
                # (n, k*3) -> reshape
                k = arr.shape[1] // 3
                person_kp = arr[target_idx].reshape(k, 3)
            else:
                person_kp = None
        except Exception:
            person_kp = None
    else:
        person_kp = None

    def get_point(name):
        idx = keypoint_index.get(name)
        if person_kp is None or idx is None or idx >= person_kp.shape[0]:
            return None
        x, y, conf = person_kp[idx]
        if conf <= 0.3:
            return None
        return int(x), int(y)

    # ✅ 선 연결
    for kp1, kp2 in skeleton:
        pt1, pt2 = get_point(kp1), get_point(kp2)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # ✅ 각도 측정 (deg 표시)
    y_offset = 30
    for label, (a, b, c) in angle_targets.items():
        ang = calculate_angle(get_point(a), get_point(b), get_point(c))
        if ang is not None:
            color = (0, 255, 0)
            if label == 'Neck bent' and ang < 30:
                color = (0, 0, 255)
            if label == 'Back bent' and ang < 40:
                color = (0, 0, 255)
            if label == 'Leg twist' and ang > 150:
                color = (0, 0, 255)
            cv2.putText(frame, f'{label}: {int(ang)} deg', (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30

    # 🖼️ 영상 출력
    cv2.imshow('Posture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
