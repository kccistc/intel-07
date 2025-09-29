# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from openvino.runtime import Core

# -----------------
# 초기화
# -----------------
ie = Core()

# Detection 모델 로드
det_model_xml = "./Detection/model/model.xml"
det_model = ie.compile_model(model=det_model_xml, device_name="CPU")
det_input_layer = det_model.input(0)
det_output_layer = det_model.output(0)

# Classification 모델 로드
cls_model_xml = "./Classification/model/model.xml"
cls_model = ie.compile_model(model=cls_model_xml, device_name="CPU")
cls_input_layer = cls_model.input(0)
cls_output_layer = cls_model.output(0)

# 클래스 라벨 및 색상
class_labels = ["T-shirt", "loongsleevs", "hudi", "jeans", "shorts"]
class_colors = {
    "T-shirt": (0, 255, 0),
    "loongsleevs": (255, 0, 0),
    "hudi": (0, 0, 255),
    "jeans": (0, 255, 255),
    "shorts": (255, 0, 255)
}

# HSV 색 범위
color_ranges = {
    "Red":   [(np.array([0, 100, 100]), np.array([10, 255, 255])),
              (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    "Orange": [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
    "Yellow": [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    "Green": [(np.array([35, 70, 70]), np.array([85, 255, 255]))],
    "Blue":  [(np.array([85, 100, 100]), np.array([130, 255, 255]))],
    "Purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
    "White": [(np.array([0, 0, 200]), np.array([180, 50, 255]))],
    "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
}

color_bgr = {
    "Red": (0, 0, 255),
    "Orange": (0, 165, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Purple": (255, 0, 255),
    "White": (200, 200, 200),
    "Black": (0, 0, 0),
}

# -----------------
# 입력/출력 폴더
# -----------------
input_folder = "./sample_image"
output_folder = "./result_image"
os.makedirs(output_folder, exist_ok=True)
jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]

# -----------------
# 이미지 반복 처리
# -----------------
for filename in jpg_files:
    img_path = os.path.join(input_folder, filename)
    orig_image = cv2.imread(img_path)
    if orig_image is None:
        print(f"[WARN] 이미지 불러오기 실패: {img_path}")
        continue
    orig_h, orig_w = orig_image.shape[:2]

    # -----------------
    # Detection 전처리
    # -----------------
    det_h, det_w = det_input_layer.shape[2:]
    scale = min(det_w / orig_w, det_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(orig_image, (new_w, new_h))
    pad_w = det_w - new_w
    pad_h = det_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114,114,114))
    det_input = np.expand_dims(np.transpose(padded[..., ::-1], (2,0,1)), 0).astype(np.float32)/255.0

    # -----------------
    # 1단계: Detection
    # -----------------
    detections = det_model({det_input_layer: det_input})[det_output_layer]

    # 상위 3개 confidence만 선택
    confs = [det[4] for det in detections[0]]
    top_indices = np.argsort(confs)[::-1][:3]

    for i in top_indices:
        det = detections[0][i]
        x_c, y_c, w_box, h_box, conf = det
        if conf < 0.1:
            continue

        # 좌표 변환
        x_c -= left
        y_c -= top
        x_c /= scale
        y_c /= scale
        w_box /= scale
        h_box /= scale
        x1 = int(x_c - w_box/2)
        y1 = int(y_c - h_box/2)
        x2 = int(x_c + w_box/2)
        y2 = int(y_c + h_box/2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w-1, x2), min(orig_h-1, y2)

        crop = orig_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # -----------------
        # 2단계: Classification
        # -----------------
        cls_h, cls_w = cls_input_layer.shape[2:]
        crop_resized = cv2.resize(crop, (cls_w, cls_h))
        crop_input = np.expand_dims(np.transpose(crop_resized[..., ::-1], (2,0,1)), 0).astype(np.float32)/255.0
        cls_result = cls_model({cls_input_layer: crop_input})[cls_output_layer]
        cls_id = int(np.argmax(cls_result))
        cls_label = class_labels[cls_id]

        # -----------------
        # 색상 검출
        # -----------------
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color_detected = "Unknown"
        for color_name, ranges in color_ranges.items():
            mask = None
            for (lower, upper) in ranges:
                m = cv2.inRange(hsv_crop, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            if cv2.countNonZero(mask) > 0.05*crop.size/3:  # crop 면적 비율 기준
                color_detected = color_name
                break

        # -----------------
        # 시각화
        # -----------------
        box_color = class_colors.get(cls_label, (0,255,0))
        cv2.rectangle(orig_image, (x1,y1), (x2,y2), box_color, 4)

        text = f"{cls_label} - {color_detected} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x1 + (x2-x1 - text_w)//2
        text_y = y1 + (y2-y1 + text_h)//2
        cv2.putText(orig_image, text, (text_x, text_y), font, font_scale, box_color, thickness)

    # -----------------
    # 저장
    # -----------------
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, orig_image)
    print(f"[INFO] {filename} 처리 완료 -> {out_path}")
