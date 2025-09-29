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
det_model_xml = "Detection/model/model.xml"
det_model = ie.compile_model(model=det_model_xml, device_name="CPU")
det_input_layer = det_model.input(0)
det_output_layer = det_model.output(0)

# Classification 모델 로드
cls_model_xml = "Classification/model/model.xml"
cls_model = ie.compile_model(model=cls_model_xml, device_name="CPU")
cls_input_layer = cls_model.input(0)
cls_output_layer = cls_model.output(0)

# 클래스 라벨
class_labels = ["T-shirt", "loongsleevs", "hudi", "jeans", "shorts"]

# 색 범위 (HSV)
color_ranges = {
    "Red":    [(np.array([0, 50, 50]), np.array([10, 255, 255])),
               (np.array([160, 50, 50]), np.array([179, 255, 255]))],
    "Orange": [(np.array([10, 50, 50]), np.array([25, 255, 255]))],
    "Yellow": [(np.array([25, 50, 50]), np.array([35, 255, 255]))],
    "Green":  [(np.array([36, 50, 50]), np.array([85, 255, 255]))],
    "Blue":   [(np.array([90, 50, 50]), np.array([140, 255, 255]))],
    "Purple": [(np.array([140, 50, 50]), np.array([160, 255, 255]))],
    "White":  [(np.array([0, 0, 200]), np.array([180, 50, 255]))],
    "Black":  [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
}

color_bgr = {
    "Red": (0,0,255),
    "Orange": (0,165,255),
    "Yellow": (0,255,255),
    "Green": (0,255,0),
    "Blue": (255,0,0),
    "Purple": (255,0,255),
    "White": (200,200,200),
    "Black": (0,0,0),
}

# -----------------
# 이미지 폴더 경로
# -----------------
input_folder = "sample_image"
output_folder = "result_image"
os.makedirs(output_folder, exist_ok=True)

# -----------------
# 이미지 처리
# -----------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(input_folder, filename)
    orig_image = cv2.imread(img_path)
    if orig_image is None:
        print(f"이미지 불러오기 실패: {img_path}")
        continue
    orig_h, orig_w = orig_image.shape[:2]

    # -----------------
    # Detection 전처리
    # -----------------
    det_h, det_w = det_input_layer.shape[2:]
    scale = min(det_w / orig_w, det_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(orig_image, (new_w, new_h))
    pad_w, pad_h = det_w - new_w, det_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114,114,114))
    det_input = np.expand_dims(np.transpose(padded[..., ::-1], (2,0,1)), 0).astype(np.float32)/255.0

    # -----------------
    # Detection 수행
    # -----------------
    detections = det_model({det_input_layer: det_input})[det_output_layer]

    for i, det in enumerate(detections[0]):
        x_c, y_c, w_box, h_box, conf = det
        if conf < 0.1:
            continue

        # padded -> 원본 이미지 좌표
        x_c = (x_c - left) / scale
        y_c = (y_c - top) / scale
        w_box /= scale
        h_box /= scale
        x1 = int(max(0, x_c - w_box/2))
        y1 = int(max(0, y_c - h_box/2))
        x2 = int(min(orig_w-1, x_c + w_box/2))
        y2 = int(min(orig_h-1, y_c + h_box/2))

        crop = orig_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # -----------------
        # Classification 수행
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
        max_ratio = 0
        color_detected = "Unknown"
        for color, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                m = cv2.inRange(hsv_crop, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            ratio = np.sum(mask>0) / (mask.size)
            if ratio > max_ratio and ratio > 0.01:  # 최소 1% 영역
                max_ratio = ratio
                color_detected = color

        # -----------------
        # 시각화
        # -----------------
        box_color = color_bgr.get(color_detected, (0,255,0))
        cv2.rectangle(orig_image, (x1,y1), (x2,y2), box_color, 4)  # 두껍게
        text = f"{cls_label} - {color_detected} ({conf:.2f})"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        text_x = x1 + (x2-x1 - text_w)//2
        text_y = y1 + (y2-y1 + text_h)//2
        cv2.putText(orig_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)

        print(f"[DEBUG] {filename} Det {i}: bbox=({x1},{y1},{x2},{y2}), conf={conf:.2f}, cls={cls_label}, color={color_detected}")

    # -----------------
    # 결과 저장
    # -----------------
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, orig_image)

print("모든 이미지 처리 완료. 결과는 result_image 폴더에 저장됨.")
