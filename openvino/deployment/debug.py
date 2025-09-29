import cv2
import numpy as np
from openvino.runtime import Core

# --------- 설정 ---------
DETECTION_MODEL = "./Detection/model/model.xml"
CLASSIFICATION_MODEL = "./Classification/model/model.xml"
IMAGE_PATH = "./sample_image.jpg"
CONF_THRESHOLD = 0.05  # Detection confidence threshold

# --------- OpenVINO 초기화 ---------
ie = Core()
det_model = ie.read_model(DETECTION_MODEL)
det_compiled = ie.compile_model(det_model, "CPU")

cls_model = ie.read_model(CLASSIFICATION_MODEL)
cls_compiled = ie.compile_model(cls_model, "CPU")

# --------- 이미지 로드 ---------
orig_image = cv2.imread(IMAGE_PATH)
orig_h, orig_w, _ = orig_image.shape

# --------- Detection 입력 전처리 ---------
input_shape = det_compiled.input(0).shape  # [1,3,H,W]
_, c, det_h, det_w = input_shape
resized_image = cv2.resize(orig_image, (det_w, det_h))
det_input = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
det_input = np.transpose(det_input, (2,0,1))  # HWC -> CHW
det_input = np.expand_dims(det_input, axis=0).astype(np.float32)

# --------- Detection 실행 ---------
det_output = det_compiled([det_input])[det_compiled.output(0)]
boxes = det_output[0]  # shape: [N,5] -> [x_c, y_c, w, h, conf]

# --------- 결과 필터링 및 원본 크기로 변환 ---------
results = []
for det in boxes:
    x_c, y_c, w, h, conf = det
    if conf < CONF_THRESHOLD:
        continue
    # 모델 기준 bbox -> 원본 이미지 좌표
    x1 = int((x_c - w/2) * orig_w / det_w)
    y1 = int((y_c - h/2) * orig_h / det_h)
    x2 = int((x_c + w/2) * orig_w / det_w)
    y2 = int((y_c + h/2) * orig_h / det_h)
    # 좌표 clip
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(orig_w-1, x2), min(orig_h-1, y2)
    results.append([x1, y1, x2, y2, conf])

# --------- Classification 실행 ---------
for i, (x1, y1, x2, y2, conf) in enumerate(results):
    crop = orig_image[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    # Classification 전처리
    cls_input_shape = cls_compiled.input(0).shape
    _, c, cls_h, cls_w = cls_input_shape
    cls_crop = cv2.resize(crop, (cls_w, cls_h))
    cls_crop = cv2.cvtColor(cls_crop, cv2.COLOR_BGR2RGB)
    cls_crop = np.transpose(cls_crop, (2,0,1))
    cls_crop = np.expand_dims(cls_crop, axis=0).astype(np.float32)
    
    cls_scores = cls_compiled([cls_crop])[cls_compiled.output(0)]
    cls_label = np.argmax(cls_scores, axis=1)[0]
    
    results[i].append(cls_label)
    results[i].append(cls_scores)

# --------- 결과 시각화 ---------
for det in results:
    x1, y1, x2, y2, conf, label, scores = det
    cv2.rectangle(orig_image, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(orig_image, f"Label:{label} Conf:{conf:.2f}", (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imwrite("result_debug.jpg", orig_image)
print("Detection + Classification 결과가 result_debug.jpg에 저장됨")
