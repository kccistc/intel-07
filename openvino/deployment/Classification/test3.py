import numpy as np
from openvino.runtime import Core
import cv2

# --------------------
# 모델 & 클래스 정보
# --------------------
model_xml = "model/model.xml"
model_bin = "model/model.bin"

class_labels = ["T-shirt", "loongsleevs", "hudi", "jeans", "shorts"]

# --------------------
# OpenVINO 초기화
# --------------------
ie = Core()
model = ie.read_model(model=model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# --------------------
# 이미지 전처리
# --------------------
image_path = "test.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"{image_path} not found")

# 모델 입력 크기에 맞게 리사이즈 (모델이 [?,3,224,224]라면)
image_resized = cv2.resize(image, (224, 224))
input_tensor = np.expand_dims(np.transpose(image_resized, (2, 0, 1)), 0).astype(np.float32)

# --------------------
# 추론
# --------------------
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

result = compiled_model({input_layer: input_tensor})[output_layer]

# --------------------
# 클래스 매핑
# --------------------
pred_idx = np.argmax(result)
pred_label = class_labels[pred_idx]
pred_conf = result[0][pred_idx]

print("예측 클래스:", pred_label)
print("신뢰도:", pred_conf)
