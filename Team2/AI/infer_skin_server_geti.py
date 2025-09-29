import torch
from PIL import Image
import numpy as np
import json
import os
import sys
from pathlib import Path
import timm
import torch.nn as nn
import torchvision
from collections import defaultdict
import requests
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# SDK 임포트
try:
    from geti_sdk.deployment import Deployment
except ImportError:
    print("Error: geti-sdk is not installed. Please run: pip install geti-sdk==2.6.*")
    sys.exit(1)

# 피부 분석 모델 정의
class RoiMultiHead(nn.Module):
    def __init__(self, backbone="resnet50", roi_label_space=None, pretrained=True):
        super().__init__()
        assert roi_label_space is not None
        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_dim = self.backbone.num_features
        self.reg_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for roi, R_keys in self.roi_reg_keys.items():
            if len(R_keys) > 0:
                self.reg_heads[roi] = nn.Linear(in_dim, len(R_keys))
        for roi, C_keys in self.roi_cls_keys.items():
            if len(C_keys) > 0:
                heads = nn.ModuleList()
                for ck in C_keys:
                    nclass = len(self.roi_cls_maps.get(roi, {}).get(ck, {}).get("values", []))
                    heads.append(nn.Linear(in_dim, nclass))
                self.cls_heads[roi] = heads

    def forward(self, x, roi_names):
        feats = self.backbone(x)
        B = feats.size(0)
        out_reg, out_cls = [None]*B, [None]*B
        for i in range(B):
            roi = roi_names[i]
            f = feats[i:i+1]
            reg_pred = None
            if roi in self.reg_heads:
                reg_pred = self.reg_heads[roi](f)
            cls_preds = None
            if roi in self.cls_heads:
                cls_preds = [head(f) for head in self.cls_heads[roi]]
            out_reg[i], out_cls[i] = reg_pred, cls_preds
        return out_reg, out_cls

# 설정
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SERVER_URL = "http://192.168.0.90:5000/receive" # 라즈베리파이 서버 주소
GETI_DEPLOYMENT_PATH = 'geti_face_v2/deployment'
SKIN_MODEL_PATH = 'runs/skin_roi_v4/best.pth'
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Flask 앱 초기화
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 보조 함수 (모델 로딩, 분석 등)
def load_geti_deployment(deployment_path):
    print("Geti 배포 로딩 중...")
    deployment = Deployment.from_folder(deployment_path)
    print("추론 모델 로딩 중...")
    deployment.load_inference_models(device="CPU")
    return deployment

def detect_rois_with_geti(deployment, image_path):
    print(f"이미지 읽는 중: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없거나 읽을 수 없음: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Geti 모델로 추론 중...")
    prediction = deployment.infer(image_rgb)

    # Use a dictionary to store only the first (highest confidence) detection for each label
    best_rois = {}
    img_height, img_width, _ = image_rgb.shape
    image_center_x = img_width / 2

    for annotation in prediction.annotations:
        shape = annotation.shape
        box = [int(shape.x), int(shape.y), int(shape.x + shape.width), int(shape.y + shape.height)]
        label_name = annotation.labels[0].name

        if label_name == 'cheek':
            box_center_x = box[0] + (box[2] - box[0]) / 2
            final_label = 'facepart::right_cheek' if box_center_x < image_center_x else 'facepart::left_cheek'
        elif label_name == 'lip':
             final_label = 'facepart::lips'
        else:
            final_label = f"facepart::{label_name}"

        # If this label has not been seen before, add it.
        # Since Geti returns sorted predictions, the first one is the best one.
        if final_label not in best_rois:
            best_rois[final_label] = {
                'box': box,
                'label_name': final_label
            }

    # Convert the dictionary of best ROIs back to a list
    detected_rois = list(best_rois.values())
    return Image.fromarray(image_rgb), detected_rois

def load_skin_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = RoiMultiHead(backbone='convnext_tiny', roi_label_space=ckpt['roi_label_space'])
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    model.eval()
    return model, ckpt['roi_label_space'], ckpt.get('reg_std')

def analyze_skin(skin_model, roi_label_space, reg_std, cropped_image, roi_name):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((384, 384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(cropped_image).unsqueeze(0)
    with torch.no_grad():
        reg_pred, cls_pred = skin_model(image_tensor.to(DEVICE), [roi_name])
    results = {}
    reg_keys = roi_label_space['regression_keys'].get(roi_name, [])
    if reg_pred[0] is not None:
        reg_values = reg_pred[0].squeeze(0).cpu().numpy()
        for i, key in enumerate(reg_keys):
            val = reg_values[i]
            if reg_std and key in reg_std:
                mean, std = reg_std[key]['mean'], reg_std[key]['std']
                val = val * std + mean
            results[key] = float(f'{val:.2f}')
    cls_keys = roi_label_space['class_keys'].get(roi_name, [])
    cls_maps = roi_label_space['class_key_to_index'].get(roi_name, {})
    if cls_pred and cls_pred[0] is not None:
        for i, key in enumerate(cls_keys):
            logits = cls_pred[0][i]
            predicted_idx = logits.argmax(dim=1).item()
            idx_to_val_map = cls_maps.get(key, {}).get('idx_to_val', {})
            predicted_value = idx_to_val_map.get(str(predicted_idx), idx_to_val_map.get(predicted_idx, 'Error'))
            results[key] = predicted_value
    return results

def send_results_to_server(results, server_url):
    try:
        # Transform the results into the simple format expected by the Raspberry Pi
        payload = {
            "forehead": {
                "moisture": results.get("forehead", {}).get("eq_forehead_moisture", 0),
                "elasticity": results.get("forehead", {}).get("eq_forehead_elasticity_Q0", 0),
                "pigmentation": results.get("forehead", {}).get("ann_forehead_pigmentation", 0)
            },
            "l_check": {
                "moisture": results.get("left_cheek", {}).get("eq_l_cheek_moisture", 0),
                "elasticity": results.get("left_cheek", {}).get("eq_l_cheek_elasticity_Q0", 0),
                "pigmentation": results.get("left_cheek", {}).get("ann_l_cheek_pigmentation", 0),
                "pore": results.get("left_cheek", {}).get("eq_l_cheek_pore", 0)
            },
            "r_check": {
                "moisture": results.get("right_cheek", {}).get("eq_r_cheek_moisture", 0),
                "elasticity": results.get("right_cheek", {}).get("eq_r_cheek_elasticity_Q0", 0),
                "pigmentation": results.get("right_cheek", {}).get("ann_r_cheek_pigmentation", 0),
                "pore": results.get("right_cheek", {}).get("eq_r_cheek_pore", 0)
            },
            "chin": {
                "moisture": results.get("chin", {}).get("eq_chin_moisture", 0),
                "elasticity": results.get("chin", {}).get("eq_chin_elasticity_Q0", 0)
            },
            "lib": {
                "dryness": results.get("lips", {}).get("ann_lip_dryness", 0)
            }
        }

        print("\n--- 서버로 전송 ---")
        print("Payload:", json.dumps(payload, indent=2))
        response = requests.post(server_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"서버 응답 ({response.status_code}): {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n에러: 서버 연결 실패 {server_url}. 상세: {e}")
    except Exception as e:
        print(f"\n데이터 전송 중 예외 발생: {e}")

def send_detection_failure_to_server(missing_parts, server_url):
    try:
        # Remove "facepart::" prefix for the payload
        simple_missing_parts = [part.replace("facepart::", "") for part in missing_parts]
        
        # Create the desired dictionary format with the specified key
        payload = {"ROI detection failed :": ', '.join(simple_missing_parts)}

        print("\n--- 탐지 실패 전송 ---")
        print("Payload:", json.dumps(payload, indent=2))
        response = requests.post(server_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"서버 응답 ({response.status_code}): {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n에러: 서버 연결 실패 {server_url}. 상세: {e}")
    except Exception as e:
        print(f"\n실패 보고 중 예외 발생: {e}")

# 모델 로딩 (전역)
print("--- 모델 로딩 중 ---")
GETI_DEPLOYMENT = load_geti_deployment(GETI_DEPLOYMENT_PATH)
SKIN_MODEL, SKIN_LABEL_SPACE, SKIN_REG_STD = load_skin_model(SKIN_MODEL_PATH)
print("--- 모델 로딩 완료. 서버 요청 대기 중. ---")

# 메인 이미지 처리 로직
def process_image_and_get_results(image_path):
    """이미지 경로를 받아 분석 후 결과를 반환"""
    print("--- 1단계: Geti로 얼굴 ROI 탐지 ---")
    original_image, detected_rois = detect_rois_with_geti(GETI_DEPLOYMENT, image_path)
    
    expected_rois = {'facepart::forehead', 'facepart::lips', 'facepart::left_cheek', 'facepart::right_cheek', 'facepart::chin'}
    detected_labels = {r['label_name'] for r in detected_rois}
    missing_rois = expected_rois - detected_labels

    if missing_rois:
        print(f"\n경고: 다음 ROI를 탐지하지 못했습니다: {', '.join(missing_rois)}")
        return None, list(missing_rois)

    print(f"\n{len(detected_rois)}개 ROI 탐지: {[r['label_name'] for r in detected_rois]}")

    print("--- 2단계: ROI별 피부 특징 분석 ---")
    final_results = defaultdict(dict)
    for roi_info in detected_rois:
        if '::' not in roi_info['label_name']:
            continue
        roi_name = roi_info['label_name'].split('::', 1)[1]
        x1, y1, x2, y2 = roi_info['box']
        cropped_roi = original_image.crop((x1, y1, x2, y2))
        
        analysis_results = analyze_skin(SKIN_MODEL, SKIN_LABEL_SPACE, SKIN_REG_STD, cropped_roi, roi_name)
        final_results[roi_name] = analysis_results
        
    print("--- 최종 결과 ---")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    return final_results, None

# Flask API 엔드포인트
@app.route("/upload", methods=["POST"])
def analyze_image_endpoint():
    """이미지를 받아 처리하고, 결과를 파이로 전송하는 API"""
    if 'image' not in request.files:
        return jsonify({"error": "요청에 'image' 파일이 없습니다"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다"}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_image_path)
        print(f"이미지 임시 저장: {temp_image_path}")

        try:
            # 저장된 이미지 처리
            results, error_parts = process_image_and_get_results(temp_image_path)

            if error_parts:
                # 실패 메시지를 rasp.py로 전송
                send_detection_failure_to_server(error_parts, SERVER_URL)
                # 이미지 전송 클라이언트에 에러 반환
                return jsonify({
                    "status": "error",
                    "message": f"ROI 탐지 실패: {', '.join(error_parts)}"
                }), 400

            # 분석 결과를 rasp.py 서버로 전송
            send_results_to_server(results, SERVER_URL)

            # 원본 클라이언트(라즈베리파이)에 성공 응답 반환
            return jsonify({
                "status": "success",
                "message": "분석 완료 후 하드웨어 컨트롤러로 결과 전송됨.",
                "analysis": dict(results)
            }), 200

        except Exception as e:
            print(f"[에러] 예외 발생: {e}")
            return jsonify({"error": "분석 중 서버 내부 오류 발생"}), 500
        finally:
            # 임시 이미지 파일 삭제
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    return jsonify({"error": "파일 업로드 실패"}), 500


# 메인 실행 블록
if __name__ == "__main__":
    # 스크립트를 커맨드라인이 아닌 웹서버로 실행
    # 모든 네트워크 인터페이스에서 요청 수신
    # 포트 5001 사용 (rasp.py의 5000번 포트와 충돌 방지)
    print("Flask 서버 시작 중...")
    app.run(host="0.0.0.0", port=5001, debug=False)
