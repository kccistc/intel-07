import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ✅ 학습 시 사용한 모델 파일 이름으로 정확하게 수정합니다.
from models.vit_detection_pretrained import VisionTransformerDetection
from utils import get_device, load_classes

# -----------------------------
# 1. 설정 및 모델 로드
# -----------------------------
TEST_IMAGE_PATH = "/home/ubuntu/intel_ai_project/received_images/latest.jpg"

device = get_device()
model_path = "vit_det_best_map_multi.pth"

classes = load_classes()  # classes.json에서 불러옴
num_classes = len(classes)  # foreground 클래스 수

# 모델 초기화 및 가중치 로드
model = VisionTransformerDetection(num_classes=num_classes, num_queries=100).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ 학습된 모델 가중치 로드 완료")
else:
    print("❌ 모델 가중치 파일이 없습니다. train.py를 먼저 실행하세요.")
    exit()

model.eval()

# -----------------------------
# 2. 이미지 전처리 및 예측 함수
# -----------------------------
def preprocess_image(image_path, img_size=224):
    # OpenCV로 이미지 읽기
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None
    
    # BGR에서 RGB로 변환 (PIL, PyTorch와 일치)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # PyTorch 모델 입력에 맞게 전처리
    pil_img = Image.fromarray(rgb_img)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(pil_img).to(device)
    
    return img_tensor, original_img

def predict(image, original_size):
    with torch.no_grad():
        outputs = model([image])

    logits = outputs["pred_logits"][0].cpu()  # [Q, C+1]
    boxes = outputs["pred_boxes"][0].cpu()    # [Q, 4]

    probs = torch.softmax(logits, dim=-1)
    scores, labels = probs.max(-1)

    width, height = original_size
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    # cxcywh → xyxy 변환
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2
    final_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    # 배경 제외 + 점수 필터
    keep = (scores > 0.5) & (labels < num_classes)
    return final_boxes[keep], scores[keep], labels[keep]

def get_dominant_color(image_crop, k=5):
    """
    주어진 이미지 영역에서 K-평균 군집화를 통해 가장 지배적인 색상을 추출합니다.
    (배경으로 추정되는 무채색을 필터링하는 로직 추가)
    """
    if image_crop.size == 0:
        return (0, 0, 0)
        
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    try:
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    except cv2.error:
        print("K-means clustering failed. Returning default color.")
        return (0, 0, 0)

    # 각 클러스터의 픽셀 수를 계산
    counts = np.bincount(labels.flatten())
    sorted_indices = np.argsort(counts)[::-1]
    total_pixels = len(labels)
    
    # 가장 큰 클러스터의 색상과 크기를 확인
    dominant_rgb = centers[sorted_indices[0]]
    dominant_bgr = tuple(dominant_rgb.astype(int)[::-1])
    dominant_count = counts[sorted_indices[0]]
    
    # 무채색 클러스터인지 판단하는 함수
    def is_achromatic(color):
        r, g, b = color
        return abs(r - g) < 25 and abs(r - b) < 25 and abs(g - b) < 25

    # 가장 큰 클러스터가 무채색이고, 전체 픽셀의 60% 이상을 차지하면 배경으로 간주
    if is_achromatic(dominant_rgb) and dominant_count / total_pixels > 0.6:
        if len(sorted_indices) > 1:
            # 두 번째로 큰 클러스터의 색상을 선택
            second_dominant_rgb = centers[sorted_indices[1]]
            second_dominant_bgr = tuple(second_dominant_rgb.astype(int)[::-1])
            return second_dominant_bgr
        else:
            # 두 번째 클러스터가 없으면 그냥 원래 대표색 반환
            return dominant_bgr
    
    # 그 외의 경우 (유채색이거나, 무채색이라도 면적이 작을 경우)
    return dominant_bgr

def get_color_category(bgr_color):
    b, g, r = bgr_color
    
    # 단순한 임계값 기반 색상 분류
    if r > 100 and g < 50 and b < 50:
        return "빨강"
    if g > 100 and r < 50 and b < 50:
        return "초록"
    if b > 100 and r < 50 and g < 50:
        return "파랑"
    if r > 100 and g > 100 and b < 50:
        return "노랑"
    if b > 100 and g > 100 and r < 50:
        return "시안"
    if b > 100 and r > 100 and g < 50:
        return "자홍"

    # 무채색 분류
    if abs(r - g) < 25 and abs(r - b) < 25 and abs(g - b) < 25:
        if r < 50:
            return "검정"
        if r > 200:
            return "흰색"
        return "회색"
        
    return "기타"

# -----------------------------
# 3. 실행 예시
# -----------------------------
if __name__ == "__main__":
    img_tensor, original_cv_img = preprocess_image(TEST_IMAGE_PATH)

    if img_tensor is None:
        exit()

    original_size = (original_cv_img.shape[1], original_cv_img.shape[0])
    boxes, scores, labels = predict(img_tensor, original_size)

    print("-" * 50)
    print(f"✅ 예측 결과 ({TEST_IMAGE_PATH})")

    if len(boxes) == 0:
        print("객체가 감지되지 않았습니다.")
    else:
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            label_name = classes[label.item()]
            confidence = score.item()
            
            cropped_img = original_cv_img[int(y1):int(y2), int(x1):int(x2)]

            dominant_color = get_dominant_color(cropped_img)
            color_category = get_color_category(dominant_color)

            cv2.rectangle(original_cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            text = f"{label_name}: {confidence:.2f}"
            color_text = f"Color: {color_category}"
            
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            color_text_size, _ = cv2.getTextSize(color_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(original_cv_img, (int(x1), int(y1) - text_size[1] - 10), (int(x1) + text_size[0], int(y1)), (0, 255, 0), -1)
            cv2.rectangle(original_cv_img, (int(x1), int(y1) - text_size[1] - color_text_size[1] - 15), (int(x1) + color_text_size[0], int(y1) - text_size[1] - 5), (0, 255, 0), -1)

            cv2.putText(original_cv_img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(original_cv_img, color_text, (int(x1), int(y1) - text_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            print(f" - 클래스: {label_name}, 신뢰도: {confidence:.2f}, 박스: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], 색상 카테고리: {color_category}")

    cv2.imshow("Detection Result", original_cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ 결과 이미지가 화면에 표시되었습니다.")
    print("-" * 50)